// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Several sandbox instances in one supervisor process.
//!
//! A sandbox instance is one running generation of a stable Sandbox
//! resource. Gateway transport connections, and the gateway supervisor
//! sessions carried over them, may reconnect within one instance without
//! creating another.
//!
//! Every hosted instance runs its tasks on one multi-thread runtime owned by
//! the host, separate from the host's control-plane runtime, so instance load
//! cannot starve the reconciler, the gateway attachment, or the health
//! listener. Each instance spawns through an [`InstanceSpawner`], which tracks
//! every task in the instance's tree, ends it at its next await point when the
//! instance is cancelled, and keeps it force-reapable, so stopping an instance
//! never leaves work behind. The instance's
//! [`SandboxInstanceBindings`] (OCSF context, OCSF JSON settings, log sink,
//! denial log, and gateway credentials) travel with its spawner, so every task
//! it starts can enter the instance's per-crate identity scopes. Gateway
//! credentials are never bound to threads; each instance passes its own
//! explicitly. Stopping an instance cancels and joins its task tree, joins the
//! blocking work it started, and revokes its credentials.
//!
//! The runtime's workers are bounded by [`INSTANCE_HOST_WORKER_THREADS`] and
//! its blocking pool by [`INSTANCE_HOST_MAX_BLOCKING_THREADS`], so a
//! supervisor's thread cost is a stated ceiling instead of a runtime per
//! instance. The blocking ceiling is deliberately far above the worker count:
//! the pool is shared by every instance, so a small bound would let one busy
//! sandbox's blocking work starve another's. Idle blocking threads cost
//! nothing (tokio retires them after about ten seconds), and the loaded
//! ceiling is the worker count plus that pool. Blocking work already running
//! on the pool cannot be aborted, so removal joins it within the same bound
//! instead of force-reaping it. Blocking work still queued when an instance
//! ends is skipped.

use std::collections::HashMap;
use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, Weak};
use std::time::Duration;

use futures::FutureExt as _;
use miette::{IntoDiagnostic as _, Result, WrapErr as _};
use openshell_core::grpc_client::GatewayCredentials;
use openshell_core::instance_tasks::{
    InstanceTaskBackend, InstanceTasks, ScopedBlockingOutput, ScopedBlockingTask, ScopedTask,
};
use openshell_ocsf::OcsfJsonSettings;
use openshell_supervisor_control::{SandboxInstancePhaseReport, SandboxInstanceStatusReport};
use openshell_supervisor_network::policy_local::{DenialLogSource, SandboxInstanceDenialLog};
use openshell_supervisor_process::log_push::LogSink;
use tokio::sync::watch;
use tokio::task::{AbortHandle, JoinHandle};
use tokio_util::sync::CancellationToken;
use tokio_util::task::TaskTracker;

use crate::{SandboxInstanceBindings, SandboxInstanceControls};

/// Worker threads of the runtime that hosts every sandbox instance, capped by
/// the machine's parallelism.
pub const INSTANCE_HOST_WORKER_THREADS: usize = 4;
/// Ceiling of the shared blocking pool, stated instead of inherited from
/// tokio's default.
///
/// Every instance shares this pool, so a bound near the worker count would let
/// one CPU-heavy sandbox's blocking work (for example its egress
/// authorization) occupy every slot and starve another's. The pool is still
/// bounded, and its threads exit when idle, so the ceiling costs nothing at
/// rest. Per-instance fairness inside the pool is a follow-up.
pub const INSTANCE_HOST_MAX_BLOCKING_THREADS: usize = 256;
/// Bound on joining an instance's task tree after it is told to stop.
pub const RUNTIME_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(5);
/// Abort passes removal runs over the force-reap registry.
///
/// The first pass can race a task that registers while it runs, so removal
/// repeats the pass after each join window. A spawn storm during teardown is
/// bounded by this count instead of delaying removal indefinitely.
const REAP_ABORT_PASSES: usize = 3;
/// Rounds removal runs joining outstanding blocking closures.
///
/// A closure registered while earlier ones are joined is covered by the next
/// round; a spawn storm during teardown is bounded by this count instead of
/// delaying removal indefinitely.
const REAP_BLOCKING_ROUNDS: usize = 3;

/// Build the runtime that hosts every sandbox instance of this process.
///
/// One shared multi-thread runtime serves every instance. It is separate from
/// the host's control-plane runtime so instance load cannot starve the
/// reconciler, the gateway attachment, or the health listener. Workers are
/// bounded by [`INSTANCE_HOST_WORKER_THREADS`] and the machine's parallelism,
/// and the blocking pool by [`INSTANCE_HOST_MAX_BLOCKING_THREADS`] instead of
/// tokio's per-runtime default of 512.
///
/// # Errors
///
/// Fails when the runtime cannot be built.
pub fn instance_host_runtime() -> std::io::Result<tokio::runtime::Runtime> {
    let parallelism = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
    let workers = INSTANCE_HOST_WORKER_THREADS.min(parallelism);
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(workers)
        .max_blocking_threads(INSTANCE_HOST_MAX_BLOCKING_THREADS)
        .thread_name("openshell-instance-worker")
        .enable_all()
        .build()
}

tokio::task_local! {
    /// Denial log of the hosted sandbox instance whose task is being polled.
    static TASK_DENIAL_LOG: SandboxInstanceDenialLog;
}

/// Run `future` with its task's denials recorded in `log`.
async fn scope_denial_log<F: Future>(log: SandboxInstanceDenialLog, future: F) -> F::Output {
    TASK_DENIAL_LOG.scope(log, future).await
}

/// Run the synchronous `f` with its denials recorded in `log`.
///
/// Tokio task scopes do not reach blocking closures, so an instance's
/// `spawn_blocking` work re-enters the scope with this variant.
fn scope_denial_log_sync<F, T>(log: SandboxInstanceDenialLog, f: F) -> T
where
    F: FnOnce() -> T,
{
    TASK_DENIAL_LOG.sync_scope(log, f)
}

/// Shorthand-log destination for hosted sandbox instances' recent denials.
///
/// Each line goes to the denial log of the instance whose task emits it,
/// through the task scope [`scope_denial_log`] enters. Lines from work outside
/// every instance's task scope are dropped, so no instance can read another's
/// denials through `policy.local`.
#[derive(Debug, Default, Clone, Copy)]
pub struct SandboxInstanceDenialWriter;

impl std::io::Write for SandboxInstanceDenialWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if let Ok(log) = TASK_DENIAL_LOG.try_with(Clone::clone) {
            for line in String::from_utf8_lossy(bytes).lines() {
                log.record(line);
            }
        }
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Body of a hosted sandbox instance, constructed on the instance runtime.
pub type SandboxInstanceBody = Box<
    dyn FnOnce(SandboxInstanceControls) -> Pin<Box<dyn Future<Output = Result<i32>> + Send>>
        + Send
        + 'static,
>;

/// Trusted identity of one sandbox instance asking to be hosted.
pub struct SandboxInstanceAdmission {
    /// Unique key for this generation within the host. It also names the
    /// instance's private directory, so it must be one path component.
    pub key: String,
    /// Digest of the launch material. Admitting the same digest under an
    /// occupied key again is idempotent.
    pub fingerprint: [u8; 32],
    pub sandbox_id: String,
}

/// Outcome of an accepted admission.
#[derive(Debug, PartialEq, Eq)]
pub enum SandboxInstanceAdmitted {
    Created,
    /// The identical admission is already hosted.
    Unchanged,
}

/// The hosted sandbox instance admission was rejected.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum SandboxInstanceHostError {
    #[error("sandbox instance key must be a single path component")]
    InvalidKey,
    #[error("key is already hosted with different launch material")]
    Conflict,
    #[error("key is still being removed")]
    Removing,
    #[error("prepare sandbox instance files: {0}")]
    Files(String),
}

/// Final status of one removal, shared by every remover.
type RemovalReport = watch::Receiver<Option<SandboxInstanceStatusReport>>;

/// How a sandbox instance leaves the host.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Departure {
    /// Stop the instance, which terminates its workload.
    Stop,
    /// Stop supervising without terminating the workload: the instance is
    /// cancelled, so its runtime loses this supervisor and keeps the workload
    /// frozen until another supervisor takes the generation over.
    Relinquish,
}

struct HostedSandboxInstance {
    /// Distinguishes this instance from a later one that reuses its key.
    serial: u64,
    fingerprint: [u8; 32],
    sandbox_id: String,
    stop: watch::Sender<bool>,
    suspend: tokio::sync::mpsc::Sender<()>,
    abort: watch::Sender<bool>,
    phase: watch::Receiver<SandboxInstancePhaseReport>,
    ready: watch::Receiver<bool>,
    /// Owns the instance's task tree, identity payload, and force-reap handles.
    spawner: Arc<InstanceSpawner>,
    /// The removal in progress, or the report of one that could not delete
    /// the instance's private files and so kept its key.
    removal: Option<RemovalReport>,
}

impl HostedSandboxInstance {
    /// Status of the instance hosted under `key`, which is its route.
    fn status(&self, key: &str) -> SandboxInstanceStatusReport {
        if let Some(reported) = self
            .removal
            .as_ref()
            .and_then(|report| report.borrow().clone())
        {
            return reported;
        }
        let phase = self.phase.borrow().clone();
        // The runner always reports a terminal phase, even while unwinding;
        // a closed channel without one still means supervision has ended.
        if !phase.is_terminal() && self.phase.has_changed().is_err() {
            return SandboxInstanceStatusReport {
                route: key.to_string(),
                sandbox_id: self.sandbox_id.clone(),
                phase: SandboxInstancePhaseReport::Failed {
                    error: "sandbox instance thread ended unexpectedly".to_string(),
                },
                ready: false,
            };
        }
        SandboxInstanceStatusReport {
            route: key.to_string(),
            sandbox_id: self.sandbox_id.clone(),
            phase,
            ready: *self.ready.borrow(),
        }
    }
}

type Instances = Arc<Mutex<HashMap<String, HostedSandboxInstance>>>;

/// Wakes observers of the registry whenever any instance's status changes.
#[derive(Clone)]
struct ChangeNotifier(Arc<watch::Sender<u64>>);

impl ChangeNotifier {
    fn notify(&self) {
        self.0.send_modify(|changes| *changes += 1);
    }
}

fn lock(instances: &Instances) -> MutexGuard<'_, HashMap<String, HostedSandboxInstance>> {
    instances
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// The one registry of hosted sandbox instances.
///
/// Each entry owns its admission fingerprint and sandbox identity, its task
/// spawner, phase, and stop signals, and a private directory under the host
/// root. Everything else an instance owns, including its gateway credentials
/// and runtime route, lives in its task tree and ends with it. A key stays
/// reserved from admission until one removal has stopped the task tree,
/// revoked its credentials, and deleted its private directory.
pub struct SandboxInstanceHost {
    instances: Instances,
    next_serial: AtomicU64,
    root: PathBuf,
    /// The shared runtime every instance's tasks run on.
    runtime: tokio::runtime::Handle,
    changes: ChangeNotifier,
}

impl SandboxInstanceHost {
    /// Host instances whose private directories live under `root`, with their
    /// tasks on `runtime`.
    ///
    /// # Errors
    ///
    /// Fails when `root` cannot be prepared as a private directory.
    pub fn new(root: PathBuf, runtime: tokio::runtime::Handle) -> Result<Self> {
        prepare_private_dir(&root)?;
        Ok(Self {
            instances: Arc::default(),
            next_serial: AtomicU64::new(0),
            root,
            runtime,
            changes: ChangeNotifier(Arc::new(watch::Sender::new(0))),
        })
    }

    /// Notified whenever an instance is admitted, changes phase or readiness,
    /// or is removed. The value only counts changes.
    #[must_use]
    pub fn changes(&self) -> watch::Receiver<u64> {
        self.changes.0.subscribe()
    }

    /// Admit a sandbox instance and start the runtime `launch` prepares in
    /// the instance's new, empty private directory.
    ///
    /// The bindings' gateway credentials are revoked once the instance's
    /// runtime has stopped, so nothing can reuse them after the instance ends.
    /// `launch` runs while the registry is locked, so concurrent admissions
    /// of one key cannot both start; it must not call back into the host.
    ///
    /// # Errors
    ///
    /// Rejects an occupied key with different launch material or one still
    /// being removed, and fails when the instance cannot be prepared or
    /// started. A failed admission retains no instance; deleting its private
    /// directory is best effort, and the next admission of the key deletes
    /// any stale files before reuse.
    pub fn admit<E: From<SandboxInstanceHostError>>(
        &self,
        admission: SandboxInstanceAdmission,
        launch: impl FnOnce(
            &Path,
        )
            -> std::result::Result<(SandboxInstanceBindings, SandboxInstanceBody), E>,
    ) -> std::result::Result<SandboxInstanceAdmitted, E> {
        let SandboxInstanceAdmission {
            key,
            fingerprint,
            sandbox_id,
        } = admission;
        if !matches!(
            Path::new(&key).components().collect::<Vec<_>>().as_slice(),
            [std::path::Component::Normal(name)] if *name == key.as_str()
        ) {
            return Err(SandboxInstanceHostError::InvalidKey.into());
        }
        let mut instances = lock(&self.instances);
        if let Some(existing) = instances.get(&key) {
            return if existing.removal.is_some() {
                Err(SandboxInstanceHostError::Removing.into())
            } else if existing.fingerprint == fingerprint {
                Ok(SandboxInstanceAdmitted::Unchanged)
            } else {
                Err(SandboxInstanceHostError::Conflict.into())
            };
        }
        let dir = self.root.join(&key);
        let started = prepare_instance_dir(&dir)
            .map_err(|error| E::from(SandboxInstanceHostError::Files(error.to_string())))
            .and_then(|()| launch(&dir));
        let running = match started {
            Ok((bindings, body)) => {
                spawn_instance(self.runtime.clone(), bindings, body, self.changes.clone())
            }
            Err(error) => {
                let _ = remove_instance_dir(&dir);
                return Err(error);
            }
        };
        instances.insert(
            key,
            HostedSandboxInstance {
                serial: self.next_serial.fetch_add(1, Ordering::Relaxed),
                fingerprint,
                sandbox_id,
                stop: running.stop,
                suspend: running.suspend,
                abort: running.abort,
                phase: running.phase,
                ready: running.ready,
                spawner: running.spawner,
                removal: None,
            },
        );
        self.changes.notify();
        Ok(SandboxInstanceAdmitted::Created)
    }

    /// Request suspension of one hosted instance. This only queues the request;
    /// the backend still decides whether suspension succeeds. Repeated pending
    /// requests coalesce. Returns false for missing, removing, or exited instances.
    #[must_use]
    pub fn request_suspend(&self, key: &str) -> bool {
        let instances = lock(&self.instances);
        let Some(instance) = instances
            .get(key)
            .filter(|instance| instance.removal.is_none())
        else {
            return false;
        };
        matches!(
            instance.suspend.try_send(()),
            Ok(()) | Err(tokio::sync::mpsc::error::TrySendError::Full(()))
        )
    }

    /// Status of a hosted sandbox instance, including one that ended but has
    /// not been removed. The report's route is the instance's key.
    #[must_use]
    pub fn status(&self, key: &str) -> Option<SandboxInstanceStatusReport> {
        lock(&self.instances)
            .get(key)
            .map(|instance| instance.status(key))
    }

    /// Status of every hosted sandbox instance.
    #[must_use]
    pub fn list(&self) -> Vec<SandboxInstanceStatusReport> {
        lock(&self.instances)
            .iter()
            .map(|(key, instance)| instance.status(key))
            .collect()
    }

    /// Stop one sandbox instance, delete its private files, and forget it.
    /// Other instances keep running.
    ///
    /// A supervising instance terminates its workload gracefully within
    /// `grace`; one still starting is cancelled at once, because startup has
    /// no safe intermediate state to wind down. Removal runs on its own task,
    /// so it completes even when the caller stops waiting, and concurrent
    /// callers share it and its result. The key stays reserved until the
    /// instance's thread has finished and its private directory is gone; if
    /// the directory cannot be deleted, the key stays reserved and the next
    /// removal tries again. Returns the final status, or `None` when `key` is
    /// not hosted.
    pub async fn remove(&self, key: &str, grace: Duration) -> Option<SandboxInstanceStatusReport> {
        self.depart(key, grace, Departure::Stop).await
    }

    /// Stop supervising one sandbox instance without stopping its workload,
    /// delete its private files, and forget it, as when its ownership moved
    /// to another supervisor. The instance is cancelled rather than asked to
    /// stop, so it never terminates the workload; the runtime freezes the
    /// workload until a supervisor granted the generation takes it over, and
    /// terminates it if none does in time. A removal already in progress is
    /// joined instead. Returns the final status, or `None` when `key` is not
    /// hosted.
    pub async fn relinquish(&self, key: &str) -> Option<SandboxInstanceStatusReport> {
        self.depart(key, RUNTIME_SHUTDOWN_TIMEOUT, Departure::Relinquish)
            .await
    }

    async fn depart(
        &self,
        key: &str,
        grace: Duration,
        departure: Departure,
    ) -> Option<SandboxInstanceStatusReport> {
        let (mut removed, sandbox_id) = self.begin_removal(key, grace, departure)?;
        // The removal task ends without a report only with its runtime.
        let reported = removed
            .wait_for(Option::is_some)
            .await
            .ok()
            .and_then(|status| status.clone());
        Some(reported.unwrap_or_else(|| SandboxInstanceStatusReport {
            route: key.to_string(),
            sandbox_id,
            phase: SandboxInstancePhaseReport::Failed {
                error: "sandbox instance removal did not finish".to_string(),
            },
            ready: false,
        }))
    }

    /// Relinquish every sandbox instance, for the shutdown of a supervisor
    /// whose instances other supervisors take over.
    pub async fn relinquish_all(&self) {
        let keys = lock(&self.instances).keys().cloned().collect::<Vec<_>>();
        futures::future::join_all(keys.iter().map(|key| self.relinquish(key))).await;
    }

    fn begin_removal(
        &self,
        key: &str,
        grace: Duration,
        departure: Departure,
    ) -> Option<(RemovalReport, String)> {
        let mut instances = lock(&self.instances);
        let instance = instances.get_mut(key)?;
        // Join a removal in progress; retry one that kept the key.
        if let Some(removed) = &instance.removal
            && removed.borrow().is_none()
        {
            return Some((removed.clone(), instance.sandbox_id.clone()));
        }
        match departure {
            Departure::Stop => {
                instance.stop.send_replace(true);
                if *instance.phase.borrow() == SandboxInstancePhaseReport::Starting {
                    instance.abort.send_replace(true);
                }
            }
            Departure::Relinquish => {
                instance.abort.send_replace(true);
            }
        }
        let (report, removed) = watch::channel(None);
        instance.removal = Some(removed.clone());
        tokio::spawn(
            Removal {
                instances: self.instances.clone(),
                key: key.to_string(),
                serial: instance.serial,
                phase: instance.phase.clone(),
                spawner: instance.spawner.clone(),
                dir: self.root.join(key),
                grace,
                report,
                changes: self.changes.clone(),
            }
            .run(),
        );
        self.changes.notify();
        Some((removed, instance.sandbox_id.clone()))
    }
}

/// The single owner of one sandbox instance's removal.
struct Removal {
    instances: Instances,
    key: String,
    serial: u64,
    phase: watch::Receiver<SandboxInstancePhaseReport>,
    spawner: Arc<InstanceSpawner>,
    dir: PathBuf,
    grace: Duration,
    report: watch::Sender<Option<SandboxInstanceStatusReport>>,
    changes: ChangeNotifier,
}

impl Removal {
    async fn run(mut self) {
        let finished = tokio::time::timeout(
            self.grace,
            self.phase.wait_for(SandboxInstancePhaseReport::is_terminal),
        )
        .await
        .is_ok();
        if !finished
            && let Some(instance) = lock(&self.instances)
                .get(&self.key)
                .filter(|instance| instance.serial == self.serial)
        {
            instance.abort.send_replace(true);
        }
        // End the instance's task tree, joining what cooperates within the
        // bound and force-reaping what does not. Once it is done the route
        // and every abortable task are gone and the credentials are revoked.
        self.spawner.reap(RUNTIME_SHUTDOWN_TIMEOUT).await;
        let dir = self.dir.clone();
        let cleanup = match tokio::task::spawn_blocking(move || remove_instance_dir(&dir)).await {
            Ok(result) => result.map_err(|error| error.to_string()),
            Err(error) => Err(error.to_string()),
        };
        let mut instances = lock(&self.instances);
        let Some(instance) = instances
            .get(&self.key)
            .filter(|instance| instance.serial == self.serial)
        else {
            return;
        };
        let mut status = instance.status(&self.key);
        status.ready = false;
        match cleanup {
            Ok(()) => {
                instances.remove(&self.key);
            }
            Err(error) => {
                tracing::warn!(%error, "could not delete removed sandbox instance files; keeping its key reserved");
                status.phase = SandboxInstancePhaseReport::Failed {
                    error: format!(
                        "sandbox instance stopped, but its private files remain: {error}"
                    ),
                };
            }
        }
        self.report.send_replace(Some(status));
        self.changes.notify();
    }
}

/// The spawner and signals of one started instance.
struct RunningInstance {
    stop: watch::Sender<bool>,
    suspend: tokio::sync::mpsc::Sender<()>,
    abort: watch::Sender<bool>,
    phase: watch::Receiver<SandboxInstancePhaseReport>,
    ready: watch::Receiver<bool>,
    spawner: Arc<InstanceSpawner>,
}

/// Spawns and tracks every task of one hosted sandbox instance.
///
/// A task spawned through the spawner runs on the host's shared instance
/// runtime, joins the instance's task tree, and enters every identity scope of
/// the instance's bindings, so teardown can wait for it and force-reap it if it
/// ignores cancellation, and its events stay attributed to the instance
/// wherever the task migrates.
struct InstanceSpawner {
    runtime: tokio::runtime::Handle,
    bindings: Arc<SandboxInstanceBindings>,
    tracker: TaskTracker,
    cancel: CancellationToken,
    /// Abort handles of the tracked tasks, so teardown can force-reap work
    /// that does not respond to cancellation.
    aborts: Mutex<AbortRegistry>,
    /// Completion signals of the blocking closures still running, so teardown
    /// can join work that abort cannot reach.
    blocking: Mutex<BlockingRegistry>,
    /// Serializes reaps, so a removal never deletes the instance's files while
    /// another reap is still joining the blocking work it took from the
    /// registry.
    reaping: tokio::sync::Mutex<()>,
    /// Ambient handle instance code uses to reach this spawner from crates
    /// that cannot name it.
    ambient: OnceLock<InstanceTasks>,
}

/// Force-reap handles plus the length at which the next prune runs.
///
/// Finished handles are kept until the registry doubles since the last scan,
/// so pruning stays O(1) amortized per spawn instead of scanning every live
/// handle on each spawn, while the registry remains bounded by live tasks.
#[derive(Default)]
struct AbortRegistry {
    handles: Vec<AbortHandle>,
    prune_at: usize,
}

impl AbortRegistry {
    /// Drop finished handles if the registry has grown to the next prune size.
    fn prune_if_due(&mut self) {
        if self.handles.len() < self.prune_at {
            return;
        }
        self.handles.retain(|abort| !abort.is_finished());
        // Scan again once the registry has doubled since this one, so a burst
        // of spawns amortizes the scan instead of repeating it per spawn.
        self.prune_at = self.handles.len().saturating_mul(2).max(1);
    }
}

/// Completion signals of the blocking closures still running, plus the length
/// at which the next prune runs.
///
/// Blocking work cannot be aborted, so teardown joins it instead of
/// force-reaping it. Finished entries are pruned on the same doubling schedule
/// as [`AbortRegistry`], so the registry stays bounded by live closures.
#[derive(Default)]
struct BlockingRegistry {
    outstanding: Vec<tokio::sync::oneshot::Receiver<()>>,
    prune_at: usize,
}

impl BlockingRegistry {
    /// Drop completion signals of closures that have already returned.
    fn prune_if_due(&mut self) {
        if self.outstanding.len() < self.prune_at {
            return;
        }
        // Keep the closures that are still running: `Empty` means no completion
        // signal yet, while `Ok` or `Closed` means the closure already returned.
        self.outstanding.retain_mut(|done| {
            matches!(
                done.try_recv(),
                Err(tokio::sync::oneshot::error::TryRecvError::Empty)
            )
        });
        self.prune_at = self.outstanding.len().saturating_mul(2).max(1);
    }
}

impl InstanceSpawner {
    fn new(runtime: tokio::runtime::Handle, bindings: SandboxInstanceBindings) -> Arc<Self> {
        let spawner = Arc::new(Self {
            runtime,
            bindings: Arc::new(bindings),
            tracker: TaskTracker::new(),
            cancel: CancellationToken::new(),
            aborts: Mutex::new(AbortRegistry::default()),
            blocking: Mutex::new(BlockingRegistry::default()),
            reaping: tokio::sync::Mutex::new(()),
            ambient: OnceLock::new(),
        });
        let backend = Arc::new(AmbientInstanceSpawner {
            spawner: Arc::downgrade(&spawner),
        });
        let _ = spawner.ambient.set(InstanceTasks::new(backend));
        spawner
    }

    /// Bindings every task of this instance is scoped to.
    fn bindings(&self) -> &SandboxInstanceBindings {
        &self.bindings
    }

    /// Cancellation the instance's tasks observe as it ends.
    fn cancellation(&self) -> CancellationToken {
        self.cancel.clone()
    }

    /// The ambient handle that routes instance-crate spawns back here.
    fn ambient(&self) -> InstanceTasks {
        self.ambient
            .get()
            .expect("the ambient handle is installed with the spawner")
            .clone()
    }

    /// Spawn `task` on the instance runtime, tracked by the instance's tree and
    /// scoped to every identity of the instance's bindings.
    ///
    /// This is for the instance root and for tasks that observe the instance's
    /// cancellation themselves: the root reports why the instance ended
    /// through its own cancellation branch, and an SSH session must disconnect
    /// before it returns. A cooperative wrapper could preempt either, so these
    /// are not wrapped.
    fn spawn<F>(self: &Arc<Self>, task: F) -> JoinHandle<F::Output>
    where
        F: Future + Send + 'static,
        F::Output: Send + 'static,
    {
        let ambient = self.ambient();
        let scoped = scoped_instance_task(Arc::clone(&self.bindings), ambient.scope(task));
        let handle = self.tracker.spawn_on(scoped, &self.runtime);
        self.register_abort(handle.abort_handle());
        handle
    }

    /// Spawn an already-scoped task, tracked by the instance's tree.
    ///
    /// The ambient handle re-enters the caller's scopes; this adds the
    /// instance's identity scopes, ends the task at its next await point when
    /// the instance is cancelled, and registers it for teardown.
    fn spawn_tracked(&self, task: ScopedTask) -> JoinHandle<()> {
        let scoped =
            cancellable_instance_task(Arc::clone(&self.bindings), self.cancel.clone(), task);
        let handle = self.tracker.spawn_on(
            async move {
                let _ = scoped.await;
            },
            &self.runtime,
        );
        self.register_abort(handle.abort_handle());
        handle
    }

    /// Run a blocking closure with the instance's identity scopes re-entered.
    ///
    /// Tokio task scopes do not reach blocking closures, so they must re-enter
    /// the scopes synchronously; the ambient task scope is already entered by
    /// [`InstanceTasks::spawn_blocking`] before the closure reaches here. An
    /// abort cannot interrupt blocking work that is already running, so the
    /// closure's completion is registered and [`Self::reap`] joins it instead.
    ///
    /// A closure still queued when the instance ends is skipped and the handle
    /// yields `None`, so removal bounds queued blocking work too.
    fn spawn_scoped_blocking(
        self: &Arc<Self>,
        task: ScopedBlockingTask,
    ) -> JoinHandle<Option<ScopedBlockingOutput>> {
        let bindings = Arc::clone(&self.bindings);
        let cancel = self.cancel.clone();
        let (done, completed) = tokio::sync::oneshot::channel();
        // Run on the instance's own bounded blocking pool, not whichever
        // runtime happened to enter the instance scope.
        let handle = self.runtime.spawn_blocking(move || {
            let output = if cancel.is_cancelled() {
                None
            } else {
                Some(scoped_instance_blocking(&bindings, task))
            };
            // Signal completion so removal joins the closure even when the
            // caller never awaits its handle. A panicking closure drops the
            // sender, which also completes the signal.
            let _ = done.send(());
            output
        });
        self.register_blocking(completed);
        handle
    }

    /// Keep a force-reap handle for a live task.
    fn register_abort(&self, abort: AbortHandle) {
        let mut aborts = self
            .aborts
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // Keeping finished tasks' handles would grow the registry with every
        // task the instance spawns; prune on a doubling schedule instead of on
        // every spawn.
        aborts.prune_if_due();
        aborts.handles.push(abort);
    }

    /// Keep a completion signal for a running blocking closure.
    fn register_blocking(&self, done: tokio::sync::oneshot::Receiver<()>) {
        let mut blocking = self
            .blocking
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        blocking.prune_if_due();
        blocking.outstanding.push(done);
    }

    /// Abort every task currently registered for force-reap.
    fn abort_outstanding(&self) {
        let aborts = self
            .aborts
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for abort in &aborts.handles {
            abort.abort();
        }
    }

    /// Join blocking closures still running, within a fresh window of
    /// `timeout`.
    ///
    /// Blocking work cannot be aborted, so removal waits for it here; an
    /// already-running closure would otherwise keep writing to the instance's
    /// private directory after the key was released and the directory deleted.
    /// A closure registered while earlier ones are joined is covered by the
    /// next round, and anything left after [`REAP_BLOCKING_ROUNDS`] is the
    /// documented residual: blocking work that outlives its removal bound.
    async fn join_blocking(&self, timeout: Duration) {
        for _ in 0..REAP_BLOCKING_ROUNDS {
            let outstanding = {
                let mut blocking = self
                    .blocking
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                std::mem::take(&mut blocking.outstanding)
            };
            if outstanding.is_empty() {
                return;
            }
            // Each round gets a fresh full window, so a closure registered
            // while an earlier round was joining is still covered.
            let deadline = tokio::time::Instant::now() + timeout;
            let joined = tokio::time::timeout_at(deadline, async {
                for done in outstanding {
                    let _ = done.await;
                }
            })
            .await;
            if joined.is_err() {
                return;
            }
        }
    }

    /// Stop the instance's task tree and the blocking work it started.
    ///
    /// Cancels the instance's tasks — which ends every tracked task at its
    /// next await point — and waits up to `timeout` for them to finish. What
    /// is left is aborted, repeating the pass so a task registered while one
    /// runs is also force-reaped, with each pass getting a fresh window of the
    /// same bound so later passes do not inherit a spent one. Blocking closures
    /// already running cannot be aborted, so they are joined afterwards the
    /// same way, one fresh window per round; a closure still queued when the
    /// instance ends is skipped instead of starting.
    async fn reap(&self, timeout: Duration) {
        // A root task's completion reaps the instance from its own task while
        // a removal may be reaping it too. Serialize them so neither returns
        // (and removal deletes the instance's files) while the other still
        // holds blocking work to join.
        let _reaping = self.reaping.lock().await;
        self.cancel.cancel();
        self.tracker.close();
        let deadline = tokio::time::Instant::now() + timeout;
        if tokio::time::timeout_at(deadline, self.tracker.wait())
            .await
            .is_err()
        {
            // Aborted tasks drop at their next scheduling point, which is
            // after `deadline` has already elapsed, so each pass gets a fresh
            // window instead of reusing the spent deadline.
            for _ in 0..REAP_ABORT_PASSES {
                if self.tracker.is_empty() {
                    break;
                }
                self.abort_outstanding();
                let abort_deadline = tokio::time::Instant::now() + timeout;
                let _ = tokio::time::timeout_at(abort_deadline, self.tracker.wait()).await;
            }
        }
        self.join_blocking(timeout).await;
    }
}

/// Ambient backend that routes instance-crate spawns to the owning spawner.
///
/// It holds a weak reference so the spawner's task-local handle never keeps
/// the spawner (and the instance's state) alive.
struct AmbientInstanceSpawner {
    spawner: Weak<InstanceSpawner>,
}

impl InstanceTaskBackend for AmbientInstanceSpawner {
    fn spawn_scoped(&self, task: ScopedTask) -> JoinHandle<()> {
        // The instance is already gone when the spawner cannot be upgraded;
        // dropping the task is the fail-closed choice, matching a spawn that
        // loses its runtime. The debug line names what was discarded so a
        // late spawn is visible instead of silent.
        self.spawner.upgrade().map_or_else(
            || {
                tracing::debug!("dropping an instance task: its sandbox instance is already gone");
                tokio::spawn(async {})
            },
            |spawner| spawner.spawn_tracked(task),
        )
    }

    fn spawn_scoped_blocking(
        &self,
        task: ScopedBlockingTask,
    ) -> JoinHandle<Option<ScopedBlockingOutput>> {
        self.spawner.upgrade().map_or_else(
            || {
                tracing::debug!(
                    "dropping a queued blocking closure: its sandbox instance is already gone"
                );
                tokio::task::spawn_blocking(|| None)
            },
            |spawner| spawner.spawn_scoped_blocking(task),
        )
    }

    fn spawn_owning_cancellation(&self, task: ScopedTask) -> JoinHandle<()> {
        self.spawner.upgrade().map_or_else(
            || {
                tracing::debug!("dropping an instance task: its sandbox instance is already gone");
                tokio::spawn(async {})
            },
            |spawner| spawner.spawn(task),
        )
    }

    fn cancelled(&self) -> Pin<Box<dyn Future<Output = ()> + Send>> {
        match self.spawner.upgrade() {
            Some(spawner) => {
                let cancelled = spawner.cancellation();
                Box::pin(async move { cancelled.cancelled().await })
            }
            // The instance is already gone, so nothing will cancel the task
            // later; report the end now instead of letting it run on.
            None => Box::pin(std::future::ready(())),
        }
    }
}

/// Run `task` inside every identity scope the instance's bindings define.
///
/// Optional slots (no log sink, file-backed denial log) are skipped, so work
/// with no task-scoped value sees none, exactly as unbound work does.
async fn scoped_instance_task<F>(bindings: Arc<SandboxInstanceBindings>, task: F) -> F::Output
where
    F: Future,
{
    let context = bindings.event_context.clone();
    let json = bindings.ocsf_json.clone();
    let sink = bindings.log_sink.clone();
    let denial = match &bindings.denial_log {
        DenialLogSource::SandboxInstance(log) => Some(log.clone()),
        DenialLogSource::ShorthandFiles(_) => None,
    };
    let task = async move {
        match sink {
            Some(sink) => openshell_supervisor_process::log_push::scope_sink(sink, task).await,
            None => task.await,
        }
    };
    let task = async move {
        match denial {
            Some(log) => scope_denial_log(log, task).await,
            None => task.await,
        }
    };
    openshell_ocsf::ctx::scope_ctx(context, openshell_ocsf::scope_json_settings(json, task)).await
}

/// Run a scoped `task`, ending it at its next await point when the instance is
/// cancelled.
///
/// A task without a cancellation branch of its own is dropped, running its
/// Drop guards, instead of burning removal's whole cooperative window until
/// the force-reap abort. This is for adopted instance work; the root task
/// keeps its own cancellation branch and is not wrapped, because this select
/// can otherwise win the poll and drop the root before it records why the
/// instance ended.
async fn cancellable_instance_task<F>(
    bindings: Arc<SandboxInstanceBindings>,
    cancel: CancellationToken,
    task: F,
) -> Option<F::Output>
where
    F: Future,
{
    tokio::select! {
        biased;
        output = scoped_instance_task(bindings, task) => Some(output),
        () = cancel.cancelled() => None,
    }
}

/// Run `f` inside every identity scope the instance's bindings define.
///
/// Blocking closures cannot inherit a task scope, so this re-enters each one
/// synchronously.
fn scoped_instance_blocking<T>(bindings: &SandboxInstanceBindings, f: impl FnOnce() -> T) -> T {
    let context = bindings.event_context.clone();
    let json = bindings.ocsf_json.clone();
    let sink = bindings.log_sink.clone();
    let denial = match &bindings.denial_log {
        DenialLogSource::SandboxInstance(log) => Some(log.clone()),
        DenialLogSource::ShorthandFiles(_) => None,
    };
    openshell_ocsf::ctx::scope_ctx_sync(context, || {
        openshell_ocsf::scope_json_settings_sync(json, || {
            let with_sink = || match sink {
                Some(sink) => openshell_supervisor_process::log_push::scope_sink_sync(sink, f),
                None => f(),
            };
            match denial {
                Some(log) => scope_denial_log_sync(log, with_sink),
                None => with_sink(),
            }
        })
    })
}

fn spawn_instance(
    runtime: tokio::runtime::Handle,
    bindings: SandboxInstanceBindings,
    body: SandboxInstanceBody,
    changes: ChangeNotifier,
) -> RunningInstance {
    let (stop, stop_requested) = watch::channel(false);
    let (suspend, suspend_requested) = tokio::sync::mpsc::channel(1);
    let (abort, abort_requested) = watch::channel(false);
    let (phase_tx, phase) = watch::channel(SandboxInstancePhaseReport::Starting);
    let (ready_tx, ready) = watch::channel(false);
    let spawner = InstanceSpawner::new(runtime, bindings);
    // The exit guard is created here and owned by the root task's future, so
    // it also drops, revoking the credentials, when the task is aborted
    // before it is ever polled.
    let exit = SandboxInstanceExit {
        credentials: spawner.bindings().gateway_credentials.clone(),
        phase: phase_tx.clone(),
        ready: ready_tx.clone(),
        outcome: None,
        changes: changes.clone(),
        spawner: Arc::downgrade(&spawner),
    };
    let task_spawner = spawner.clone();
    spawner.spawn(run_hosted_sandbox_instance(
        exit,
        task_spawner,
        body,
        SandboxInstanceSignals {
            stop_requested,
            suspend_requested,
            abort_requested,
            phase: phase_tx,
            ready: ready_tx,
            changes,
        },
    ));
    RunningInstance {
        stop,
        suspend,
        abort,
        phase,
        ready,
        spawner,
    }
}

struct SandboxInstanceSignals {
    stop_requested: watch::Receiver<bool>,
    suspend_requested: tokio::sync::mpsc::Receiver<()>,
    abort_requested: watch::Receiver<bool>,
    phase: watch::Sender<SandboxInstancePhaseReport>,
    ready: watch::Sender<bool>,
    changes: ChangeNotifier,
}

/// Reports a hosted sandbox instance's end and revokes its credentials,
/// including when its root task unwinds or is aborted.
struct SandboxInstanceExit {
    credentials: GatewayCredentials,
    phase: watch::Sender<SandboxInstancePhaseReport>,
    ready: watch::Sender<bool>,
    outcome: Option<SandboxInstancePhaseReport>,
    changes: ChangeNotifier,
    /// The instance whose remaining tasks end with this root task. Weak so the
    /// guard in the root future never keeps the spawner (and its tree) alive.
    spawner: Weak<InstanceSpawner>,
}

impl Drop for SandboxInstanceExit {
    fn drop(&mut self) {
        self.credentials.revoke();
        self.ready.send_replace(false);
        let outcome = self
            .outcome
            .take()
            .unwrap_or_else(|| SandboxInstancePhaseReport::Failed {
                error: "sandbox instance thread ended without reporting a result".to_string(),
            });
        self.phase.send_replace(outcome);
        self.changes.notify();
        // The root has resolved, so the instance's remaining tasks must end
        // with it, exactly as the per-instance runtime's shutdown did before
        // instances shared one. They are reaped from a separate, untracked
        // task because the root is itself tracked: reaping from inside its own
        // completion would wait on itself. Drop can run without a runtime,
        // where an explicit removal's own reap still covers the tree.
        if let Some(spawner) = self.spawner.upgrade()
            && let Ok(runtime) = tokio::runtime::Handle::try_current()
        {
            runtime.spawn(async move {
                spawner.reap(RUNTIME_SHUTDOWN_TIMEOUT).await;
            });
        }
    }
}

/// Run one hosted sandbox instance's root task until it exits, panics, or is
/// aborted. The exit guard travels in the task's future, so panic unwind and
/// abort both drop it and revoke the instance's credentials.
async fn run_hosted_sandbox_instance(
    mut exit: SandboxInstanceExit,
    spawner: Arc<InstanceSpawner>,
    body: SandboxInstanceBody,
    signals: SandboxInstanceSignals,
) {
    let SandboxInstanceSignals {
        stop_requested,
        suspend_requested,
        mut abort_requested,
        phase,
        ready,
        changes,
    } = signals;
    let cancel = spawner.cancellation();
    let outcome = AssertUnwindSafe(async {
        let (readiness, mut accepted) = watch::channel(false);
        let mirror_phase = phase.clone();
        let mirror_ready = ready.clone();
        let mirror_changes = changes.clone();
        let mirror = spawner.spawn(async move {
            loop {
                let is_ready = *accepted.borrow_and_update();
                let changed = mirror_ready.send_replace(is_ready) != is_ready;
                let promoted = is_ready
                    && mirror_phase.send_if_modified(|current| {
                        let promote = *current == SandboxInstancePhaseReport::Starting;
                        if promote {
                            *current = SandboxInstancePhaseReport::Supervising;
                        }
                        promote
                    });
                if changed || promoted {
                    mirror_changes.notify();
                }
                if accepted.changed().await.is_err() {
                    if mirror_ready.send_replace(false) {
                        mirror_changes.notify();
                    }
                    return;
                }
            }
        });
        let mut stop = stop_requested.clone();
        let stop_phase = phase.clone();
        let stop_changes = changes.clone();
        let shutdown = Box::pin(async move {
            let _ = stop.wait_for(|requested| *requested).await;
            stop_phase.send_replace(SandboxInstancePhaseReport::Stopping);
            stop_changes.notify();
        });
        let instance = body(SandboxInstanceControls {
            shutdown,
            shutdown_ends_process: false,
            suspend: Box::pin(tokio_stream::wrappers::ReceiverStream::new(
                suspend_requested,
            )),
            readiness,
        });
        let outcome = tokio::select! {
            result = instance => result,
            _ = abort_requested.wait_for(|requested| *requested) => {
                Err(miette::miette!("sandbox instance was cancelled before it finished"))
            }
            () = cancel.cancelled() => {
                Err(miette::miette!("sandbox instance was cancelled before it finished"))
            }
        };
        mirror.abort();
        outcome
    })
    .catch_unwind()
    .await;
    exit.outcome = Some(match outcome {
        Ok(Ok(exit_code)) => SandboxInstancePhaseReport::Exited { exit_code },
        Ok(Err(error)) => SandboxInstancePhaseReport::Failed {
            error: error.to_string(),
        },
        Err(_) => SandboxInstancePhaseReport::Failed {
            error: "sandbox instance panicked".to_string(),
        },
    });
}

/// Prepare a hosted sandbox instance of `config` whose private files live in
/// `dir`: bindings with its own gateway credentials, log push stream, and
/// denial log, and a body that supervises it.
///
/// # Errors
///
/// Fails when the launch credentials are invalid.
pub fn sandbox_instance_launch(
    backend_setup: Box<dyn crate::backend_setup::BackendSetup + Send>,
    mut config: crate::SandboxInstanceConfig,
    dir: &Path,
) -> Result<(SandboxInstanceBindings, SandboxInstanceBody)> {
    // Instances must never share proxy trust files.
    config
        .proxy_tls
        .directory
        .get_or_insert_with(|| dir.to_path_buf());
    // Log push, supervision, and renewal share these credentials.
    let gateway_credentials = GatewayCredentials::from_auth_bundle(&config.auth_bundle)?;
    let log_push = config
        .sandbox_id
        .clone()
        .zip(config.openshell_endpoint.as_deref().map(|endpoint| {
            openshell_core::grpc_client::GatewayEndpoint::new(endpoint, gateway_credentials.clone())
        }));
    let (log_tx, log_rx) =
        tokio::sync::mpsc::channel(openshell_supervisor_process::log_push::LOG_PUSH_QUEUE_LINES);
    let bindings = SandboxInstanceBindings {
        gateway_credentials,
        event_context: Arc::new(config.event_context()),
        ocsf_json: OcsfJsonSettings::default(),
        log_sink: log_push
            .as_ref()
            .map(|(sandbox_id, _)| LogSink::new(sandbox_id.as_str(), log_tx)),
        denial_log: DenialLogSource::SandboxInstance(SandboxInstanceDenialLog::default()),
    };
    let instance_bindings = bindings.clone();
    let body: SandboxInstanceBody = Box::new(move |controls| {
        Box::pin(async move {
            let _log_push = log_push.map(|(sandbox_id, gateway)| {
                openshell_supervisor_process::log_push::start_log_push(gateway, sandbox_id, log_rx)
            });
            Box::pin(crate::run_sandbox_instance(
                backend_setup.as_ref(),
                config,
                instance_bindings,
                controls,
            ))
            .await
        })
    });
    Ok((bindings, body))
}

/// Create `dir` as an empty private directory, replacing files a previous
/// process left behind.
fn prepare_instance_dir(dir: &Path) -> Result<()> {
    remove_instance_dir(dir)
        .into_diagnostic()
        .wrap_err_with(|| format!("delete stale sandbox instance files {}", dir.display()))?;
    prepare_private_dir(dir)
}

fn remove_instance_dir(dir: &Path) -> std::io::Result<()> {
    match std::fs::remove_dir_all(dir) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        result => result,
    }
}

#[cfg(unix)]
pub(crate) fn prepare_private_dir(path: &Path) -> Result<()> {
    use std::os::unix::fs::{DirBuilderExt as _, PermissionsExt as _};

    match std::fs::DirBuilder::new().mode(0o700).create(path) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            let metadata = std::fs::symlink_metadata(path).into_diagnostic()?;
            if !metadata.is_dir() {
                return Err(miette::miette!(
                    "sandbox instance path is not a directory: {}",
                    path.display()
                ));
            }
            std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))
                .into_diagnostic()?;
        }
        Err(error) => {
            return Err(error)
                .into_diagnostic()
                .wrap_err_with(|| format!("create private directory {}", path.display()));
        }
    }
    Ok(())
}

#[cfg(not(unix))]
pub(crate) fn prepare_private_dir(_path: &Path) -> Result<()> {
    Err(miette::miette!(
        "hosted sandbox instances require a Unix host"
    ))
}

// Hosted sandbox instances require a Unix host.
#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use futures::StreamExt as _;
    use http_body_util::BodyExt as _;
    use openshell_core::grpc_client::GatewayEndpoint;
    use openshell_core::proto::{
        GatewayMessage, SandboxLogLine, SessionAccepted, SupervisorMessage, gateway_message,
        open_shell_client::OpenShellClient,
    };
    use tokio::sync::mpsc;
    use tonic::codegen::tokio_stream::wrappers::TcpListenerStream;
    use tonic::codegen::{Body, BoxFuture, BoxStream, Bytes, Service, StdError};
    use tracing_subscriber::layer::SubscriberExt as _;

    type Captured = Arc<Mutex<Vec<(String, Vec<u8>)>>>;

    /// Records each gRPC request's bearer and body, then answers `UNIMPLEMENTED`.
    async fn capture_gateway() -> (String, Captured) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let captured = Captured::default();
        let sink = captured.clone();
        tokio::spawn(async move {
            loop {
                let (stream, _) = listener.accept().await.unwrap();
                let sink = sink.clone();
                tokio::spawn(async move {
                    let service = hyper::service::service_fn(
                        move |request: http::Request<hyper::body::Incoming>| {
                            let sink = sink.clone();
                            async move {
                                let authorization = request
                                    .headers()
                                    .get("authorization")
                                    .and_then(|value| value.to_str().ok())
                                    .unwrap_or_default()
                                    .to_string();
                                let body = request
                                    .into_body()
                                    .collect()
                                    .await
                                    .map(|body| body.to_bytes().to_vec())
                                    .unwrap_or_default();
                                sink.lock().unwrap().push((authorization, body));
                                http::Response::builder()
                                    .header("content-type", "application/grpc")
                                    .header("grpc-status", "12")
                                    .body(http_body_util::Empty::<Bytes>::new())
                            }
                        },
                    );
                    let _ = hyper::server::conn::http2::Builder::new(
                        hyper_util::rt::TokioExecutor::new(),
                    )
                    .serve_connection(hyper_util::rt::TokioIo::new(stream), service)
                    .await;
                });
            }
        });
        (format!("http://{address}"), captured)
    }

    fn bearers_for(captured: &Captured, sandbox: &str) -> Vec<String> {
        captured
            .lock()
            .unwrap()
            .iter()
            .filter(|(_, body)| {
                body.windows(sandbox.len())
                    .any(|bytes| bytes == sandbox.as_bytes())
            })
            .map(|(authorization, _)| authorization.clone())
            .collect()
    }

    /// A loopback gateway that accepts the supervisor session stream.
    ///
    /// Unlike [`capture_gateway`], this is a real tonic server on a real
    /// HTTP/2 connection, so the instance and the gateway both run
    /// connection-bearing tasks while teardown happens. It is still a test
    /// stub: it understands only `ConnectSupervisor` and ignores the
    /// supervisor's hello, so it under-counts the descriptors a production
    /// gateway connection holds.
    #[derive(Clone)]
    struct LoopbackGateway;

    impl tonic::server::NamedService for LoopbackGateway {
        const NAME: &'static str = "openshell.v1.OpenShell";
    }

    struct LoopbackConnect;

    impl tonic::server::StreamingService<SupervisorMessage> for LoopbackConnect {
        type Response = GatewayMessage;
        type ResponseStream = BoxStream<GatewayMessage>;
        type Future = BoxFuture<tonic::Response<Self::ResponseStream>, tonic::Status>;

        fn call(
            &mut self,
            _request: tonic::Request<tonic::Streaming<SupervisorMessage>>,
        ) -> Self::Future {
            Box::pin(async move {
                let accepted = GatewayMessage {
                    payload: Some(gateway_message::Payload::SessionAccepted(SessionAccepted {
                        session_id: "loopback-session".to_string(),
                        ..Default::default()
                    })),
                };
                // Accept the session, then keep the stream open until the
                // supervisor connection is dropped.
                let stream: Self::ResponseStream = Box::pin(
                    futures::stream::once(async move { Ok(accepted) })
                        .chain(futures::stream::pending()),
                );
                Ok(tonic::Response::new(stream))
            })
        }
    }

    impl<B> Service<http::Request<B>> for LoopbackGateway
    where
        B: Body<Data = Bytes> + Send + 'static,
        B::Error: Into<StdError> + Send + 'static,
    {
        type Response = http::Response<tonic::body::Body>;
        type Error = std::convert::Infallible;
        type Future = BoxFuture<Self::Response, Self::Error>;

        fn poll_ready(
            &mut self,
            _cx: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Result<(), Self::Error>> {
            std::task::Poll::Ready(Ok(()))
        }

        fn call(&mut self, request: http::Request<B>) -> Self::Future {
            Box::pin(async move {
                let response = match request.uri().path() {
                    "/openshell.v1.OpenShell/ConnectSupervisor" => {
                        tonic::server::Grpc::new(tonic_prost::ProstCodec::<
                            GatewayMessage,
                            SupervisorMessage,
                        >::default())
                        .streaming(LoopbackConnect, request)
                        .await
                    }
                    _ => http::Response::builder()
                        .status(200)
                        .header("grpc-status", "12")
                        .header("content-type", "application/grpc")
                        .body(tonic::body::Body::empty())
                        .unwrap(),
                };
                Ok(response)
            })
        }
    }

    async fn loopback_gateway() -> (String, JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            let _ = tonic::transport::Server::builder()
                .add_service(LoopbackGateway)
                .serve_with_incoming(TcpListenerStream::new(listener))
                .await;
        });
        (address, server)
    }

    struct Instance {
        identity: tokio::sync::oneshot::Receiver<String>,
        logs: mpsc::Receiver<SandboxLogLine>,
        call_again: mpsc::Sender<()>,
        credentials: GatewayCredentials,
    }

    fn start_instance(host: &SandboxInstanceHost, gateway_url: &str, sandbox: &str) -> Instance {
        let sandbox = sandbox.to_string();
        let key = sandbox.clone();
        let credentials =
            GatewayCredentials::from_gateway_token(&format!("token-{sandbox}")).unwrap();
        let gateway = GatewayEndpoint::new(gateway_url, credentials.clone());
        let (log_tx, logs) = mpsc::channel(16);
        let bindings = SandboxInstanceBindings {
            gateway_credentials: credentials.clone(),
            event_context: Arc::new(crate::sandbox_instance_event_context(
                Some(&sandbox),
                Some(&sandbox),
                String::new(),
            )),
            ocsf_json: OcsfJsonSettings::default(),
            log_sink: Some(LogSink::new(sandbox.as_str(), log_tx)),
            denial_log: DenialLogSource::SandboxInstance(SandboxInstanceDenialLog::default()),
        };
        let (identity_tx, identity) = tokio::sync::oneshot::channel();
        let (call_again, mut calls) = mpsc::channel(1);
        let body: SandboxInstanceBody = Box::new(move |controls| {
            Box::pin(async move {
                let SandboxInstanceControls {
                    shutdown,
                    readiness,
                    ..
                } = controls;
                // A task the instance spawns must never inherit another
                // instance's identity.
                let observed = tokio::spawn(async {
                    tokio::task::spawn_blocking(|| openshell_ocsf::ctx::ctx().sandbox_id.clone())
                        .await
                        .unwrap()
                })
                .await
                .unwrap();
                let _ = identity_tx.send(observed);
                let _ =
                    openshell_core::grpc_client::fetch_settings_snapshot(&gateway, &sandbox).await;
                let dispatch = tracing::Dispatch::new(
                    tracing_subscriber::registry()
                        .with(openshell_supervisor_process::log_push::LogPushLayer::thread_bound()),
                );
                // Emitted by the instance's root task, which the spawner scopes
                // to this instance's log sink.
                tracing::dispatcher::with_default(&dispatch, || {
                    tracing::info!(target: "instance_test", "emitted by {sandbox}");
                });
                readiness.send_replace(true);
                tokio::pin!(shutdown);
                loop {
                    tokio::select! {
                        () = &mut shutdown => return Ok(0),
                        Some(()) = calls.recv() => {
                            let _ = openshell_core::grpc_client::fetch_settings_snapshot(
                                &gateway, &sandbox,
                            )
                            .await;
                        }
                    }
                }
            })
        });
        assert_eq!(
            admit(host, &key, &key, 0, bindings, body),
            Ok(SandboxInstanceAdmitted::Created)
        );
        Instance {
            identity,
            logs,
            call_again,
            credentials,
        }
    }

    /// Resolve once the watch value becomes `true`.
    ///
    /// Kept as a helper so `tokio::select!` stores `()` instead of the watch
    /// guard it would otherwise hold across another branch's await, which
    /// would make the spawned future `!Send`.
    async fn wait_for_true(receiver: &mut watch::Receiver<bool>) {
        let _ = receiver.wait_for(|requested| *requested).await;
    }

    /// Start an instance whose root task opens a real supervisor session to
    /// `gateway_url` and holds it open until it is told to stop.
    ///
    /// The session is spawned through the ambient instance spawner, so it is
    /// tracked by the instance and carries its identity. `established` fires
    /// once the streaming RPC has been accepted.
    fn start_session_instance(
        host: &SandboxInstanceHost,
        gateway_url: &str,
        sandbox: &str,
        established: tokio::sync::oneshot::Sender<()>,
    ) -> GatewayCredentials {
        let credentials =
            GatewayCredentials::from_gateway_token(&format!("token-{sandbox}")).unwrap();
        let bindings = SandboxInstanceBindings {
            gateway_credentials: credentials.clone(),
            event_context: Arc::new(crate::sandbox_instance_event_context(
                Some(sandbox),
                Some(sandbox),
                String::new(),
            )),
            ocsf_json: OcsfJsonSettings::default(),
            log_sink: None,
            denial_log: DenialLogSource::SandboxInstance(SandboxInstanceDenialLog::default()),
        };
        let gateway_url = gateway_url.to_string();
        let key = sandbox.to_string();
        let body: SandboxInstanceBody = Box::new(move |controls| {
            Box::pin(async move {
                let (stop, mut stop_requested) = watch::channel(false);
                let session = openshell_core::instance_tasks::spawn(async move {
                    let mut client = OpenShellClient::connect(gateway_url)
                        .await
                        .expect("connect to the loopback gateway");
                    let mut inbound = client
                        .connect_supervisor(futures::stream::empty::<SupervisorMessage>())
                        .await
                        .expect("open a supervisor session")
                        .into_inner();
                    let _ = established.send(());
                    loop {
                        tokio::select! {
                            () = wait_for_true(&mut stop_requested) => return,
                            // Hold the HTTP/2 stream open; the stub gateway
                            // sends one message and then nothing.
                            _ = inbound.message() => {}
                        }
                    }
                });
                controls.readiness.send_replace(true);
                controls.shutdown.await;
                stop.send_replace(true);
                let _ = session.await;
                Ok(0)
            })
        });
        admit(host, &key, &key, 1, bindings, body).unwrap();
        credentials
    }

    /// An instance whose body emits OCSF context, log lines, and denial lines
    /// on demand from a tracked child task.
    struct ChattyInstance {
        key: String,
        logs: mpsc::Receiver<SandboxLogLine>,
        observed: mpsc::Receiver<String>,
        emit: mpsc::Sender<()>,
        denial: SandboxInstanceDenialLog,
    }

    fn start_chatty_instance(host: &SandboxInstanceHost, key: &str) -> ChattyInstance {
        let credentials = GatewayCredentials::from_gateway_token(&format!("token-{key}")).unwrap();
        let (log_tx, logs) = mpsc::channel(64);
        let (observed_tx, observed) = mpsc::channel(64);
        let denial = SandboxInstanceDenialLog::default();
        let bindings = SandboxInstanceBindings {
            gateway_credentials: credentials,
            event_context: Arc::new(crate::sandbox_instance_event_context(
                Some(key),
                Some(key),
                String::new(),
            )),
            ocsf_json: OcsfJsonSettings::default(),
            log_sink: Some(LogSink::new(key, log_tx)),
            denial_log: DenialLogSource::SandboxInstance(denial.clone()),
        };
        let (emit, mut emissions) = mpsc::channel::<()>(8);
        let (stop, mut stop_requested) = watch::channel(false);
        let body_sandbox = key.to_string();
        let body: SandboxInstanceBody = Box::new(move |controls| {
            Box::pin(async move {
                let child = openshell_core::instance_tasks::spawn(async move {
                    let dispatch = tracing::Dispatch::new(tracing_subscriber::registry().with(
                        openshell_supervisor_process::log_push::LogPushLayer::thread_bound(),
                    ));
                    loop {
                        tokio::select! {
                            () = wait_for_true(&mut stop_requested) => return,
                            Some(()) = emissions.recv() => {
                                // Read the instance identity the task scope
                                // supplies before emitting anything.
                                let observed = openshell_ocsf::ctx::ctx().sandbox_id.clone();
                                let _ = observed_tx.send(observed).await;
                                tracing::dispatcher::with_default(&dispatch, || {
                                    tracing::info!(
                                        target: "cross_attribution",
                                        "busy event from {body_sandbox}"
                                    );
                                });
                                let line = format!(
                                    "2026-05-06T17:02:00.000Z OCSF NET:OPEN [MED] DENIED curl(42) -> {body_sandbox}.example:443 [policy:- engine:opa]"
                                );
                                std::io::Write::write_all(
                                    &mut SandboxInstanceDenialWriter,
                                    line.as_bytes(),
                                )
                                .unwrap();
                            }
                        }
                    }
                });
                controls.readiness.send_replace(true);
                controls.shutdown.await;
                stop.send_replace(true);
                let _ = child.await;
                Ok(0)
            })
        });
        admit(host, key, key, 1, bindings, body).unwrap();
        ChattyInstance {
            key: key.to_string(),
            logs,
            observed,
            emit,
            denial,
        }
    }

    async fn wait_supervising(host: &SandboxInstanceHost, key: &str) {
        tokio::time::timeout(Duration::from_secs(10), async {
            while host.status(key).map(|status| status.phase)
                != Some(SandboxInstancePhaseReport::Supervising)
            {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("instance reaches supervision");
    }

    fn unbound() -> SandboxInstanceBindings {
        SandboxInstanceBindings {
            gateway_credentials: GatewayCredentials::from_gateway_token("unbound-token").unwrap(),
            event_context: Arc::new(crate::sandbox_instance_event_context(
                None,
                None,
                String::new(),
            )),
            ocsf_json: OcsfJsonSettings::default(),
            log_sink: None,
            denial_log: DenialLogSource::SandboxInstance(SandboxInstanceDenialLog::default()),
        }
    }

    /// Admit `body` under `key`, recording `launch` in its private directory.
    fn admit(
        host: &SandboxInstanceHost,
        key: &str,
        sandbox_id: &str,
        launch: u8,
        bindings: SandboxInstanceBindings,
        body: SandboxInstanceBody,
    ) -> std::result::Result<SandboxInstanceAdmitted, SandboxInstanceHostError> {
        host.admit(
            SandboxInstanceAdmission {
                key: key.to_string(),
                fingerprint: [launch; 32],
                sandbox_id: sandbox_id.to_string(),
            },
            |dir| {
                std::fs::write(dir.join("launch"), [launch])
                    .map_err(|error| SandboxInstanceHostError::Files(error.to_string()))?;
                Ok((bindings, body))
            },
        )
    }

    fn private_host() -> (tempfile::TempDir, Arc<SandboxInstanceHost>) {
        let root = tempfile::tempdir().unwrap();
        let host = SandboxInstanceHost::new(
            root.path().join("instances"),
            tokio::runtime::Handle::current(),
        )
        .unwrap();
        (root, Arc::new(host))
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn suspend_requests_are_isolated_and_retryable_between_hosted_instances() {
        let (_root, host) = private_host();
        let (observed, mut requests) = mpsc::channel(8);
        for key in ["alpha", "beta"] {
            let observed = observed.clone();
            let body: SandboxInstanceBody = Box::new(move |mut controls| {
                Box::pin(async move {
                    controls.readiness.send_replace(true);
                    loop {
                        tokio::select! {
                            Some(()) = futures::StreamExt::next(&mut controls.suspend) => {
                                // Leave this instance running, as after a failed
                                // suspension, so a later request can retry it.
                                observed.send(key).await.unwrap();
                            }
                            () = &mut controls.shutdown => return Ok(0),
                        }
                    }
                })
            });
            admit(&host, key, key, 1, unbound(), body).unwrap();
            wait_supervising(&host, key).await;
        }
        assert!(!host.request_suspend("missing"));
        for key in ["alpha", "alpha", "beta"] {
            assert!(host.request_suspend(key));
            assert_eq!(
                tokio::time::timeout(Duration::from_secs(1), requests.recv())
                    .await
                    .unwrap(),
                Some(key)
            );
        }
        assert!(
            requests.try_recv().is_err(),
            "requests must never reach another instance"
        );
        for key in ["alpha", "beta"] {
            assert!(host.status(key).unwrap().ready);
        }
        for key in ["alpha", "beta"] {
            host.remove(key, Duration::from_secs(1)).await;
        }
        assert!(!host.request_suspend("alpha"));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn one_removal_holds_an_idempotent_admission_until_its_files_are_gone() {
        let (root, host) = private_host();
        let dir = root.path().join("instances").join("generation");
        let winding_down = |code| -> SandboxInstanceBody {
            Box::new(move |controls| {
                Box::pin(async move {
                    controls.readiness.send_replace(true);
                    controls.shutdown.await;
                    // Winding down takes time; the key must stay taken meanwhile.
                    tokio::time::sleep(Duration::from_millis(300)).await;
                    Ok(code)
                })
            })
        };
        let admit_launch = |launch, code| {
            admit(
                &host,
                "generation",
                &format!("sandbox-{launch}"),
                launch,
                unbound(),
                winding_down(code),
            )
        };
        assert_eq!(admit_launch(1, 3), Ok(SandboxInstanceAdmitted::Created));
        assert_eq!(admit_launch(1, 3), Ok(SandboxInstanceAdmitted::Unchanged));
        assert_eq!(admit_launch(2, 3), Err(SandboxInstanceHostError::Conflict));
        assert_eq!(
            admit(
                &host,
                "../escape",
                "sandbox-1",
                1,
                unbound(),
                winding_down(0)
            ),
            Err(SandboxInstanceHostError::InvalidKey)
        );
        wait_supervising(&host, "generation").await;

        // A caller that gives up, such as an expired RPC, strands nothing.
        assert!(
            tokio::time::timeout(
                Duration::from_millis(20),
                host.remove("generation", Duration::from_secs(10))
            )
            .await
            .is_err()
        );
        tokio::time::timeout(Duration::from_secs(10), async {
            while host.status("generation").map(|status| status.phase)
                != Some(SandboxInstancePhaseReport::Stopping)
            {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("instance observes the stop request");
        assert_eq!(admit_launch(1, 3), Err(SandboxInstanceHostError::Removing));
        assert!(dir.join("launch").exists());
        let removers = [(); 2].map(|()| {
            let host = host.clone();
            tokio::spawn(async move { host.remove("generation", Duration::from_secs(10)).await })
        });
        for remover in removers {
            let status = remover.await.unwrap().expect("instance was hosted");
            assert_eq!(status.sandbox_id, "sandbox-1");
            assert_eq!(
                status.phase,
                SandboxInstancePhaseReport::Exited { exit_code: 3 }
            );
        }
        assert!(host.status("generation").is_none());
        assert!(!dir.exists());

        // The key is free again, and the earlier removal cannot touch its new owner.
        assert_eq!(admit_launch(4, 4), Ok(SandboxInstanceAdmitted::Created));
        wait_supervising(&host, "generation").await;
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(
            host.status("generation").map(|status| status.phase),
            Some(SandboxInstancePhaseReport::Supervising)
        );
        assert_eq!(std::fs::read(dir.join("launch")).unwrap(), [4]);

        // Files that cannot be deleted keep the key reserved until a later
        // removal deletes them. Root ignores the permission this relies on.
        if !nix::unistd::geteuid().is_root() {
            use std::os::unix::fs::PermissionsExt as _;

            let instances = root.path().join("instances");
            std::fs::set_permissions(&instances, std::fs::Permissions::from_mode(0o500)).unwrap();
            let status = host
                .remove("generation", Duration::from_secs(10))
                .await
                .unwrap();
            assert!(
                matches!(&status.phase, SandboxInstancePhaseReport::Failed { error } if error.contains("private files remain")),
                "{status:?}"
            );
            assert_eq!(host.status("generation"), Some(status));
            assert_eq!(admit_launch(5, 5), Err(SandboxInstanceHostError::Removing));
            std::fs::set_permissions(&instances, std::fs::Permissions::from_mode(0o700)).unwrap();
        }
        let status = host.remove("generation", Duration::from_secs(10)).await;
        assert_eq!(
            status.map(|status| status.phase),
            Some(SandboxInstancePhaseReport::Exited { exit_code: 4 })
        );
        assert!(!dir.exists());
        assert!(host.list().is_empty());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn relinquishing_ends_supervision_without_stopping_the_workload() {
        let (root, host) = private_host();
        let dir = root.path().join("instances").join("relinquished");
        let credentials = GatewayCredentials::from_gateway_token("relinquished-token").unwrap();
        // Only the stop path terminates the workload.
        let (terminated_tx, mut terminated) = mpsc::channel::<()>(1);
        let body: SandboxInstanceBody = Box::new(move |controls| {
            Box::pin(async move {
                controls.readiness.send_replace(true);
                controls.shutdown.await;
                let _ = terminated_tx.send(()).await;
                Ok(0)
            })
        });
        admit(
            &host,
            "relinquished",
            "sandbox-relinquished",
            1,
            SandboxInstanceBindings {
                gateway_credentials: credentials.clone(),
                ..unbound()
            },
            body,
        )
        .unwrap();
        wait_supervising(&host, "relinquished").await;

        let status = host
            .relinquish("relinquished")
            .await
            .expect("instance was hosted");
        assert!(!status.ready);
        assert!(status.phase.is_terminal(), "{status:?}");
        assert_eq!(
            terminated.try_recv(),
            Err(mpsc::error::TryRecvError::Disconnected),
            "relinquishing must never run the workload's stop path"
        );
        assert!(host.status("relinquished").is_none());
        assert!(!dir.exists());
        let revoked = GatewayEndpoint::new("http://127.0.0.1:9", credentials);
        let error = openshell_core::grpc_client::fetch_settings_snapshot(&revoked, "relinquished")
            .await
            .unwrap_err();
        assert!(error.to_string().contains("revoked"), "{error}");
        // The key is free for a later term of the same instance.
        assert_eq!(
            admit(
                &host,
                "relinquished",
                "sandbox-relinquished",
                2,
                unbound(),
                Box::new(|controls| Box::pin(async move {
                    controls.shutdown.await;
                    Ok(0)
                })),
            ),
            Ok(SandboxInstanceAdmitted::Created)
        );
        host.relinquish_all().await;
        assert!(host.list().is_empty());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn panicking_instance_fails_closed_and_revokes_its_credentials() {
        let (_root, host) = private_host();
        let credentials = GatewayCredentials::from_gateway_token("panicking-token").unwrap();
        let panicking: SandboxInstanceBody = Box::new(|controls| {
            Box::pin(async move {
                controls.readiness.send_replace(true);
                tokio::task::yield_now().await;
                panic!("hosted sandbox instance failure");
            })
        });
        admit(
            &host,
            "panicking",
            "sandbox-panicking",
            1,
            SandboxInstanceBindings {
                gateway_credentials: credentials.clone(),
                ..unbound()
            },
            panicking,
        )
        .unwrap();
        let status = tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                if let Some(status) = host.status("panicking")
                    && status.phase.is_terminal()
                {
                    return status;
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("panicking instance reports an end");
        assert_eq!(
            status,
            SandboxInstanceStatusReport {
                route: "panicking".to_string(),
                sandbox_id: "sandbox-panicking".to_string(),
                phase: SandboxInstancePhaseReport::Failed {
                    error: "sandbox instance panicked".to_string(),
                },
                ready: false,
            }
        );
        let revoked = GatewayEndpoint::new("http://127.0.0.1:9", credentials);
        let error = openshell_core::grpc_client::fetch_settings_snapshot(&revoked, "panicking")
            .await
            .unwrap_err();
        assert!(error.to_string().contains("revoked"), "{error}");
        assert!(
            host.remove("panicking", Duration::from_secs(1))
                .await
                .is_some()
        );
    }

    /// Teardown must not return before force-reaped tasks have dropped.
    ///
    /// The instance's exit guard revokes its credentials as the root task
    /// unwinds, and `Removal` deletes the private directory and frees the key
    /// as soon as [`InstanceSpawner::reap`] returns. A tracked task running
    /// outside the cooperative wrapper cannot observe cancellation, so only
    /// the abort after the first bound ends it; `reap` must wait a fresh
    /// window for that abort to take effect instead of reusing the deadline
    /// that already elapsed.
    #[tokio::test(flavor = "multi_thread")]
    async fn reaping_waits_for_aborted_tasks_to_drop_before_removal() {
        struct NotifyOnDrop(Arc<std::sync::atomic::AtomicBool>);

        impl Drop for NotifyOnDrop {
            fn drop(&mut self) {
                // Drop work (revoking credentials and closing what the task
                // owns) takes real time, so a reap that returns at the first
                // bound cannot have observed it.
                std::thread::sleep(Duration::from_millis(30));
                self.0.store(true, Ordering::SeqCst);
            }
        }

        let spawner = InstanceSpawner::new(tokio::runtime::Handle::current(), unbound());
        let dropped = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let (started_tx, started) = tokio::sync::oneshot::channel();
        let task_dropped = dropped.clone();
        // Spawn outside the cooperative wrapper, as the abort backstop exists
        // for: only the abort after the cooperative bound ends this task.
        let handle = spawner.tracker.spawn_on(
            async move {
                let _notify = NotifyOnDrop(task_dropped);
                let _ = started_tx.send(());
                std::future::pending::<()>().await;
            },
            &spawner.runtime,
        );
        spawner.register_abort(handle.abort_handle());
        started.await.unwrap();
        spawner.reap(Duration::from_millis(100)).await;
        assert!(
            dropped.load(Ordering::SeqCst),
            "reap returned before the aborted task dropped and revoked its credentials"
        );
    }

    /// The spawner enters every identity scope of its bindings, in tasks and in
    /// blocking work; work that bypasses it carries no identity.
    #[tokio::test(flavor = "multi_thread")]
    async fn the_spawner_scopes_tasks_and_blocking_work_to_the_instance() {
        const DENIAL: &str = "2026-05-06T17:02:00.000Z OCSF NET:OPEN [MED] DENIED curl(42) -> blocked.com:443 [policy:- engine:opa]";
        let (log_tx, mut logs) = mpsc::channel(16);
        let denial = SandboxInstanceDenialLog::default();
        let bindings = SandboxInstanceBindings {
            gateway_credentials: GatewayCredentials::from_gateway_token("spawner-token").unwrap(),
            event_context: Arc::new(crate::sandbox_instance_event_context(
                Some("sandbox-spawner"),
                Some("sandbox-spawner"),
                String::new(),
            )),
            ocsf_json: OcsfJsonSettings::default(),
            log_sink: Some(LogSink::new("sandbox-spawner", log_tx)),
            denial_log: DenialLogSource::SandboxInstance(denial.clone()),
        };
        let spawner = InstanceSpawner::new(tokio::runtime::Handle::current(), bindings);
        let dispatch = tracing::Dispatch::new(
            tracing_subscriber::registry()
                .with(openshell_supervisor_process::log_push::LogPushLayer::thread_bound()),
        );

        let task_dispatch = dispatch.clone();
        let (identity_tx, identity) = tokio::sync::oneshot::channel();
        spawner
            .spawn(async move {
                let scoped = openshell_ocsf::ctx::ctx().sandbox_id.clone();
                // A plain spawn is not re-scoped, even from inside a scoped
                // task, so its events stay unattributed.
                let plain = tokio::spawn(async { openshell_ocsf::ctx::ctx().sandbox_id.clone() })
                    .await
                    .unwrap();
                // Work spawned through the ambient instance spawner joins the
                // tree and keeps the instance's identity.
                let (adopted_tx, adopted) = tokio::sync::oneshot::channel();
                openshell_core::instance_tasks::spawn(async move {
                    let _ = adopted_tx.send(openshell_ocsf::ctx::ctx().sandbox_id.clone());
                })
                .await
                .unwrap();
                let adopted = adopted.await.unwrap();
                tracing::dispatcher::with_default(&task_dispatch, || {
                    tracing::info!(target: "spawner_test", "scoped event");
                });
                let mut writer = SandboxInstanceDenialWriter;
                std::io::Write::write_all(&mut writer, DENIAL.as_bytes()).unwrap();
                let _ = identity_tx.send((scoped, plain, adopted));
            })
            .await
            .unwrap();
        let (scoped, plain, adopted) = identity.await.unwrap();
        assert_eq!(scoped, "sandbox-spawner");
        assert_ne!(plain, "sandbox-spawner");
        assert_eq!(
            adopted, "sandbox-spawner",
            "an ambient instance spawn must keep the instance's identity"
        );
        assert_eq!(
            logs.recv().await.expect("scoped log line").sandbox_id,
            "sandbox-spawner"
        );
        assert!(
            format!("{denial:?}").contains("retained: 1"),
            "the scoped denial must reach the instance log: {denial:?}"
        );

        // A blocking closure cannot inherit the task scope, so the spawner
        // re-enters every scope synchronously.
        let blocking_dispatch = dispatch.clone();
        let blocking_identity = spawner
            .spawn_scoped_blocking(Box::new(move || {
                let scoped = openshell_ocsf::ctx::ctx().sandbox_id.clone();
                tracing::dispatcher::with_default(&blocking_dispatch, || {
                    tracing::info!(target: "spawner_test", "blocking event");
                });
                let mut writer = SandboxInstanceDenialWriter;
                std::io::Write::write_all(&mut writer, DENIAL.as_bytes()).unwrap();
                let output: ScopedBlockingOutput = Box::new(scoped);
                output
            }))
            .await
            .unwrap()
            .expect("the closure must not be skipped")
            .downcast::<String>()
            .expect("the closure returns its sandbox identity");
        assert_eq!(*blocking_identity, "sandbox-spawner");
        assert_eq!(
            logs.recv().await.expect("blocking log line").message,
            "blocking event"
        );
        assert!(
            format!("{denial:?}").contains("retained: 2"),
            "the blocking denial must reach the instance log: {denial:?}"
        );
    }

    /// An instance-body spawn routed through the ambient instance spawner
    /// joins the instance's task tree, so removal reaps it instead of leaving
    /// it ticking on the shared runtime, and cancellation ends it at its next
    /// await point rather than burning the cooperative bound.
    #[tokio::test(flavor = "multi_thread")]
    async fn removal_reaps_an_adopted_spawn_in_an_instance_body() {
        struct ReapedOnDrop(Arc<std::sync::atomic::AtomicBool>);

        impl Drop for ReapedOnDrop {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }

        let (_root, host) = private_host();
        let ticks = Arc::new(AtomicU64::new(0));
        let reaped = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let task_ticks = ticks.clone();
        let task_reaped = reaped.clone();
        let body: SandboxInstanceBody = Box::new(move |controls| {
            Box::pin(async move {
                // Adopted: the ambient instance spawner owns this task.
                openshell_core::instance_tasks::spawn(async move {
                    let _reaped = ReapedOnDrop(task_reaped);
                    loop {
                        task_ticks.fetch_add(1, Ordering::SeqCst);
                        tokio::time::sleep(Duration::from_millis(5)).await;
                    }
                });
                controls.readiness.send_replace(true);
                controls.shutdown.await;
                Ok(0)
            })
        });
        admit(&host, "adopted", "sandbox-adopted", 1, unbound(), body).unwrap();
        wait_supervising(&host, "adopted").await;
        while ticks.load(Ordering::SeqCst) == 0 {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }

        let started = tokio::time::Instant::now();
        let stopped = host
            .remove("adopted", Duration::from_secs(10))
            .await
            .expect("instance was hosted");
        assert_eq!(
            stopped.phase,
            SandboxInstancePhaseReport::Exited { exit_code: 0 }
        );
        assert!(
            started.elapsed() < RUNTIME_SHUTDOWN_TIMEOUT / 2,
            "removing an instance with an adopted task took {:?}, not well under the cooperative bound",
            started.elapsed()
        );
        let after_removal = ticks.load(Ordering::SeqCst);
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(
            ticks.load(Ordering::SeqCst),
            after_removal,
            "an adopted spawn must be reaped with its instance"
        );
        assert!(
            reaped.load(Ordering::SeqCst),
            "reaping an adopted spawn must drop it, running its guards"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn hosted_instances_keep_identity_credentials_and_lifecycle_separate() {
        let (gateway_url, captured) = capture_gateway().await;
        let (_root, host) = private_host();
        let mut alpha = start_instance(&host, &gateway_url, "sandbox-alpha");
        let mut beta = start_instance(&host, &gateway_url, "sandbox-beta");
        wait_supervising(&host, "sandbox-alpha").await;
        wait_supervising(&host, "sandbox-beta").await;
        assert_eq!(
            admit(
                &host,
                "sandbox-alpha",
                "sandbox-alpha",
                1,
                unbound(),
                Box::new(|_| Box::pin(async { Ok(0) })),
            ),
            Err(SandboxInstanceHostError::Conflict)
        );

        // A task an instance spawns must never inherit another instance's
        // identity.
        for observed in [(&mut alpha.identity).await, (&mut beta.identity).await] {
            let observed = observed.unwrap();
            assert!(!["sandbox-alpha", "sandbox-beta"].contains(&observed.as_str()));
        }
        // Other tests in this process may set the process-wide context; a
        // thread outside every instance must still carry no instance identity.
        let outside = openshell_ocsf::ctx::ctx().sandbox_id.clone();
        assert!(!["sandbox-alpha", "sandbox-beta"].contains(&outside.as_str()));
        assert_eq!(
            bearers_for(&captured, "sandbox-alpha"),
            ["Bearer token-sandbox-alpha"]
        );
        assert_eq!(
            bearers_for(&captured, "sandbox-beta"),
            ["Bearer token-sandbox-beta"]
        );
        // An event must reach its own instance's log stream and never another's.
        for (instance, sandbox) in [(&mut alpha, "sandbox-alpha"), (&mut beta, "sandbox-beta")] {
            let line = tokio::time::timeout(Duration::from_secs(10), instance.logs.recv())
                .await
                .expect("instance log line")
                .expect("instance log stream stays open");
            assert_eq!(line.sandbox_id, sandbox);
            assert_eq!(line.message, format!("emitted by {sandbox}"));
            assert!(
                instance.logs.try_recv().is_err(),
                "no line may cross instances"
            );
        }

        let stopped = host
            .remove("sandbox-alpha", Duration::from_secs(10))
            .await
            .expect("alpha was hosted");
        assert_eq!(
            stopped.phase,
            SandboxInstancePhaseReport::Exited { exit_code: 0 }
        );
        assert!(!stopped.ready);
        assert!(host.status("sandbox-alpha").is_none());
        let revoked = GatewayEndpoint::new(gateway_url.as_str(), alpha.credentials.clone());
        let error = openshell_core::grpc_client::fetch_settings_snapshot(&revoked, "sandbox-alpha")
            .await
            .unwrap_err();
        assert!(error.to_string().contains("revoked"), "{error}");

        beta.call_again.send(()).await.unwrap();
        tokio::time::timeout(Duration::from_secs(10), async {
            while bearers_for(&captured, "sandbox-beta").len() < 2 {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("beta keeps calling the gateway");
        assert_eq!(
            bearers_for(&captured, "sandbox-beta"),
            ["Bearer token-sandbox-beta", "Bearer token-sandbox-beta"]
        );
        assert_eq!(
            host.status("sandbox-beta").map(|status| status.phase),
            Some(SandboxInstancePhaseReport::Supervising)
        );

        let stopped = host.remove("sandbox-beta", Duration::from_secs(10)).await;
        assert_eq!(
            stopped.map(|status| status.phase),
            Some(SandboxInstancePhaseReport::Exited { exit_code: 0 })
        );
    }

    /// Emit `events` events from `instance`'s tracked child task and check that
    /// all of them carry that instance's identity.
    async fn emit_and_check(instance: &mut ChattyInstance, events: usize) {
        for _ in 0..events {
            instance.emit.send(()).await.unwrap();
        }
        for _ in 0..events {
            let identity = tokio::time::timeout(Duration::from_secs(10), instance.observed.recv())
                .await
                .expect("busy task reports its identity")
                .expect("identity stream stays open");
            assert_eq!(identity, instance.key);
            let line = tokio::time::timeout(Duration::from_secs(10), instance.logs.recv())
                .await
                .expect("busy task emits a log line")
                .expect("log stream stays open");
            assert_eq!(line.sandbox_id, instance.key);
            assert!(
                line.message.contains(&instance.key),
                "a log line from {} carried another instance's message: {line:?}",
                instance.key
            );
        }
    }

    /// One instance busy while another idles, then swapped, must never let an
    /// event reach the wrong instance.
    ///
    /// Thread-bound identity cannot catch this: both instances share the
    /// runtime's workers, so a task that read a worker's binding could emit
    /// under whichever instance last ran there. The test alternates which
    /// instance is busy and asserts that the idle one receives nothing and
    /// that every event carries the busy instance's identity.
    #[tokio::test(flavor = "multi_thread")]
    async fn busy_and_idle_instances_never_exchange_events() {
        let (_root, host) = private_host();
        let mut alpha = start_chatty_instance(&host, "chatty-alpha");
        let mut beta = start_chatty_instance(&host, "chatty-beta");
        wait_supervising(&host, "chatty-alpha").await;
        wait_supervising(&host, "chatty-beta").await;

        for round in 0..8 {
            let (busy, idle) = if round % 2 == 0 {
                (&mut alpha, &mut beta)
            } else {
                (&mut beta, &mut alpha)
            };
            emit_and_check(busy, 2).await;
            assert!(
                idle.logs.try_recv().is_err(),
                "an idle instance received the busy instance's event"
            );
        }

        // Each instance retained only the denials its own tasks emitted.
        assert!(
            format!("{:?}", alpha.denial).contains("retained: 8"),
            "{:?}",
            alpha.denial
        );
        assert!(
            format!("{:?}", beta.denial).contains("retained: 8"),
            "{:?}",
            beta.denial
        );

        host.remove("chatty-alpha", Duration::from_secs(10)).await;
        host.remove("chatty-beta", Duration::from_secs(10)).await;
    }

    /// Removal must join the instance's task tree and return the process to
    /// its descriptor and socket baseline.
    ///
    /// Unlike the [`capture_gateway`] tests, this opens a real supervisor
    /// session to a real tonic server, so client and server connection tasks
    /// are alive when the instance is removed. The loopback stub still
    /// under-counts production: it has no DNS, keepalive, or relay paths.
    #[cfg(target_os = "linux")]
    #[tokio::test(flavor = "multi_thread")]
    async fn removal_joins_the_task_tree_and_releases_its_descriptors() {
        let (gateway_url, _server) = loopback_gateway().await;
        let (_root, host) = private_host();
        tokio::time::sleep(Duration::from_millis(100)).await;
        let baseline_fds = crate::proc_census::fd_count().expect("read /proc/self/fd");
        let baseline_sockets = crate::proc_census::socket_count().expect("read /proc/self/fd");

        let mut established = Vec::new();
        let mut credentials = Vec::new();
        for index in 0..2 {
            let sandbox = format!("sandbox-session-{index}");
            let (established_tx, established_rx) = tokio::sync::oneshot::channel();
            established.push(established_rx);
            credentials.push(start_session_instance(
                &host,
                &gateway_url,
                &sandbox,
                established_tx,
            ));
            wait_supervising(&host, &sandbox).await;
        }
        for session in established {
            session.await.expect("the supervisor session is accepted");
        }
        let working_fds = crate::proc_census::fd_count().expect("read /proc/self/fd");
        let working_sockets = crate::proc_census::socket_count().expect("read /proc/self/fd");
        assert!(
            working_fds >= baseline_fds + 2 && working_sockets >= baseline_sockets + 2,
            "the sessions must hold real descriptors: fds {working_fds} vs {baseline_fds}, sockets {working_sockets} vs {baseline_sockets}"
        );

        let spawners = lock(&host.instances)
            .values()
            .map(|instance| instance.spawner.clone())
            .collect::<Vec<_>>();
        assert_eq!(spawners.len(), 2);

        for index in 0..2 {
            let sandbox = format!("sandbox-session-{index}");
            let status = host
                .remove(&sandbox, Duration::from_secs(10))
                .await
                .expect("instance was hosted");
            assert!(status.phase.is_terminal(), "{status:?}");
        }
        assert!(host.list().is_empty());

        // Removal waited for every tracked task, including the session task.
        for spawner in &spawners {
            assert!(
                spawner.tracker.is_empty(),
                "removal returned with tracked instance tasks still alive"
            );
            let aborts = spawner
                .aborts
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            assert!(
                aborts.handles.iter().all(AbortHandle::is_finished),
                "removal returned with live abortable tasks"
            );
        }

        // The closed sessions must return their descriptors and sockets.
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        loop {
            let fds = crate::proc_census::fd_count().expect("read /proc/self/fd");
            let sockets = crate::proc_census::socket_count().expect("read /proc/self/fd");
            if fds <= baseline_fds && sockets <= baseline_sockets {
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "removal leaked descriptors or sockets: fds {fds} vs {baseline_fds}, sockets {sockets} vs {baseline_sockets}"
            );
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        let final_fds = crate::proc_census::fd_count().expect("read /proc/self/fd");
        let final_sockets = crate::proc_census::socket_count().expect("read /proc/self/fd");
        assert!(
            final_fds < working_fds && final_sockets < working_sockets,
            "removal must release the sessions' descriptors: fds {final_fds} vs {working_fds}, sockets {final_sockets} vs {working_sockets}"
        );

        for (index, credentials) in credentials.into_iter().enumerate() {
            let revoked = GatewayEndpoint::new(gateway_url.as_str(), credentials);
            let error = openshell_core::grpc_client::fetch_settings_snapshot(
                &revoked,
                &format!("sandbox-session-{index}"),
            )
            .await
            .unwrap_err();
            assert!(error.to_string().contains("revoked"), "{error}");
        }
    }

    /// Removal ends an adopted task that ignores shutdown and the abort watch
    /// at its next await point, and revokes the credentials.
    ///
    /// The task never observes the shutdown signal or the abort watch, so only
    /// the instance's cancellation ends it. Cancellation is cooperative, so
    /// removal drops it and runs its guards well within the cooperative bound
    /// instead of burning it before the force-reap abort.
    #[tokio::test(flavor = "multi_thread")]
    async fn removal_ends_a_stuck_task_cooperatively_and_revokes_credentials() {
        struct DroppedOnExit(Arc<std::sync::atomic::AtomicBool>);

        impl Drop for DroppedOnExit {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }

        let (_root, host) = private_host();
        let credentials = GatewayCredentials::from_gateway_token("stuck-token").unwrap();
        let dropped = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let (started_tx, started) = tokio::sync::oneshot::channel();
        let task_dropped = dropped.clone();
        let body: SandboxInstanceBody = Box::new(move |controls| {
            Box::pin(async move {
                openshell_core::instance_tasks::spawn(async move {
                    let _dropped = DroppedOnExit(task_dropped);
                    let _ = started_tx.send(());
                    // Ignores both shutdown and the abort watch; only the
                    // instance's cancellation ends it.
                    std::future::pending::<()>().await;
                });
                controls.readiness.send_replace(true);
                controls.shutdown.await;
                Ok(0)
            })
        });
        admit(
            &host,
            "stuck",
            "sandbox-stuck",
            1,
            SandboxInstanceBindings {
                gateway_credentials: credentials.clone(),
                ..unbound()
            },
            body,
        )
        .unwrap();
        wait_supervising(&host, "stuck").await;
        started.await.unwrap();

        let started_at = tokio::time::Instant::now();
        let removed = tokio::time::timeout(
            RUNTIME_SHUTDOWN_TIMEOUT,
            host.remove("stuck", Duration::from_millis(100)),
        )
        .await
        .expect("removal finishes within its cooperative bound")
        .expect("instance was hosted");
        assert!(removed.phase.is_terminal(), "{removed:?}");
        assert!(
            started_at.elapsed() < RUNTIME_SHUTDOWN_TIMEOUT / 2,
            "removal took {:?}, not well under the cooperative bound",
            started_at.elapsed()
        );
        assert!(
            dropped.load(Ordering::SeqCst),
            "removal must drop a stuck task at its next await point"
        );
        assert!(host.status("stuck").is_none());

        let revoked = GatewayEndpoint::new("http://127.0.0.1:9", credentials);
        let error = openshell_core::grpc_client::fetch_settings_snapshot(&revoked, "sandbox-stuck")
            .await
            .unwrap_err();
        assert!(error.to_string().contains("revoked"), "{error}");
    }

    /// A workload that returns on its own ends the instance's remaining tasks,
    /// even though the assignment is still desired and no removal runs.
    ///
    /// The instance stays hosted with a terminal phase, so the gateway can
    /// keep reporting it; only its work must end, as it did when each
    /// instance had its own runtime.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_returning_workload_ends_the_instances_adopted_tasks() {
        struct DroppedOnExit(Arc<std::sync::atomic::AtomicBool>);

        impl Drop for DroppedOnExit {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }

        let (_root, host) = private_host();
        let credentials = GatewayCredentials::from_gateway_token("exit-token").unwrap();
        let dropped = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let (started_tx, started) = tokio::sync::oneshot::channel();
        let task_dropped = dropped.clone();
        let body: SandboxInstanceBody = Box::new(move |controls| {
            Box::pin(async move {
                // Adopted: the ambient instance spawner owns this task, and it
                // would run forever if nothing ended it.
                openshell_core::instance_tasks::spawn(async move {
                    let _dropped = DroppedOnExit(task_dropped);
                    let _ = started_tx.send(());
                    std::future::pending::<()>().await;
                });
                // Let the adopted task run before the workload returns.
                let _ = started.await;
                controls.readiness.send_replace(true);
                Ok(0)
            })
        });
        admit(
            &host,
            "returning",
            "sandbox-returning",
            1,
            SandboxInstanceBindings {
                gateway_credentials: credentials.clone(),
                ..unbound()
            },
            body,
        )
        .unwrap();

        tokio::time::timeout(Duration::from_secs(10), async {
            while !dropped.load(Ordering::SeqCst) {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("a returning workload must end its adopted tasks");

        let status = host
            .status("returning")
            .expect("the still-desired assignment stays hosted");
        assert_eq!(
            status.phase,
            SandboxInstancePhaseReport::Exited { exit_code: 0 }
        );
        let spawner = lock(&host.instances)
            .get("returning")
            .expect("the assignment stays hosted")
            .spawner
            .clone();
        tokio::time::timeout(Duration::from_secs(10), async {
            while !spawner.tracker.is_empty() {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("the instance's task tree empties when its root returns");

        let revoked = GatewayEndpoint::new("http://127.0.0.1:9", credentials);
        let error =
            openshell_core::grpc_client::fetch_settings_snapshot(&revoked, "sandbox-returning")
                .await
                .unwrap_err();
        assert!(error.to_string().contains("revoked"), "{error}");
    }

    /// One shared runtime, not one per instance: adding instances must not add
    /// a thread per instance.
    ///
    /// Slack is the instance runtime's blocking-pool ceiling
    /// (`2 * INSTANCE_HOST_WORKER_THREADS`): workers exist when the runtime is
    /// built, so the only legitimate growth between the two samples is the
    /// blocking pool filling. The bound is
    /// `INSTANCE_HOST_WORKER_THREADS + SLACK = 12`, while a per-instance
    /// thread leak would add 16 between 16 and 32 instances and fail it.
    #[cfg(target_os = "linux")]
    #[tokio::test(flavor = "multi_thread")]
    async fn doubling_the_instance_count_does_not_add_a_thread_per_instance() {
        const SLACK: usize = 2 * INSTANCE_HOST_WORKER_THREADS;

        let (gateway_url, _captured) = capture_gateway().await;
        let root = tempfile::tempdir().unwrap();
        let runtime = instance_host_runtime().expect("build the shared instance runtime");
        let host = Arc::new(
            SandboxInstanceHost::new(root.path().join("instances"), runtime.handle().clone())
                .unwrap(),
        );
        for index in 0..16 {
            let sandbox = format!("sandbox-{index:04}");
            // The stub's receivers are unused; the host keeps the instance.
            drop(start_instance(&host, &gateway_url, &sandbox));
            wait_supervising(&host, &sandbox).await;
        }
        let (sixteen, settled) = settled_threads().await;
        assert!(settled, "thread census did not settle at 16 instances");

        for index in 16..32 {
            let sandbox = format!("sandbox-{index:04}");
            drop(start_instance(&host, &gateway_url, &sandbox));
            wait_supervising(&host, &sandbox).await;
        }
        let (thirty_two, settled) = settled_threads().await;
        assert!(settled, "thread census did not settle at 32 instances");

        let growth = thirty_two
            .threads
            .len()
            .saturating_sub(sixteen.threads.len());
        assert!(
            thirty_two.threads.len()
                <= sixteen.threads.len() + INSTANCE_HOST_WORKER_THREADS + SLACK,
            "hosting 32 instances used {growth} more threads than 16 (bound {}); a per-instance thread would add 16",
            INSTANCE_HOST_WORKER_THREADS + SLACK,
        );

        host.relinquish_all().await;
        runtime.shutdown_background();
    }

    /// Wait until the thread census is unchanged across two samples.
    ///
    /// Returns the last census and whether it stabilized within the sampling
    /// budget. A sample that lost a tid to churn counts as movement even when
    /// the surviving thread count is unchanged, and so does a replaced thread:
    /// the census compares whole threads, not just how many there are.
    #[cfg(target_os = "linux")]
    async fn settled_threads() -> (crate::proc_census::ThreadCensus, bool) {
        let mut previous = crate::proc_census::threads().expect("read /proc/self/task");
        let mut stable = 0;
        for _ in 0..40 {
            tokio::time::sleep(Duration::from_millis(250)).await;
            let current = crate::proc_census::threads().expect("read /proc/self/task");
            if current.threads == previous.threads && current.skipped == 0 {
                stable += 1;
                if stable == 2 {
                    return (current, true);
                }
            } else {
                stable = 0;
            }
            previous = current;
        }
        (previous, false)
    }

    /// Measure the process cost of hosting N sandbox instances.
    ///
    /// Prints one JSON census line with the thread count and per-thread `comm`
    /// names, descriptor and socket counts, and `VmRSS` for the whole test
    /// process after the instances settle. It asserts nothing about the
    /// counts: absolute values depend on the host and on everything else the
    /// test process runs.
    ///
    /// Run with:
    /// `cargo test -p openshell-supervisor --features experimental shared_supervisor_cost_baseline -- --ignored --nocapture --test-threads=1`
    ///
    /// Set `OPENSHELL_TEST_INSTANCE_COUNT` to change N (default 8) and
    /// `OPENSHELL_TEST_SETTLE_SECONDS` to change the fixed wait (default 5).
    #[cfg(target_os = "linux")]
    #[tokio::test(flavor = "multi_thread")]
    #[ignore = "manual shared-supervisor cost baseline; counts depend on the host"]
    async fn shared_supervisor_cost_baseline() {
        let count = std::env::var("OPENSHELL_TEST_INSTANCE_COUNT")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(8);
        let settle_seconds = std::env::var("OPENSHELL_TEST_SETTLE_SECONDS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(5);
        let (gateway_url, _captured) = capture_gateway().await;
        let (_root, host) = private_host();
        for index in 0..count {
            let sandbox = format!("sandbox-{index:04}");
            // The test stub's receivers are unused; dropping them leaves the
            // instance body running under the host's own records.
            drop(start_instance(&host, &gateway_url, &sandbox));
            wait_supervising(&host, &sandbox).await;
        }
        tokio::time::sleep(Duration::from_secs(settle_seconds)).await;
        let (threads, settled) = settled_threads().await;
        let thread_names: HashMap<_, _> = threads
            .threads
            .iter()
            .map(|(tid, name)| (tid.to_string(), name.clone()))
            .collect();
        println!(
            "shared_supervisor_cost_baseline {}",
            serde_json::json!({
                "instances": count,
                "settle_seconds": settle_seconds,
                "settled": settled,
                "thread_count": threads.threads.len(),
                "thread_names": thread_names,
                "fd_count": crate::proc_census::fd_count().expect("read /proc/self/fd"),
                "socket_count": crate::proc_census::socket_count().expect("read /proc/self/fd"),
                "vm_rss_kib": crate::proc_census::vm_rss_kib().expect("read /proc/self/status"),
            })
        );
        host.relinquish_all().await;
    }

    /// Ambient work spawned after its owning instance is gone is dropped
    /// rather than leaked onto the shared runtime untracked.
    #[tokio::test(flavor = "multi_thread")]
    async fn ambient_spawns_after_the_instance_is_gone_are_dropped() {
        let spawner = InstanceSpawner::new(tokio::runtime::Handle::current(), unbound());
        let ambient = spawner.ambient();
        // Dropping the last strong reference leaves only the ambient handle's
        // weak one, as after the instance is removed.
        drop(spawner);

        let ran = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let task_ran = ran.clone();
        ambient
            .spawn(async move { task_ran.store(true, Ordering::SeqCst) })
            .await
            .expect("the discarded handle completes");
        assert!(
            !ran.load(Ordering::SeqCst),
            "a task spawned after its instance is gone must be dropped"
        );

        let ran_blocking = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let closure_ran = ran_blocking.clone();
        let skipped = ambient
            .spawn_blocking(move || closure_ran.store(true, Ordering::SeqCst))
            .await
            .expect("the discarded blocking handle completes");
        assert!(
            skipped.is_none(),
            "a blocking closure skipped after its instance is gone reports so"
        );
        assert!(
            !ran_blocking.load(Ordering::SeqCst),
            "a blocking closure spawned after its instance is gone must be dropped"
        );
    }

    /// The force-reap registry is bounded by live tasks instead of growing
    /// with every task the instance spawns over its lifetime.
    #[tokio::test(flavor = "multi_thread")]
    async fn the_abort_registry_stays_bounded_by_live_tasks() {
        let spawner = InstanceSpawner::new(tokio::runtime::Handle::current(), unbound());
        for _ in 0..64 {
            spawner.spawn(async {}).await.expect("task completes");
        }
        let aborts = spawner
            .aborts
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        assert_eq!(
            aborts.handles.len(),
            1,
            "the registry retained finished tasks: {} handles",
            aborts.handles.len()
        );
    }

    /// Blocking work still queued when its instance ends must not start.
    ///
    /// The pool is saturated by one running closure, so the second one is
    /// queued. Removal cancels the instance; when the pool frees the slot, the
    /// queued closure observes the cancellation and is skipped instead of
    /// running unattributed after the instance is gone.
    #[tokio::test(flavor = "multi_thread")]
    async fn queued_blocking_work_is_skipped_after_removal() {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .max_blocking_threads(1)
            .enable_all()
            .build()
            .unwrap();
        let spawner = InstanceSpawner::new(runtime.handle().clone(), unbound());

        // Occupy the pool's only thread so the next closure cannot start.
        let (occupied_tx, occupied) = tokio::sync::oneshot::channel();
        let (release_tx, release) = std::sync::mpsc::channel::<()>();
        runtime.spawn(async move {
            let _ = tokio::task::spawn_blocking(move || {
                let _ = occupied_tx.send(());
                let _ = release.recv();
            })
            .await;
        });
        occupied.await.expect("the pool thread is occupied");

        let ran = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let closure_ran = ran.clone();
        let handle = {
            let _context = runtime.enter();
            spawner.spawn_scoped_blocking(Box::new(move || {
                closure_ran.store(true, Ordering::SeqCst);
                let output: ScopedBlockingOutput = Box::new(());
                output
            }))
        };

        // Cancel the instance while the closure is still queued, then free the
        // pool so it would run if it did not check the cancellation.
        spawner.reap(Duration::from_millis(50)).await;
        release_tx.send(()).expect("release the pool thread");
        let outcome = handle.await.expect("the queued closure completes");
        assert!(
            outcome.is_none(),
            "a queued closure must report that it was skipped"
        );
        assert!(
            !ran.load(Ordering::SeqCst),
            "a queued blocking closure must not start after removal"
        );
        runtime.shutdown_background();
    }

    /// A blocking closure already running is joined before removal completes.
    ///
    /// Blocking work cannot be aborted, so without tracking it a closure
    /// mid-run would keep touching the instance's files after removal released
    /// the key and deleted the private directory. Removal waits for it within
    /// its bound, even though the caller dropped the handle.
    #[tokio::test(flavor = "multi_thread")]
    async fn reaping_joins_a_running_blocking_closure_before_removal() {
        let spawner = InstanceSpawner::new(tokio::runtime::Handle::current(), unbound());
        let started = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let finished = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let closure_started = started.clone();
        let closure_finished = finished.clone();
        // The caller drops the handle, as the fire-and-forget call sites do.
        drop(spawner.spawn_scoped_blocking(Box::new(move || {
            closure_started.store(true, Ordering::SeqCst);
            // Long enough that a reap which returns without joining cannot
            // observe the end of the closure.
            std::thread::sleep(Duration::from_millis(250));
            closure_finished.store(true, Ordering::SeqCst);
            let output: ScopedBlockingOutput = Box::new(());
            output
        })));
        while !started.load(Ordering::SeqCst) {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }

        spawner.reap(Duration::from_secs(5)).await;

        assert!(
            finished.load(Ordering::SeqCst),
            "reap returned before the running blocking closure finished"
        );
    }

    /// A prune while a closure is still running must keep it outstanding.
    ///
    /// Registering a second closure while the first one runs prunes the
    /// registry. Dropping the running closure there loses the signal reap joins
    /// on, so removal returns before that closure stops touching the instance's
    /// files. Both closures must be joined even though the second registration
    /// triggered a prune.
    #[tokio::test(flavor = "multi_thread")]
    async fn reaping_joins_every_running_blocking_closure_across_a_prune() {
        let spawner = InstanceSpawner::new(tokio::runtime::Handle::current(), unbound());
        let first_started = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let second_started = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let first_finished = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let second_finished = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let (release_first_tx, release_first) = std::sync::mpsc::channel::<()>();
        let (release_second_tx, release_second) = std::sync::mpsc::channel::<()>();

        let started = first_started.clone();
        let finished = first_finished.clone();
        // The caller drops the handle, as the fire-and-forget call sites do.
        drop(spawner.spawn_scoped_blocking(Box::new(move || {
            started.store(true, Ordering::SeqCst);
            let _ = release_first.recv();
            // Well after the second closure, so a reap that lost this entry
            // returns before the closure finishes.
            std::thread::sleep(Duration::from_millis(300));
            finished.store(true, Ordering::SeqCst);
            let output: ScopedBlockingOutput = Box::new(());
            output
        })));
        while !first_started.load(Ordering::SeqCst) {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }

        // Registering the second closure prunes the registry while the first is
        // still running.
        let started = second_started.clone();
        let finished = second_finished.clone();
        drop(spawner.spawn_scoped_blocking(Box::new(move || {
            started.store(true, Ordering::SeqCst);
            let _ = release_second.recv();
            std::thread::sleep(Duration::from_millis(20));
            finished.store(true, Ordering::SeqCst);
            let output: ScopedBlockingOutput = Box::new(());
            output
        })));
        while !second_started.load(Ordering::SeqCst) {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }

        release_first_tx
            .send(())
            .expect("release the first closure");
        release_second_tx
            .send(())
            .expect("release the second closure");
        spawner.reap(Duration::from_secs(5)).await;

        assert!(
            second_finished.load(Ordering::SeqCst),
            "reap returned before the second blocking closure finished"
        );
        assert!(
            first_finished.load(Ordering::SeqCst),
            "a prune dropped the running first closure, so reap did not join it"
        );
    }

    /// Each join round gets a fresh window, so a closure registered while an
    /// earlier round is joining is not bound by a deadline that round already
    /// spent.
    #[tokio::test(start_paused = true)]
    async fn join_blocking_gives_each_round_a_fresh_window() {
        let spawner = InstanceSpawner::new(tokio::runtime::Handle::current(), unbound());
        let timeout = Duration::from_secs(1);

        // The first closure finishes late in the first round, leaving only the
        // remainder of the shared deadline for the next one.
        let (first_done, first_completed) = tokio::sync::oneshot::channel();
        spawner.register_blocking(first_completed);
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(900)).await;
            let _ = first_done.send(());
        });

        // Register a second closure while that first round is joining. It
        // finishes after the first deadline (1000ms) but within a fresh second
        // window (900ms + 1000ms).
        let second_finished = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let register_spawner = spawner.clone();
        let register_finished = second_finished.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(1)).await;
            let (second_done, second_completed) = tokio::sync::oneshot::channel();
            register_spawner.register_blocking(second_completed);
            tokio::spawn(async move {
                tokio::time::sleep(Duration::from_millis(1499)).await;
                register_finished.store(true, Ordering::SeqCst);
                let _ = second_done.send(());
            });
        });

        let joining = tokio::spawn({
            let spawner = spawner.clone();
            async move { spawner.join_blocking(timeout).await }
        });
        joining.await.expect("the join task completes");

        assert!(
            second_finished.load(Ordering::SeqCst),
            "the second round reused the first round's spent deadline"
        );
    }

    /// The instance's cancellation token ends a root body that ignores its
    /// shutdown signal and the abort watch, and the fail-closed outcome
    /// revokes the credentials before the task tree is reaped.
    #[tokio::test(flavor = "multi_thread")]
    async fn cancelling_an_instance_ends_a_root_body_that_ignores_shutdown() {
        let credentials = GatewayCredentials::from_gateway_token("cancel-token").unwrap();
        let spawner = InstanceSpawner::new(
            tokio::runtime::Handle::current(),
            SandboxInstanceBindings {
                gateway_credentials: credentials.clone(),
                ..unbound()
            },
        );
        let (_stop, stop_requested) = watch::channel(false);
        let (_suspend, suspend_requested) = mpsc::channel(1);
        // Held for the whole test and never set, so only the instance's
        // cancellation token can end the root task.
        let (_abort, abort_requested) = watch::channel(false);
        let (phase_tx, phase) = watch::channel(SandboxInstancePhaseReport::Starting);
        let (ready_tx, ready) = watch::channel(false);
        let exit = SandboxInstanceExit {
            credentials: credentials.clone(),
            phase: phase_tx.clone(),
            ready: ready_tx.clone(),
            outcome: None,
            changes: ChangeNotifier(Arc::new(watch::Sender::new(0))),
            spawner: Arc::downgrade(&spawner),
        };
        let (started_tx, started) = tokio::sync::oneshot::channel();
        let body: SandboxInstanceBody = Box::new(move |controls| {
            Box::pin(async move {
                let SandboxInstanceControls { shutdown, .. } = controls;
                // Ignore the shutdown signal; the body never returns.
                drop(shutdown);
                let _ = started_tx.send(());
                std::future::pending::<()>().await;
                Ok(0)
            })
        });
        let task_spawner = spawner.clone();
        let _root = spawner.spawn(run_hosted_sandbox_instance(
            exit,
            task_spawner,
            body,
            SandboxInstanceSignals {
                stop_requested,
                suspend_requested,
                abort_requested,
                phase: phase_tx,
                ready: ready_tx,
                changes: ChangeNotifier(Arc::new(watch::Sender::new(0))),
            },
        ));
        started.await.expect("the instance body runs");

        spawner.reap(Duration::from_secs(1)).await;

        assert_eq!(
            *phase.borrow(),
            SandboxInstancePhaseReport::Failed {
                error: "sandbox instance was cancelled before it finished".to_string(),
            }
        );
        assert!(!*ready.borrow(), "a cancelled instance is never ready");
        assert!(
            spawner.tracker.is_empty(),
            "cancelling the instance must end its whole task tree"
        );

        let revoked = GatewayEndpoint::new("http://127.0.0.1:9", credentials);
        let error =
            openshell_core::grpc_client::fetch_settings_snapshot(&revoked, "sandbox-cancelled")
                .await
                .unwrap_err();
        assert!(error.to_string().contains("revoked"), "{error}");
    }

    const WRITER_DENIAL: &str = "2026-05-06T17:02:00.000Z OCSF NET:OPEN [MED] DENIED curl(42) -> writer.example:443 [policy:- engine:opa]";

    /// A denial written inside a task scope reaches that scope's log, so an
    /// instance's tasks record into its own log wherever they run.
    #[test]
    fn a_denial_written_inside_a_task_scope_reaches_its_log() {
        let log = SandboxInstanceDenialLog::default();
        let scoped = log.clone();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(scope_denial_log(scoped, async {
            let mut writer = SandboxInstanceDenialWriter;
            std::io::Write::write_all(&mut writer, WRITER_DENIAL.as_bytes()).unwrap();
        }));
        assert!(
            format!("{log:?}").contains("retained: 1"),
            "the task scope must record the denial: {log:?}"
        );
    }

    /// A denial written outside every task scope is dropped, so unbound work
    /// records into no instance's log.
    #[test]
    fn a_denial_written_outside_any_task_scope_is_dropped() {
        let log = SandboxInstanceDenialLog::default();
        let mut writer = SandboxInstanceDenialWriter;
        std::io::Write::write_all(&mut writer, WRITER_DENIAL.as_bytes()).unwrap();
        assert!(
            format!("{log:?}").contains("retained: 0"),
            "an unscoped denial must reach no instance's log: {log:?}"
        );
    }
}
