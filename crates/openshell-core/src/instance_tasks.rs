// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Ambient task spawning for sandbox instances.
//!
//! A hosted sandbox instance's code lives in several crates that must not
//! depend on the instance host. Instead of threading a spawner through every
//! constructor, the host installs an [`InstanceTaskBackend`] in a task scope
//! around the instance's root task; [`spawn`], [`spawn_blocking`], and
//! [`run_blocking`] look it up and route the work through it.
//!
//! Tasks spawned this way are tracked by the owning instance and enter every
//! identity scope it defines, even when a plain `tokio::spawn` would have lost
//! both. Outside an instance (a dedicated supervisor, the control plane, or a
//! unit test) the same calls fall back to plain tokio, so call sites behave as
//! before when no instance owns them.

use std::any::Any;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use tokio::task::{JoinError, JoinHandle};

/// A task future as the owning instance spawns it.
pub type ScopedTask = Pin<Box<dyn Future<Output = ()> + Send + 'static>>;

/// Value a blocking closure returns, boxed so the ambient backend can hand it
/// back through the join handle without knowing its type. Callers downcast it
/// to the type they spawned.
pub type ScopedBlockingOutput = Box<dyn Any + Send + 'static>;

/// A blocking closure as the owning instance runs it.
pub type ScopedBlockingTask = Box<dyn FnOnce() -> ScopedBlockingOutput + Send + 'static>;

/// Owner of one sandbox instance's tasks.
///
/// Implemented by the instance host. Every method runs the work on the
/// instance's runtime, tracked by its task tree, and inside the instance's
/// identity scopes; the ambient task scope is re-entered by the callers in this
/// module, not by the backend.
///
/// # Outside an instance scope
///
/// [`spawn`], [`spawn_blocking`], and [`run_blocking`] fall back to plain tokio
/// when no instance owns the caller, so the control plane, dedicated
/// supervisors, and unit tests keep their current behavior. Only work spawned
/// inside an instance's task scope is routed here.
pub trait InstanceTaskBackend: Send + Sync + 'static {
    /// Spawn `task` tracked by the instance's task tree.
    fn spawn_scoped(&self, task: ScopedTask) -> JoinHandle<()>;
    /// Run `task` blocking, with the instance's identity scopes re-entered
    /// synchronously. Blocking work cannot be aborted; it is not tracked.
    ///
    /// The handle yields the closure's boxed value, or `None` when the
    /// instance ended before the queued closure could start.
    fn spawn_scoped_blocking(
        &self,
        task: ScopedBlockingTask,
    ) -> JoinHandle<Option<ScopedBlockingOutput>>;

    /// Spawn `task` tracked by the instance's task tree, without the
    /// cancellation wrapper [`spawn_scoped`](Self::spawn_scoped) may apply.
    ///
    /// The task observes [`cancelled`](Self::cancelled) itself, so it can
    /// finish work that must not be abandoned, such as closing a session,
    /// before the instance's tree is reaped. The default defers to
    /// [`spawn_scoped`](Self::spawn_scoped), which is correct for a backend
    /// that does not end its tasks early; a backend that does must override
    /// this and spawn the task as-is.
    fn spawn_owning_cancellation(&self, task: ScopedTask) -> JoinHandle<()> {
        self.spawn_scoped(task)
    }

    /// Resolve when the owning instance ends.
    ///
    /// The default never resolves, matching a backend that does not end its
    /// tasks early.
    fn cancelled(&self) -> Pin<Box<dyn Future<Output = ()> + Send>> {
        Box::pin(std::future::pending())
    }
}

tokio::task_local! {
    /// Backend of the sandbox instance whose task is currently running.
    static CURRENT: Arc<dyn InstanceTaskBackend>;
}

/// Handle to the sandbox instance that owns the current task, if any.
#[derive(Clone)]
pub struct InstanceTasks(Arc<dyn InstanceTaskBackend>);

impl InstanceTasks {
    #[must_use]
    pub fn new(backend: Arc<dyn InstanceTaskBackend>) -> Self {
        Self(backend)
    }

    /// The instance owning the current task, when running inside one.
    #[must_use]
    pub fn current() -> Option<Self> {
        CURRENT.try_with(Clone::clone).ok().map(Self)
    }

    /// Run `future` with this instance ambient, so tasks it spawns join the
    /// same tree.
    ///
    /// Takes the handle by value so the returned future owns it and can be
    /// spawned.
    pub async fn scope<F>(self, future: F) -> F::Output
    where
        F: Future + Send,
    {
        CURRENT.scope(self.0, future).await
    }

    /// Run `f` with this instance ambient, for blocking closures.
    pub fn scope_sync<F, T>(&self, f: F) -> T
    where
        F: FnOnce() -> T,
    {
        CURRENT.sync_scope(self.0.clone(), f)
    }

    /// Spawn `task` in this instance's tree.
    pub fn spawn<F>(&self, task: F) -> JoinHandle<()>
    where
        F: Future<Output = ()> + Send + 'static,
    {
        let ambient = self.clone();
        self.0
            .spawn_scoped(Box::pin(async move { ambient.scope(task).await }))
    }

    /// Resolve when this instance ends.
    pub async fn cancelled(&self) {
        self.0.cancelled().await;
    }

    /// Spawn `task` in this instance's tree without its cancellation wrapper.
    ///
    /// The task observes [`Self::cancelled`] and must finish its own cleanup,
    /// such as closing a session, before returning; the instance's teardown
    /// waits for it like any other tracked task.
    pub fn spawn_owning_cancellation<F>(&self, task: F) -> JoinHandle<()>
    where
        F: Future<Output = ()> + Send + 'static,
    {
        let ambient = self.clone();
        self.0
            .spawn_owning_cancellation(Box::pin(async move { ambient.scope(task).await }))
    }

    /// Run `func` blocking with this instance's identity, without tracking it.
    ///
    /// The closure's value travels back through the join handle, so callers
    /// downcast it to the type they spawned instead of paying for a second
    /// channel. The handle yields `None` when the instance ended before the
    /// queued closure started.
    pub fn spawn_blocking<F, T>(&self, func: F) -> JoinHandle<Option<ScopedBlockingOutput>>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        let ambient = self.clone();
        self.0.spawn_scoped_blocking(Box::new(move || {
            let value: ScopedBlockingOutput = Box::new(ambient.scope_sync(func));
            value
        }))
    }
}

/// Spawn `task` in the ambient instance's tree, or plainly when none owns the
/// caller.
pub fn spawn<F>(task: F) -> JoinHandle<()>
where
    F: Future<Output = ()> + Send + 'static,
{
    match InstanceTasks::current() {
        Some(instance) => instance.spawn(task),
        None => tokio::spawn(task),
    }
}

/// Spawn `task` in the ambient instance's tree without its cancellation
/// wrapper, or plainly when none owns the caller.
///
/// The task observes [`InstanceTasks::cancelled`] itself, so it can close what
/// it owns before the instance's tree is reaped.
pub fn spawn_owning_cancellation<F>(task: F) -> JoinHandle<()>
where
    F: Future<Output = ()> + Send + 'static,
{
    match InstanceTasks::current() {
        Some(instance) => instance.spawn_owning_cancellation(task),
        None => tokio::spawn(task),
    }
}

/// Spawn `task` when the caller is inside a runtime, in the ambient
/// instance's tree when one owns the caller.
///
/// `Drop` implementations can run on a thread with no runtime, where
/// `tokio::spawn` panics; those sites skip the spawn instead of draining.
/// Within a runtime this behaves as [`spawn`], so an instance-owned task stays
/// tracked and scoped.
pub fn spawn_if_in_runtime<F>(task: F) -> Option<JoinHandle<()>>
where
    F: Future<Output = ()> + Send + 'static,
{
    if tokio::runtime::Handle::try_current().is_err() {
        return None;
    }
    Some(spawn(task))
}

/// Run `func` blocking with the ambient instance's identity, or plainly when
/// none owns the caller.
///
/// The handle yields the closure's boxed value, or `None` when an instance
/// ended before the queued closure started.
pub fn spawn_blocking<F, T>(func: F) -> JoinHandle<Option<ScopedBlockingOutput>>
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    match InstanceTasks::current() {
        Some(instance) => instance.spawn_blocking(func),
        None => tokio::task::spawn_blocking(move || {
            let value: ScopedBlockingOutput = Box::new(func());
            Some(value)
        }),
    }
}

/// Why a [`run_blocking`] closure produced no value.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum BlockingTaskError {
    /// The closure panicked.
    #[error("the blocking task panicked")]
    Panicked,
    /// The task was cancelled before the closure returned.
    #[error("the blocking task was cancelled")]
    Cancelled,
    /// The closure's boxed value was not the type the caller spawned.
    ///
    /// In-tree backends box exactly the requested type, so this only fires for
    /// a backend that hands back a foreign value; it is not a closure panic.
    #[error("the blocking task returned an unexpected value type")]
    Mismatched,
}

impl From<JoinError> for BlockingTaskError {
    fn from(error: JoinError) -> Self {
        if error.is_panic() {
            Self::Panicked
        } else {
            Self::Cancelled
        }
    }
}

/// Run `func` blocking and wait for its value, preserving the ambient
/// instance's identity inside the closure.
///
/// Blocking work cannot be aborted, so the closure runs to completion even if
/// the caller is cancelled; the caller then drops the result.
///
/// # Errors
///
/// Fails when the closure panics or the task is cancelled. A queued closure
/// that never started because its instance ended is reported as cancelled.
pub async fn run_blocking<F, T>(func: F) -> Result<T, BlockingTaskError>
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    match InstanceTasks::current() {
        Some(instance) => match instance.spawn_blocking(func).await {
            Ok(Some(value)) => value
                .downcast::<T>()
                .map(|value| *value)
                .map_err(|_| BlockingTaskError::Mismatched),
            Ok(None) => Err(BlockingTaskError::Cancelled),
            Err(error) => Err(BlockingTaskError::from(error)),
        },
        None => tokio::task::spawn_blocking(func)
            .await
            .map_err(BlockingTaskError::from),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    /// Backend that records the work routed to it and runs it as-is, except
    /// when told to drop blocking closures, which models a backend that loses
    /// the work before it runs.
    #[derive(Default)]
    struct RecordingBackend {
        scoped: AtomicUsize,
        blocking: AtomicUsize,
        drop_blocking: AtomicBool,
    }

    impl RecordingBackend {
        fn dropping_blocking() -> Self {
            let backend = Self::default();
            backend.drop_blocking.store(true, Ordering::SeqCst);
            backend
        }

        fn scoped_spawns(&self) -> usize {
            self.scoped.load(Ordering::SeqCst)
        }

        fn blocking_spawns(&self) -> usize {
            self.blocking.load(Ordering::SeqCst)
        }
    }

    impl InstanceTaskBackend for RecordingBackend {
        fn spawn_scoped(&self, task: ScopedTask) -> JoinHandle<()> {
            self.scoped.fetch_add(1, Ordering::SeqCst);
            tokio::spawn(task)
        }

        fn spawn_scoped_blocking(
            &self,
            task: ScopedBlockingTask,
        ) -> JoinHandle<Option<ScopedBlockingOutput>> {
            self.blocking.fetch_add(1, Ordering::SeqCst);
            if self.drop_blocking.load(Ordering::SeqCst) {
                drop(task);
                return tokio::task::spawn_blocking(|| None);
            }
            tokio::task::spawn_blocking(move || Some(task()))
        }
    }

    /// The ambient instance is visible only where one owns the task.
    #[tokio::test]
    async fn current_is_none_outside_an_instance_and_some_inside_one() {
        assert!(InstanceTasks::current().is_none());

        let instance = InstanceTasks::new(Arc::new(RecordingBackend::default()));
        let inside = instance
            .scope(async { InstanceTasks::current().is_some() })
            .await;
        assert!(inside, "a scoped task must see its instance");

        assert!(
            InstanceTasks::current().is_none(),
            "the scope must not outlive the future it wraps"
        );
    }

    /// A spawn routed through the ambient instance joins its backend and keeps
    /// the instance visible to the spawned work.
    #[tokio::test]
    async fn spawn_inside_an_instance_is_routed_and_keeps_its_identity() {
        let backend = Arc::new(RecordingBackend::default());
        let instance = InstanceTasks::new(backend.clone());
        let observed = instance
            .scope(async {
                let (seen_tx, seen) = tokio::sync::oneshot::channel();
                spawn(async move {
                    let _ = seen_tx.send(InstanceTasks::current().is_some());
                });
                seen.await.expect("spawned task reports")
            })
            .await;

        assert!(observed, "an instance spawn must re-enter its scope");
        assert_eq!(backend.scoped_spawns(), 1);
    }

    /// Outside an instance the same call spawns plainly, so dedicated
    /// supervisors and unit code are unaffected.
    #[tokio::test]
    async fn spawn_outside_an_instance_falls_back_to_tokio() {
        let (seen_tx, seen) = tokio::sync::oneshot::channel();
        spawn(async move {
            let _ = seen_tx.send(InstanceTasks::current().is_some());
        });
        assert!(!seen.await.expect("spawned task reports"));
    }

    /// Blocking work routed through an instance re-enters the instance's
    /// identity synchronously, because task scopes do not reach it otherwise.
    #[tokio::test]
    async fn spawn_blocking_inside_an_instance_is_routed_and_scoped() {
        let backend = Arc::new(RecordingBackend::default());
        let instance = InstanceTasks::new(backend.clone());
        let observed = instance
            .scope(async {
                let (seen_tx, seen) = tokio::sync::oneshot::channel();
                let finished = spawn_blocking(move || {
                    let _ = seen_tx.send(InstanceTasks::current().is_some());
                })
                .await
                .expect("blocking closure runs");
                assert!(finished.is_some(), "the closure must not be skipped");
                seen.await.expect("blocking closure reports")
            })
            .await;

        assert!(observed, "blocking work must see its instance");
        assert_eq!(backend.blocking_spawns(), 1);
    }

    /// `run_blocking` returns the closure's value from the instance's backend,
    /// with the instance visible inside the closure.
    #[tokio::test]
    async fn run_blocking_inside_an_instance_is_routed_and_scoped() {
        let backend = Arc::new(RecordingBackend::default());
        let instance = InstanceTasks::new(backend.clone());
        let value = instance
            .scope(async { run_blocking(|| (InstanceTasks::current().is_some(), 7)).await })
            .await
            .expect("blocking closure runs");

        assert_eq!(value, (true, 7));
        assert_eq!(backend.blocking_spawns(), 1);
    }

    /// A panicking closure is reported as [`BlockingTaskError::Panicked`], not
    /// as a missing value.
    #[tokio::test]
    async fn run_blocking_maps_a_panicking_closure_inside_an_instance() {
        let instance = InstanceTasks::new(Arc::new(RecordingBackend::default()));
        let result = instance
            .scope(async { run_blocking::<_, ()>(|| panic!("closure failure")).await })
            .await;
        assert_eq!(result, Err(BlockingTaskError::Panicked));
    }

    /// A closure whose result never reaches the caller is reported as
    /// [`BlockingTaskError::Cancelled`], which callers on the egress
    /// authorization paths surface as a failed authorization.
    #[tokio::test]
    async fn run_blocking_maps_a_lost_closure_inside_an_instance() {
        let instance = InstanceTasks::new(Arc::new(RecordingBackend::dropping_blocking()));
        let result = instance.scope(async { run_blocking(|| {}).await }).await;
        assert_eq!(result, Err(BlockingTaskError::Cancelled));
    }

    /// Backend that hands back a value of a different type than the caller
    /// spawned, which is what a broken backend does.
    struct ForeignValueBackend;

    impl InstanceTaskBackend for ForeignValueBackend {
        fn spawn_scoped(&self, task: ScopedTask) -> JoinHandle<()> {
            tokio::spawn(task)
        }

        fn spawn_scoped_blocking(
            &self,
            _task: ScopedBlockingTask,
        ) -> JoinHandle<Option<ScopedBlockingOutput>> {
            tokio::task::spawn_blocking(|| {
                let value: ScopedBlockingOutput = Box::new(0_u8);
                Some(value)
            })
        }
    }

    /// A backend that hands back the wrong value type is reported distinctly,
    /// not as a closure panic.
    #[tokio::test]
    async fn run_blocking_maps_a_mismatched_value_inside_an_instance() {
        let instance = InstanceTasks::new(Arc::new(ForeignValueBackend));
        let result = instance
            .scope(async { run_blocking(String::new).await })
            .await;
        assert_eq!(result, Err(BlockingTaskError::Mismatched));
    }

    /// Without an instance, `run_blocking` runs on the caller's runtime and
    /// keeps tokio's join-error mapping for panics.
    #[tokio::test]
    async fn run_blocking_without_an_instance_uses_plain_tokio() {
        assert_eq!(run_blocking(|| 7).await, Ok(7));
        assert_eq!(
            run_blocking::<_, ()>(|| panic!("closure failure")).await,
            Err(BlockingTaskError::Panicked)
        );
    }
}
