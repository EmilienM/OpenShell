// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pins the tokio 1.53.3 task-scope facts the shared instance runtime rests on.
//!
//! Sandbox instances share one runtime, so instance identity cannot live on
//! worker threads. It moves to per-task scopes that a scoped spawner enters
//! for every task it starts. These tests are the evidence for that design:
//!
//! * a scope does not cross a plain spawn, so spawn sites must be re-scoped;
//! * `spawn_blocking` does not see the enclosing scope, which is why the
//!   scope helpers need the synchronous variant;
//! * tracing layers read a task-local of the emitting task, which is how
//!   OCSF and log-push attribution keep working without thread bindings;
//! * `sync_scope` makes a value visible inside a blocking closure.

use std::sync::{Arc, Mutex};

use tracing::Subscriber;
use tracing_subscriber::layer::{Context, Layer};

tokio::task_local! {
    /// Stand-in for the per-crate identity task-locals of the design.
    static SCOPE: u32;
}

/// Records the scope of each event it observes, the way the log-push and OCSF
/// layers read their task-local sources.
#[derive(Clone, Default)]
struct ScopeRecorder(Arc<Mutex<Vec<Option<u32>>>>);

impl<S: Subscriber> Layer<S> for ScopeRecorder {
    fn on_event(&self, _event: &tracing::Event<'_>, _context: Context<'_, S>) {
        self.0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(SCOPE.try_with(|value| *value).ok());
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_scope_does_not_cross_a_plain_spawn() {
    let seen = SCOPE
        .scope(7, async {
            assert_eq!(SCOPE.with(|value| *value), 7);
            tokio::spawn(async { SCOPE.try_with(|value| *value).ok() })
                .await
                .unwrap()
        })
        .await;
    assert_eq!(seen, None);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_scope_does_not_reach_spawn_blocking() {
    let seen = SCOPE
        .scope(7, async {
            tokio::task::spawn_blocking(|| SCOPE.try_with(|value| *value).ok())
                .await
                .unwrap()
        })
        .await;
    assert_eq!(seen, None);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn tracing_layers_read_the_task_local_of_the_emitting_task() {
    use tracing_subscriber::layer::SubscriberExt as _;

    let recorded = ScopeRecorder::default();
    let dispatch = tracing::Dispatch::new(tracing_subscriber::registry().with(recorded.clone()));
    SCOPE
        .scope(7, async {
            tracing::dispatcher::with_default(&dispatch, || {
                tracing::info!(target: "task_scope_spike", "scoped event");
            });
        })
        .await;
    assert_eq!(
        *recorded
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
        [Some(7)]
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sync_scope_reaches_a_blocking_closure() {
    let seen =
        tokio::task::spawn_blocking(|| SCOPE.sync_scope(9, || SCOPE.try_with(|value| *value).ok()))
            .await
            .unwrap();
    assert_eq!(seen, Some(9));
}
