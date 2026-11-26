// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Process-wide and sandbox-instance-task-bound [`EventContext`].
//!
//! A dedicated supervisor initialises the process-wide context once via
//! [`set_ctx`]. A process hosting several sandbox instances instead attributes
//! each instance's tasks to that instance: the instance's scoped spawner
//! enters [`scope_ctx`] for every task it starts, so attribution follows the
//! task when it migrates between worker threads.
//!
//! [`ctx`] resolves in precedence order: the task scope, then the process-wide
//! context. It falls back to a default context when none is set (e.g. unit
//! tests that exercise builders without booting the sandbox).

use crate::{ContextRef, EventContext, EventOrigin};
use std::future::Future;
use std::sync::{Arc, LazyLock, OnceLock};

static OCSF_CTX: OnceLock<EventContext> = OnceLock::new();

static OCSF_CTX_FALLBACK: LazyLock<EventContext> = LazyLock::new(|| EventContext {
    sandbox_id: String::new(),
    sandbox_name: String::new(),
    container_image: String::new(),
    container_image_uid: None,
    hostname: "test".to_string(),
    product_version: env!("CARGO_PKG_VERSION").to_string(),
    proxy_ip: std::net::IpAddr::from([127, 0, 0, 1]),
    proxy_port: 3128,
    origin: EventOrigin::Supervisor,
});

tokio::task_local! {
    /// Context of the hosted sandbox instance whose task is being polled.
    static TASK_CTX: Arc<EventContext>;
}

/// Initialise the process-wide OCSF sandbox context.
///
/// Returns `false` if the context was already set; the caller may log and
/// continue. Intended to be called exactly once during sandbox startup.
pub fn set_ctx(ctx: EventContext) -> bool {
    OCSF_CTX.set(ctx).is_ok()
}

/// Run `future` with events emitted by its task attributed to `ctx`.
///
/// A hosted sandbox instance enters this scope for every task it spawns, so
/// its identity travels with the task instead of with a worker thread.
pub async fn scope_ctx<F: Future>(ctx: Arc<EventContext>, future: F) -> F::Output {
    TASK_CTX.scope(ctx, future).await
}

/// Run the synchronous `f` with events attributed to `ctx`.
///
/// Tokio task scopes do not reach blocking closures, so an instance's
/// `spawn_blocking` work re-enters the scope with this variant.
pub fn scope_ctx_sync<F, T>(ctx: Arc<EventContext>, f: F) -> T
where
    F: FnOnce() -> T,
{
    TASK_CTX.sync_scope(ctx, f)
}

/// Return the context for events emitted by the calling task.
///
/// A task-scoped context wins over the process-wide context. Falls back to a
/// default context if none is set.
#[must_use]
pub fn ctx() -> ContextRef<'static> {
    if let Ok(bound) = TASK_CTX.try_with(Arc::clone) {
        return ContextRef::Shared(bound);
    }
    ContextRef::Borrowed(OCSF_CTX.get().unwrap_or(&OCSF_CTX_FALLBACK))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn context(sandbox_id: &str) -> Arc<EventContext> {
        Arc::new(EventContext {
            sandbox_id: sandbox_id.to_string(),
            sandbox_name: sandbox_id.to_string(),
            container_image: String::new(),
            container_image_uid: None,
            hostname: "test-host".to_string(),
            product_version: "0.0.0".to_string(),
            proxy_ip: std::net::IpAddr::V4(std::net::Ipv4Addr::LOCALHOST),
            proxy_port: 3128,
            origin: EventOrigin::Supervisor,
        })
    }

    #[test]
    fn fallback_context_names_no_sandbox() {
        assert!(OCSF_CTX_FALLBACK.container().is_none());
    }

    #[tokio::test]
    async fn a_task_scope_attributes_its_task_and_does_not_cross_a_plain_spawn() {
        let (scoped, plain) = scope_ctx(context("sandbox-scoped"), async {
            let scoped = ctx().sandbox_id.clone();
            // A plain spawn is not re-scoped, so an instance body's bare
            // `tokio::spawn` carries no identity until it spawns through the
            // instance's spawner.
            let plain = tokio::spawn(async { ctx().sandbox_id.clone() })
                .await
                .unwrap();
            (scoped, plain)
        })
        .await;
        assert_eq!(scoped, "sandbox-scoped");
        assert_ne!(plain, "sandbox-scoped");
    }

    #[test]
    fn sync_scope_attributes_blocking_work() {
        let seen = scope_ctx_sync(context("sandbox-blocking"), || ctx().sandbox_id.clone());
        assert_eq!(seen, "sandbox-blocking");
        assert_ne!(ctx().sandbox_id, "sandbox-blocking");
    }
}
