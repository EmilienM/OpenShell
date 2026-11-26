// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tracing layer that writes OCSF JSONL or marked console records to a writer.

use std::future::Future;
use std::io::Write;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use chrono::Utc;
use tracing::Subscriber;
use tracing_subscriber::Layer;
use tracing_subscriber::layer::Context;

use crate::tracing_layers::event_bridge::{OCSF_TARGET, clone_current_event};

/// OCSF JSON output settings owned by one hosted sandbox instance.
///
/// Instances apply their own `ocsf_json_enabled` setting to this value. Scoping
/// the instance's tasks to them makes every [`OcsfJsonlLayer`] use them for
/// events those tasks emit.
#[derive(Clone, Debug, Default)]
pub struct OcsfJsonSettings {
    pub enabled: Arc<AtomicBool>,
}

tokio::task_local! {
    /// JSON settings of the hosted sandbox instance whose task is being polled.
    static TASK_JSON_SETTINGS: OcsfJsonSettings;
}

/// Run `future` with its task's OCSF events using `settings`.
pub async fn scope_json_settings<F: Future>(settings: OcsfJsonSettings, future: F) -> F::Output {
    TASK_JSON_SETTINGS.scope(settings, future).await
}

/// Run the synchronous `f` with OCSF events using `settings`.
///
/// Tokio task scopes do not reach blocking closures, so an instance's
/// `spawn_blocking` work re-enters the scope with this variant.
pub fn scope_json_settings_sync<F, T>(settings: OcsfJsonSettings, f: F) -> T
where
    F: FnOnce() -> T,
{
    TASK_JSON_SETTINGS.sync_scope(settings, f)
}

/// Resolve the settings that apply to an event from its emitting task's scope.
fn scoped_json_settings() -> Option<OcsfJsonSettings> {
    TASK_JSON_SETTINGS.try_with(Clone::clone).ok()
}

/// A tracing `Layer` that intercepts OCSF events and writes JSONL output.
///
/// Only events with `target: "ocsf"` are processed; non-OCSF events are ignored.
///
/// An optional enabled flag (`Arc<AtomicBool>`) can be set via
/// [`with_enabled_flag`](Self::with_enabled_flag). When the flag is present and
/// `false`, the layer short-circuits without writing. This allows the sandbox
/// to hot-toggle OCSF JSONL output at runtime via the `ocsf_json_enabled`
/// setting without rebuilding the subscriber.
pub struct OcsfJsonlLayer<W: Write + Send + 'static> {
    writer: Mutex<W>,
    enabled: Option<Arc<AtomicBool>>,
    console_format: bool,
}

impl<W: Write + Send + 'static> OcsfJsonlLayer<W> {
    /// Create a new JSONL layer writing to the given writer.
    #[must_use]
    pub fn new(writer: W) -> Self {
        Self {
            writer: Mutex::new(writer),
            enabled: None,
            console_format: false,
        }
    }

    /// Attach a shared boolean flag that controls whether the layer writes.
    ///
    /// When the flag is `false`, the layer receives events but discards them.
    /// When the flag is absent (the default), the layer always writes.
    #[must_use]
    pub fn with_enabled_flag(mut self, flag: Arc<AtomicBool>) -> Self {
        self.enabled = Some(flag);
        self
    }

    /// Prefix each compact JSON record with a UTC timestamp and `OCSF-JSON`.
    ///
    /// Console collectors can match the marker and decode the remainder as
    /// JSON. File sinks retain plain JSONL unless this option is selected.
    #[must_use]
    pub fn with_console_format(mut self) -> Self {
        self.console_format = true;
        self
    }
}

impl<S, W> Layer<S> for OcsfJsonlLayer<W>
where
    S: Subscriber,
    W: Write + Send + 'static,
{
    fn on_event(&self, event: &tracing::Event<'_>, _ctx: Context<'_, S>) {
        if event.metadata().target() != OCSF_TARGET {
            return;
        }

        // A hosted sandbox instance's settings replace the process-wide ones.
        let instance = scoped_json_settings();
        let enabled = instance
            .as_ref()
            .map_or(self.enabled.as_ref(), |settings| Some(&settings.enabled));

        // If an enabled flag is set and it reads `false`, skip writing.
        if let Some(flag) = enabled
            && !flag.load(Ordering::Relaxed)
        {
            return;
        }

        if let Some(ocsf_event) = clone_current_event()
            && let Ok(mut w) = self.writer.lock()
        {
            let Ok(line) = ocsf_event.to_json_line() else {
                return;
            };
            let line = if self.console_format {
                let ts = Utc::now().format("%Y-%m-%dT%H:%M:%S%.3fZ");
                format!("{ts} OCSF-JSON {line}")
            } else {
                line
            };
            // Submit the entire record in one write, including its newline.
            let _ = w.write_all(line.as_bytes());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builders::{BaseEventBuilder, test_sandbox_context};
    use tracing_subscriber::layer::SubscriberExt as _;

    #[derive(Clone, Default)]
    struct SharedWriter(Arc<Mutex<Vec<u8>>>);

    impl Write for SharedWriter {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    fn emit(message: &str) {
        crate::ocsf_emit!(
            BaseEventBuilder::new(&test_sandbox_context())
                .message(message)
                .build()
        );
    }

    fn dispatch(writer: SharedWriter) -> tracing::Dispatch {
        tracing::Dispatch::new(tracing_subscriber::registry().with(OcsfJsonlLayer::new(writer)))
    }

    #[test]
    fn test_jsonl_layer_creation() {
        let buffer: Vec<u8> = Vec::new();
        let _layer = OcsfJsonlLayer::new(buffer);
    }

    #[tokio::test]
    async fn a_task_scope_controls_the_layer_and_does_not_cross_a_plain_spawn() {
        let writer = SharedWriter::default();
        let dispatch = dispatch(writer.clone());
        let disabled = OcsfJsonSettings {
            enabled: Arc::new(AtomicBool::new(false)),
        };

        scope_json_settings(disabled, async {
            tracing::dispatcher::with_default(&dispatch, || emit("hidden"));
        })
        .await;
        assert!(
            writer.0.lock().unwrap().is_empty(),
            "task-scoped settings must win over the layer's own flag"
        );

        // A plain spawn is not re-scoped, so it uses the layer's own flag.
        tokio::spawn(async move {
            tracing::dispatcher::with_default(&dispatch, || emit("unscoped"));
        })
        .await
        .unwrap();
        assert!(
            !writer.0.lock().unwrap().is_empty(),
            "unscoped events must not inherit an instance's settings"
        );
    }

    #[test]
    fn sync_scope_controls_the_layer_for_blocking_work() {
        let writer = SharedWriter::default();
        let dispatch = dispatch(writer.clone());
        let disabled = OcsfJsonSettings {
            enabled: Arc::new(AtomicBool::new(false)),
        };

        scope_json_settings_sync(disabled, || {
            tracing::dispatcher::with_default(&dispatch, || emit("hidden"));
        });
        assert!(writer.0.lock().unwrap().is_empty());

        scope_json_settings_sync(
            OcsfJsonSettings {
                enabled: Arc::new(AtomicBool::new(true)),
            },
            || {
                tracing::dispatcher::with_default(&dispatch, || emit("written"));
            },
        );
        assert!(!writer.0.lock().unwrap().is_empty());
    }
}
