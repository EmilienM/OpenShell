---
authors:
  - "@EmilienM"
state: review
links:
  - https://github.com/ai-openshell/openshell/issues/5957
  - https://github.com/ai-openshell/openshell/issues/5660
---

# RFC 0025 - Shared instance runtime and task-scoped sandbox identity

## Summary

Host every sandbox instance of a shared supervisor on one multi-thread runtime with bounded workers, instead of giving each instance its own OS thread and private two-worker runtime. Sandbox identity moves from thread-local bindings to per-crate task scopes that a scoped spawner enters for every instance task, and removal joins or force-reaps the instance's task tree. Idle thread cost stops scaling with sandbox count while logs, OCSF attribution, and fail-closed credential revocation keep working.

## Motivation

A shared supervisor runs up to `maxInstancesPerMember` sandboxes in one process. Each hosted instance currently owns a thread and a private two-worker tokio runtime, so an idle 128-sandbox supervisor holds hundreds of threads and their stacks (#5660). A one-worker runtime per instance would still cost a runtime and a thread per sandbox, missing the acceptance criterion of threads per idle sandbox well below four.

Keeping thread-local identity is not an option either. Tasks on a shared runtime migrate across workers, so a worker's binding describes whichever instance ran there last, not the task reading it.

## Proposal

**One bounded runtime.** `SandboxInstanceHost` owns one multi-thread runtime, built at shared-mode startup and kept separate from the control-plane runtime so instance load cannot starve the reconciler, gateway attachment, or health listener. `INSTANCE_HOST_WORKER_THREADS` is 4, capped by available parallelism, and `max_blocking_threads` is a stated 256. The blocking pool is deliberately far above the worker count because every instance shares it: a bound near the worker count would let one CPU-heavy sandbox's blocking work (for example its egress authorization) occupy every slot and starve another's. Idle blocking threads cost nothing because tokio retires them after about ten seconds, so the loaded ceiling is 4 workers plus 256 blocking threads, against the old world's per-instance pools with tokio's 512 default each. Tokio starts replacement workers under load, so loaded thread counts are not flat: the ceiling is a stated number, not a promise about every moment. Per-instance fairness inside the shared pool is a recorded follow-up (a per-instance semaphore or a deadline on the egress-authorization path), not solved here.

**Task-scoped identity and the scoped spawner.** The four identity slots stay split per owning crate, because `openshell-ocsf` cannot read a task-local defined in the supervisor: OCSF event context, OCSF JSON settings, and log-push sink each gain a `tokio::task_local!` and lose their thread binding, resolving the task scope first and then the process-wide value (the layer's own flag for JSON settings), so dedicated mode is unchanged. The denial log never had a process-wide value: `SandboxInstanceDenialWriter` resolves through the task scope alone and drops lines from work outside every instance's task scope. Each crate exposes an async scope and a `sync_scope` variant for blocking closures.

`InstanceSpawner` wraps `TaskTracker`, a `CancellationToken`, per-task abort handles, and every crate's scope. Per-instance code in crates that cannot name the spawner calls `openshell_core::instance_tasks::{spawn, spawn_blocking, run_blocking}`, which find the ambient instance through a task-local backend and fall back to plain tokio outside one. Adoption covers the supervisor, process, network, and middleware crates, the sandbox backend, and `grpc_client` credential renewal; a bare spawn in an instance body would lose both identity and tracking.

The ambient backend is a deliberate deviation from the original plan, which threaded an explicit spawner through every constructor. That would have touched constructors across six crates to reach a handful of spawn sites. The tradeoff is that the fallback is silent: a spawn made outside an instance scope still works, so an untracked spawn is possible by construction rather than a compile error. The adoption audit and a debug line when a late spawn finds its instance gone are the mitigations. The audit therefore covers every way instance-reachable code can start a task, not only `tokio::spawn`: `JoinSet::spawn`, `Handle::spawn`, and hyper's `Executor::execute`, because the app ingress starts connection, websocket-relay, and connection-driver tasks that way. The ingress's HTTP/2 server executor and its backend relay executor both delegate to the tracked spawner, so the whole per-request `proxy` future, not only a connection task, joins the instance tree and its identity scopes instead of running untracked on plain tokio.

The spike, pinned by `task_scope_spike` against tokio 1.53.3, established four facts: a task scope does not cross a plain spawn; `spawn_blocking` does not see the enclosing scope; tracing layer callbacks do see the emitting task's scope; and `sync_scope` reaches a blocking closure. Spawns are therefore re-scoped and blocking work re-enters scopes synchronously.

**Teardown.** The exit guard lives inside the instance root task, so panic unwind and abort both drop it and revoke credentials. Removal cancels the instance token, which ends each tracked task at its next await point, running its Drop guards; the root keeps its own cancellation branch so it reports why the instance ended instead of the generic exit. Removal closes and joins the tracker, repeats a bounded abort pass for whatever did not observe the token so a task registered while a pass runs is still reaped, then joins blocking closures already running within a final window, because abort cannot reach blocking work. A blocking closure still queued when the instance ends observes the cancellation and is skipped instead of starting after removal. A normal removal therefore completes without spending the full cooperative bound; a task that ignores cancellation costs the cooperative plus abort windows, and blocking work that outlives its join window is the documented residual. The key stays reserved until the task tree and blocking work are gone and the private directory is deleted.

## Risks

- First-party per-instance spawns are adopted, not only library-internal ones: the app ingress's per-connection `JoinSet` children, its websocket-upgrade relay and hyper connection-driver executor tasks, the middleware websocket pump, and `grpc_client` credential renewal all route through the ambient instance, so they are tracked and identity-bound. Library-internal tasks (tonic pools, hyper-util, DNS internals) remain untracked and emit without instance identity, where thread bindings attributed them; their owning objects drop with the instance, so this is a recorded attribution change, not a leak.
- The ambient fallback is silent by design: a spawn outside an instance scope runs plainly, so the adoption audit, not the type system, keeps untracked spawns out.
- A task that ignores cancellation costs up to twice the five-second removal bound, and blocking work that outlives its join window adds a third.
- Bounded workers introduce head-of-line blocking under CPU-heavy instance load, and one sandbox's blocking work can still occupy shared blocking-pool slots; per-instance fairness is a follow-up.
- The cleanup test drives a real loopback gateway and supervisor session but no production DNS, keepalive, or relay paths, so it under-counts production descriptors and sockets.

## Alternatives

- One worker thread per instance, or a `current_thread` runtime per instance: fails the below-four threshold, and `block_in_place` in the L7 proxy requires multi-thread anyway.
- Keeping thread-locals and binding workers per task: wrong by construction, because tasks migrate.
- Per-event explicit routing for every emission site: correct but invasive; it remains the fallback if task scopes fail, with the spawner shrinking to tracking only.

## Open questions

- Should per-instance fairness inside the shared blocking pool be enforced with a semaphore or with a deadline on the egress-authorization path? Tracked on #5660 as a follow-up.
- Should "sandbox instance thread ended unexpectedly" be renamed now that instances have no thread? Kept verbatim for compatibility.
