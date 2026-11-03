# Roadmap

What is planned, and what was deliberately decided against. This is a working list, not a promise: items move, shrink or disappear. Finished items are deleted, and the history lives in the merged pull requests.

### Keeping up with deepagents

The library depends on deepagents' helper internals, so this is the most likely source of breakage.

- A scheduled CI run against the newest deepagents release, ignoring the `<0.8` cap, to learn early when a release breaks the library.
- A CI job that tests the floor (`uv pip install --resolution lowest-direct`), so `deepagents>=0.7.10` stays honest. Add its name to the branch protection ruleset.
- Upstream: [deepagents#7480](https://github.com/langchain-ai/deepagents/issues/7480), where `ls()` and `grep()` do not report clipped output and `ls()` ignores the exit code. When it is fixed: turn the two characterization tests into regression tests, remove the README bullet, and raise the floor.

### Tests

- A scripted fake chat model that drives a real agent through `glob`, `read_file`, `write_file`, `edit_file`, `ls` and `execute`. It needs no API key, and it would have caught the glob bug in CI.
- Provisioning semantics: labels, `max_lifetime_seconds` (a container exits on its own, and `stop()` does not wait out the grace period), image handling (pull failure, an image with its own `ENTRYPOINT`, a non-root `USER`), working directories, and volumes (read-only mounts, and the README's `DAC_OVERRIDE` claim).
- The two missing test images: one with its own `ENTRYPOINT`, one with a non-root `USER`.
- Concurrency: 20-30 simultaneous `execute()` calls on one sandbox (docker-py's default connection pool is 10).
- The README snippets as tests, so documentation drift fails CI.
- Hardening cleanups: a smaller memory test (64m), a faster pids test with a small limit, an exact `CapEff` assertion for `cap_add`, a cgroup read for the CPU limit, and the `HostConfig` normalizing helper noted as a TODO in `test_hardening_integration.py`.
- Large payloads (50-100 MB upload and download, output far beyond the limit), opt-in and not part of the default run.

### API and behavior

- `get_or_create()`: catch the name conflict from `provision()` and re-run the reuse check, which makes the creation race harmless.
- Optionally make the reuse check public (`verify(container, spec)`), for users who attach to containers themselves. Adding it is not a breaking change.
- A `runtime` field on `ContainerSpec` (for example `runsc` for gVisor), after the experiment below shows what holds under it.
- Busybox/Alpine support: `timeout` exits with 143 and no note there. Either keep documenting "GNU coreutils required" or make the timeout detection independent of the runtime.
- Network: the library switches networking on or off but cannot restrict it to specific hosts. A dedicated Docker network can be passed as the network mode, but that is unverified.

### Research

- **gVisor:** register `runsc` as the daemon's default runtime on a throwaway machine, run the whole integration suite, and record what fails. The hardening tests read `/proc`, cgroup limits and process groups, which gVisor reimplements.
- **Evaluation with a real model:** scenarios for task success, recovery from sandbox error messages, telling a limit from a bug, large outputs, and prompt injection (a poisoned file tells a cooperative agent to do something destructive). Opt-in (a marker or a separate directory), skipped without an API key, scored from container state and not by an LLM judge where possible, with repeated trials. A scripted attack battery run directly through `execute()` belongs in the normal integration tests instead.
- **containerd:** only if someone needs to run without a Docker daemon. It is a second backend behind a new abstraction, and the public API exposes docker-py types today. gVisor does not need it, since it works behind Docker and Podman.

### Publishing

- PyPI: check that the name is free, publish to TestPyPI first, and consider trusted publishing from CI.
- A changelog, once there is more than one release to describe.

## Not planned

Decisions that are made and should not be reopened without a new reason.

- **`get_or_create()` does not start or recreate a stopped container.** Raising can be loosened later (an opt-in `restart=True`), automatic restart cannot be tightened. It would also extend a "hard upper bound" on a container's life, and recreating would delete its files.
- **No "0 means no timeout".** Zero and negative timeouts and output limits are rejected. It never worked: the client-side wait still fired after about 7 seconds. Pass a large value instead.
- **`python3` is not mandatory for the whole library.** `execute()` and file transfer work without it, so `verify_python()` is optional on `ContainerSandbox` and on by default in `ManagedContainerSandbox`. A broken Python standard library is out of scope.
- **`find()` stays an unverified lookup.** It has no spec to compare against, recovery after a reuse error needs it, and verifying would remove the "bring your own container" route.
- **The "Docker did not answer" path has no integration test.** It cannot be provoked faithfully, and a unit test with a blocking fake covers the logic.
- **The two-layer timeout design stays.** The client-side layer exists for the case where the daemon hangs, and it cannot kill a command, which the README documents.