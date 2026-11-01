# agent-container-sandbox

Docker container sandbox for [LangChain Deep Agents](https://github.com/langchain-ai/deepagents): run commands and move files inside a container, with the container lifecycle kept separate from the sandbox itself.

- **`ContainerSandbox`** is a thin adapter around a container you already have. It runs commands and transfers files. It never creates, stops, or removes containers.
- **`ContainerProvisioner`** creates, finds, and deletes containers. It never runs your commands.
- **`ContainerSpec`** is an immutable description of the container you want.
- **`ManagedContainerSandbox`** is a context manager for scripts. It provisions a container, hands you a `ContainerSandbox`, and always deletes the container when the `with` block ends.

You own the Docker client. The library never closes it. You also own the container lifecycle, unless you hand it to `ManagedContainerSandbox`.

## Installation

```bash
uv add git+https://github.com/Konovor/agent-container-sandbox
```

**Prerequisites**

- Docker installed and running.
- An image that provides `sh`, `sleep`, and `timeout` (GNU coreutils). Very minimal images (for example `scratch`-based ones) will not work.
- `python3` in the image, if you use the file helpers inherited from `BaseSandbox` (`read`, `write`, `edit`, `ls`, `glob`). They run Python inside the container. Without it, `read`, `write`, `edit` and `glob` return an error, but `ls` returns an empty list instead of an error. `grep`, `execute`, `upload_files` and `download_files` do not need it (observed with deepagents 0.7.19 and 0.7.32). `ManagedContainerSandbox` checks for `python3` by default and fails early if it is missing.

The recommended image meets all of this:

```bash
docker pull ghcr.io/astral-sh/uv:0.10.0-python3.13-trixie-slim
```

Pulling beforehand is optional: the provisioner pulls a missing image automatically (this needs registry access and can make the first `provision()` slow). If the pull fails, it raises `DockerImageNotFound`.

## Quick start

```python
import docker

from agent_container_sandbox import ContainerSpec, ManagedContainerSandbox

IMAGE = "ghcr.io/astral-sh/uv:0.10.0-python3.13-trixie-slim"

client = docker.from_env()
try:
    with ManagedContainerSandbox(client, ContainerSpec(image=IMAGE)) as sandbox:
        result = sandbox.execute("echo 'Hello from your sandbox!'")
        print(result.output)  # Hello from your sandbox!
        print(result.exit_code)  # 0
finally:
    client.close()
```

The container is deleted when the block ends, even if the body raises. You still close the client yourself.

Pass any `docker.DockerClient`; see the docker-py documentation for remote and TLS setups.

### Manual lifecycle

Use the provisioner and sandbox directly when you need reuse, want to leave the container running, or need the `Container` object:

```python
import docker

from agent_container_sandbox import ContainerProvisioner, ContainerSandbox, ContainerSpec

IMAGE = "ghcr.io/astral-sh/uv:0.10.0-python3.13-trixie-slim"

client = docker.from_env()
provisioner = ContainerProvisioner(client)
container = provisioner.provision(ContainerSpec(image=IMAGE, name="my-sandbox"))

try:
    sandbox = ContainerSandbox(container)
    print(sandbox.execute("echo 'Hello from your sandbox!'").output)
finally:
    provisioner.delete(container)
    client.close()
```

Always delete the container in a `finally` block. If your process crashes, `max_lifetime_seconds` (see below) is the safety net.

## Using it with an agent

Any LangChain chat model works. This example uses Google Gemini, which requires `langchain-google-genai` and a `GOOGLE_API_KEY`.

```python
from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model

# `sandbox` is the ContainerSandbox from the snippets above
agent = create_deep_agent(
    model=init_chat_model("google_genai:gemini-3.5-flash-lite"),
    system_prompt="You are a coding assistant with sandbox access.",
    backend=sandbox,
)
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Write and run a script for Armstrong numbers"}]}
)
```

The container has no network by default, so tasks that need `pip install`, `git clone` or web requests require `ContainerSpec(image=IMAGE, network_mode="bridge")`.

For plain LangChain agents, pass the sandbox through `FilesystemMiddleware(backend=sandbox)` from `deepagents.middleware.filesystem`.

With `ManagedContainerSandbox`, keep the agent call inside the `with` block. The container is gone afterwards.

### Direct usage

`ContainerSandbox` inherits the file helpers from `BaseSandbox` (`read`, `write`, `ls`, `edit`, and so on) and adds binary transfer:

```python
sandbox.write("/workspace/test.txt", "Hello, World!")

sandbox.upload_files([("data.bin", b"\x00\x01\x02")])  # relative paths resolve to working_dir
files = sandbox.download_files(["data.bin"])
print(files[0].content)  # b'\x00\x01\x02'

result = sandbox.execute("sleep 10", timeout=2)
print(result.exit_code)  # 124
```

`upload_files` and `download_files` are intended for the developer, not the model: you control what enters the sandbox and what you extract.

## Container lifecycle

### `ContainerSpec`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `image` | `str` | required | Image name. Pulled automatically if missing locally. |
| `name` | `str \| None` | `None` | Container name. If omitted, a unique name is generated. Required for `find` and `get_or_create`, where it is the lookup key. |
| `volumes` | `dict` | `{}` | Mounts, e.g. `{"/host": {"bind": "/container", "mode": "rw"}}`. |
| `working_dir` | `str` | `/workspace` | Working directory. Created if missing. |
| `max_lifetime_seconds` | `int` | `3600` | Hard upper bound on the container's life (see below). |
| `labels` | `dict` | `{}` | Extra labels. The marker label below is always added. |
| `mem_limit` | `str \| None` | `"2g"` | Memory cap. Swap is capped at the same value. `None` = unlimited. |
| `cpus` | `float \| None` | `2.0` | CPU cap in cores. `None` = unlimited. |
| `network_mode` | `str \| None` | `"none"` | Container network mode. `"none"` disables networking. Use `"bridge"` for normal network access. `"host"` shares the host's network stack and removes the isolation. `None` sends nothing, so the daemon's default applies (usually full network access). |
| `pids_limit` | `int \| None` | `512` | Maximum processes and threads. `None` = unlimited. |
| `no_new_privileges` | `bool` | `True` | Block privilege gain through setuid binaries. |
| `cap_drop_all` | `bool` | `True` | Drop all Linux capabilities. |
| `cap_add` | `tuple[str, ...]` | `()` | Capabilities to add back after dropping, e.g. `("CHOWN", "DAC_OVERRIDE")`. |

All fields are keyword-only: write `ContainerSpec(image=IMAGE)`, not `ContainerSpec(IMAGE)`.

### `ContainerProvisioner`

| Method | Description |
|--------|-------------|
| `provision(spec)` | Create and start a new container. Cleans up after itself if setup fails. |
| `find(name)` | Return the container with exactly this name, or `None`. |
| `get_or_create(spec)` | Return the existing container named `spec.name`, or provision one. |
| `delete(container)` | Stop and remove a container. Tolerates it already being gone. |

**Reusing containers.** `get_or_create` lets several `ContainerSandbox` objects share one container, or attach to one that another process started. A sandbox over an attached container uses that container's own working directory, which may differ from your current spec. `ManagedContainerSandbox` does not reuse containers.

### `ManagedContainerSandbox`

A convenience layer over `ContainerProvisioner` and `ContainerSandbox`. It adds no capability of its own.

```python
with ManagedContainerSandbox(client, spec) as sandbox:
    ...  # sandbox is a ContainerSandbox
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `client` | `DockerClient` | required | Your Docker client. Never closed by this class. |
| `spec` | `ContainerSpec` | required | The container to provision. If `spec.name` is `None`, a unique name is generated. |
| `working_dir` | `str \| None` | container's configured working directory | Passed to `ContainerSandbox`. |
| `command_timeout` | `int` | `150` | Passed to `ContainerSandbox`. Must be positive. |
| `max_output_size` | `int` | `65536` | Passed to `ContainerSandbox`. Must be positive. |
| `verify_timeout` | `bool` | `True` | Run `verify_timeout()` on entry, so an image without a usable `timeout` binary fails early. |
| `verify_python` | `bool` | `True` | Run `verify_python()` on entry, so an image without `python3` fails early. Set it to `False` if you only use `execute` and file transfer. |

- **Cleanup is guaranteed.** The container is deleted on exit, including when the body raises. If setup fails after the container was created (for example `MissingTimeoutError` or `MissingPythonError`), it is deleted before the exception propagates.
- **Errors are never masked.** If deletion itself fails, a warning is logged and your original exception still propagates. The container then stops on its own after `max_lifetime_seconds`.
- **Single-use.** Entering the same instance twice raises `RuntimeError`. Create a new instance.
- **Fixed names can collide.** With an explicit `spec.name`, entering while a container of that name still exists fails with a docker-py error. Omit the name unless you need it.

### How the container is kept alive

The provisioner starts the container with a placeholder process, `sleep <max_lifetime_seconds>`, and real work arrives later through `exec_run`. When the sleep ends, the container stops on its own, even if your Python process crashed. This is a maximum lifetime, not an idle timeout (Docker has no idle TTL), so raise `max_lifetime_seconds` for long sessions.

### Hardening

Containers are hardened by default, to limit the damage an agent can do to the host by accident or by mistake. Every setting is a `ContainerSpec` field.

| Setting | Default | Protects against | What the agent sees when it hits the limit |
|---------|---------|------------------|--------------------------------------------|
| `mem_limit` (plus equal swap cap) | `2g` | A runaway allocation exhausting host memory | The process is killed: `Killed`, exit code `137` |
| `pids_limit` | `512` | Fork bombs exhausting the host's process table | `Cannot fork` errors from the shell |
| `cpus` | `2.0` | One container starving the rest of the host | Commands run slower, so they reach their timeout sooner |
| `network_mode` | `none` | Reaching your LAN, cloud metadata endpoints, and the internet | Name-resolution or connection errors, for example `Temporary failure in name resolution` |
| `no_new_privileges` | on | Privilege escalation through setuid binaries such as `sudo` | `sudo` and similar tools stop working |
| `cap_drop_all` | on | Kernel attack surface reachable through Linux capabilities | `Permission denied` on some root-only operations |

Mention these limits in your system prompt or tool description, so the model can tell an exceeded limit from a bug in its own code.

**Adjusting the defaults.** Set a field to `None` or `False` to remove a restriction, for example `ContainerSpec(image=IMAGE, mem_limit="8g", pids_limit=None)`. `pids_limit=None` sends an explicit unlimited value, so a daemon-wide default (Podman applies 2048, for example) cannot apply. For the network, `network_mode="bridge"` allows access, which reaches your LAN, cloud metadata endpoints, and the internet, and `None` leaves the daemon's default. `"host"` shares the host's network stack and removes the isolation entirely. With all capabilities dropped, root inside the container loses some abilities. The most likely surprise is writing to a mounted host directory owned by another user, which needs `DAC_OVERRIDE`. `apt-get` needs `SETUID`, `SETGID`, and `CHOWN`, plus a `network_mode` that allows network access. Add back only what you need with `cap_add=("CHOWN", "DAC_OVERRIDE")`.

**Verifying the setup.** Confirm that Docker applied the settings:

```python
sandbox.execute("grep -E 'CapEff|NoNewPrivs' /proc/self/status")
# CapEff: 0000000000000000, NoNewPrivs: 1   (with the default empty cap_add)
```

### Finding orphans

Every container created by this library carries the label `created-by=agent-container-sandbox`, even if you pass no labels. To list leftovers:

```bash
docker ps -a --filter label=created-by=agent-container-sandbox
```

## Execution behavior

### `ContainerSandbox`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `container` | `Container` | required | A running container. You own its lifecycle. |
| `working_dir` | `str \| None` | container's configured working directory (`/` if none) | Directory for commands and relative paths. If omitted, it is read from the container's configuration, which for provisioned containers is `ContainerSpec.working_dir`. If you pass it explicitly, it must already exist. |
| `command_timeout` | `int` | `150` | Default per-command timeout in seconds. Must be positive. |
| `max_output_size` | `int` | `65536` | Output beyond this many bytes is truncated. Must be positive. |

A non-positive `command_timeout` or `max_output_size` raises `ValueError` in the constructor, before any call to Docker. `execute()` returns the deepagents `ExecuteResponse` (`output`, `exit_code`, `truncated`).

- **`timeout` must be positive.** A per-call `timeout` of zero or less is rejected without running anything: `execute()` returns exit code `1` and a `Sandbox error:` message that names the value. Zero does not mean "no limit" here. For long-running commands, pass a large value instead.
- **Timeouts are enforced inside the container.** Each command runs as `timeout -k 2 N sh -c <command>`: SIGTERM at the deadline, SIGKILL 2 seconds later if the process is still running. The exit code is `124` after a normal timeout. It is `137` when SIGKILL was needed, which is also the code for an out-of-memory kill. Busybox-based images such as Alpine report `143` instead, without the "timed out" note. Processes that detach (`setsid`, double fork) can outlive `timeout`. Deleting the container is the only hard guarantee.
- **Docker-level hangs are reported separately.** If Docker itself does not answer within the timeout plus 7 seconds, you get exit code `124` and a message that the command's state is unknown and the sandbox may be unhealthy. The command may still be running.
- **Output is truncated by bytes**, without splitting a UTF-8 character, and `truncated` is set to `True`. Output is read fully into memory first, so avoid commands that print gigabytes.
- **Errors raised by the sandbox itself** are prefixed with `Sandbox error:` in `output`, to tell them apart from errors printed by your command.

`download_files` reports problems per file in `FileDownloadResponse.error`: `"file_not_found"`, `"is_directory"`, or a message string for anything else. A container that no longer exists produces a message string, not `"file_not_found"`. `upload_files` reports failures in `FileUploadResponse.error`. Neither raises for a single bad file.

`upload_files` never replaces an existing directory. If the target path is a directory, that file gets a per-file error and the directory is left untouched.

### Checking a container

Every `execute()` call needs `timeout` inside the container. `ContainerProvisioner` does not check for it, because its own keep-alive command does not use it. If your image might lack `timeout`, or you hand `ContainerSandbox` a container you did not provision, check once up front:

```python
sandbox = ContainerSandbox(container)
sandbox.verify_timeout()  # raises MissingTimeoutError if `timeout` is unusable
sandbox.verify_python()  # raises MissingPythonError if `python3` is unusable
```

Without the `timeout` check, every `execute()` call fails with exit code `127` and an "executable file not found in $PATH" message from the container runtime in `output`. `execute()` does not raise, so an agent would see this on every command and may misread it as a problem with its own code. `verify_timeout()` turns it into a single `MissingTimeoutError` at setup time.

`verify_python()` matters for the file helpers `read`, `write`, `edit`, `ls` and `glob`, which run Python inside the container. Without it, `ls` returns an empty list instead of an error, so an agent can conclude that a directory is empty. `execute()` and file transfer do not need Python, so `verify_python()` is a separate, optional call.

`ManagedContainerSandbox` runs both checks on entry by default, `timeout` first. Turn either off with `verify_timeout=False` or `verify_python=False`.

## Errors

Everything the library raises on purpose inherits from `SandboxError`:

```
SandboxError
├── DockerImageNotFound   image missing and could not be pulled (including auth or rate-limit failures); has an .image attribute
├── MissingTimeoutError   the container cannot run commands under `timeout`
└── MissingPythonError    the container has no usable `python3`
```

```python
from agent_container_sandbox import DockerImageNotFound

try:
    container = provisioner.provision(spec)
except DockerImageNotFound as e:
    print(f"Could not get image {e.image}: {e}")
```

Other failures, such as a container that cannot start or a daemon that is unreachable, surface as the underlying `docker-py` exceptions. Entering a `ManagedContainerSandbox` twice raises a plain `RuntimeError`, and a non-positive `command_timeout` or `max_output_size` raises a plain `ValueError`, because both are programming errors.

## Known limitations

- **`get_or_create` reuses by name only.** It does not check that an existing container's image, volumes, working directory, or hardening settings match the spec, and it does not check that the container is running. A container created earlier without limits is returned as is.
- **`get_or_create` is not safe against races.** Two callers creating the same name at the same moment can conflict.
- **`ls` and `grep` do not report clipped output.** When a listing or a search exceeds `max_output_size`, the deepagents helpers return the part that fit without any signal that results are missing, so the answer looks complete (observed with deepagents 0.7.19 and 0.7.32, reported as [deepagents#7480](https://github.com/langchain-ai/deepagents/issues/7480)). `glob` and `execute` do report truncation. Raise `max_output_size` for large directories or trees.
- **Network access is all or nothing.** The library can switch networking on or off but cannot restrict it to specific hosts. If the agent needs some access but not all of it, attach the container to a dedicated Docker network or use firewall rules.
- **Mounted directories are exposed.** A read-write volume gives the agent full access to its contents. Mount a dedicated directory, never a home directory, and never the Docker socket. Likewise, if your own code runs in a container, mounting the host socket gives that container control of the host's Docker daemon.
- **Commands run as the image's default user,** which is usually root inside the container. Capabilities are dropped, but the container is not rootless.
- **Disk usage is not limited.** The container's writable layer can grow until the host's disk is full.
- **Docker is not a hard security boundary.** Containers share the host kernel, so a kernel vulnerability bypasses all of the settings above. For untrusted code, add a stronger runtime such as gVisor or Kata, or run Docker in rootless mode.

## License

MIT, see LICENSE