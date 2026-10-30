"""Container sandbox backends for LangChain DeepAgents."""

from importlib.metadata import PackageNotFoundError, version

from .errors import DockerImageNotFound, MissingPythonError, MissingTimeoutError, SandboxError
from .managed import ManagedContainerSandbox
from .provisioning import ContainerProvisioner, ContainerSpec
from .sandbox import ContainerSandbox

try:
    __version__ = version("agent-container-sandbox")
except PackageNotFoundError:  # running from a source checkout without install metadata
    __version__ = "0+unknown"

__all__ = [
    "ContainerSandbox",
    "ContainerProvisioner",
    "ContainerSpec",
    "SandboxError",
    "DockerImageNotFound",
    "MissingTimeoutError",
    "MissingPythonError",
    "ManagedContainerSandbox",
    "__version__",
]
