from .PyDEME_core import *
import pathlib
import os
import importlib.metadata
import sys
import platform

# Inspired by https://github.com/cupy/cupy/blob/3be80296b5f8c7d7f60dbee84d3b16e8be39b35f/cupy/_environment.py#L513
def set_cudatoolkit_includedir():
    if "CONDA_PREFIX" in os.environ:
        if sys.platform.startswith('linux'):
            arch = platform.machine()
            if arch == "aarch64":
                arch = "sbsa"
            assert arch, "platform.machine() returned an empty string"
            target_dir = f"{arch}-linux"
            include_dir = os.path.join(sys.prefix, "targets", target_dir, "include")
            RuntimeDataHelper.SetCUDAToolkitHeaders(include_dir)
        elif sys.platform.startswith('win'):
            include_dir = os.path.join(sys.prefix, "Library", "include")
            target_include_dir = os.path.join(include_dir, "targets", "x64")
            RuntimeDataHelper.SetCUDAToolkitHeaders(include_dir)
            RuntimeDataHelper.SetCUDAToolkitTargetHeaders(target_include_dir)
        else:
            # No idea what this platform is. Do nothing?
            raise RuntimeError("Unsupported platform")
    else:
        # TODO: pip wheel version of CUDA
        RuntimeDataHelper.SetCUDAToolkitHeaders(os.path.join(os.environ["CUDA_HOME"], "include"))


RuntimeDataHelper.SetPathPrefix(str(pathlib.Path(__file__).parent.resolve()))
set_cudatoolkit_includedir()