import os
import sys
from pathlib import Path


def configure_windows_cuda_dll_paths() -> None:
    if os.name != "nt":
        return

    site_packages = Path(sys.prefix) / "Lib" / "site-packages"
    dll_dirs = [
        site_packages / "torch" / "lib",
        site_packages / "ctranslate2",
    ]
    existing_dirs = [path for path in dll_dirs if path.exists()]

    for dll_dir in existing_dirs:
        try:
            os.add_dll_directory(str(dll_dir))
        except (AttributeError, OSError):
            pass

    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    filtered_paths = [
        path
        for path in path_parts
        if "NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\bin".lower() not in path.lower()
    ]
    os.environ["PATH"] = os.pathsep.join([str(path) for path in existing_dirs] + filtered_paths)
