# SPDX-License-Identifier: Apache-2.0
from setuptools import setup, find_packages
import os

ext_modules = []
cmdclass = {}

try:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    # Discover CUDA_HOME
    cuda_home = os.environ.get("CUDA_HOME") or getattr(torch.utils.cpp_extension, "CUDA_HOME", None)
    if cuda_home is None or not os.path.exists(cuda_home):
        for candidate in [
            "/usr/local/cuda",
            "/usr/local/cuda-12",
            "/usr/local/cuda-12.8",
            "/usr/local/cuda-12.4",
            "/usr/local/cuda-11",
        ]:
            if os.path.exists(candidate):
                cuda_home = candidate
                os.environ["CUDA_HOME"] = candidate
                break

    # Check for nvcc availability
    nvcc_available = False
    if cuda_home and os.path.exists(os.path.join(cuda_home, "bin", "nvcc")):
        nvcc_available = True
    elif os.system("nvcc --version > /dev/null 2>&1") == 0:
        nvcc_available = True

    if (torch.cuda.is_available() or "CUDA_HOME" in os.environ) and nvcc_available:
        ext_modules.append(
            CUDAExtension(
                name="cautious_decoding_cuda",
                sources=[
                    "csrc/bindings.cpp",
                    "csrc/tree_scoring.cu",
                    "csrc/tree_attention.cu",
                ],
                extra_compile_args={
                    "cxx": ["-O3"],
                    "nvcc": ["-O3", "--use_fast_math"],
                },
            )
        )
        cmdclass["build_ext"] = BuildExtension
except Exception:
    # If torch is not available during build/metadata extraction, or nvcc is missing,
    # proceed with install without crashing.
    pass

setup(
    name="vllm-gpu-cautious-decoding",
    version="0.2.0",
    description="GPU-Level Cautious Tree Search Decoding plugin for vLLM",
    author="Carlo Cetrone",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    python_requires=">=3.8",
    install_requires=[
        "vllm>=0.6.0",
        "torch>=2.0.0",
    ],
    entry_points={
        "vllm.general_plugins": [
            "cautious_gpu = cautious_gpu.plugin:register",
        ],
    },
)
