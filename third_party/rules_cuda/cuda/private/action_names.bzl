CUDA_COMPILE = "cuda-compile"  # cuda compile comprise of host and device compilation

CUDA_DEVICE_LINK = "cuda-dlink"

ACTION_NAMES = struct(
    cuda_compile = CUDA_COMPILE,
    device_link = CUDA_DEVICE_LINK,
)
