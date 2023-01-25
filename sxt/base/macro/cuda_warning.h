#pragma once

#ifdef __CUDACC__
#define CUDA_DISABLE_HOSTDEV_WARNING _Pragma("nv_exec_check_disable")
#else
#define CUDA_DISABLE_HOSTDEV_WARNING
#endif
