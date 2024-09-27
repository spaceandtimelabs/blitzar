/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2024-present Space and Time Labs, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

/*
 * This is a workaround to define _VSTD before including cub/cub.cuh.
 * It should be removed when we can upgrade to a newer version of CUDA.
 *
 * We need to define _VSTD in order to use the clang version defined in
 * clang.nix and the CUDA toolkit version defined in cuda.nix.
 *
 * _VSTD was deprecated and removed from the LLVM truck.
 * NVIDIA: https://github.com/NVIDIA/cccl/pull/1331
 * LLVM: https://github.com/llvm/llvm-project/commit/683bc94e1637bd9bacc978f5dc3c79cfc8ff94b9
 *
 * We cannot currently use any CUDA toolkit above 12.4.1 because the Kubernetes
 * cluster currently cannot install a driver above 550.
 *
 * See CUDA toolkit and driver support:
 * https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
 */
#include <__config>

#define _VSTD std::_LIBCPP_ABI_NAMESPACE

#include "cub/cub.cuh"
