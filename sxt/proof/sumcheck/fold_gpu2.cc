/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2025-present Space and Time Labs, Inc.
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
#include "sxt/proof/sumcheck/fold_gpu2.h"

#include "sxt/algorithm/iteration/kernel_fit.h"
#include "sxt/base/device/property.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/error/assert.h"
#include "sxt/base/iterator/split.h"
#include "sxt/base/num/ceil_log2.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/device/for_each.h"
#include "sxt/execution/device/synchronization.h"
#include "sxt/execution/kernel/kernel_dims.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/async_device_resource.h"
#include "sxt/memory/resource/device_resource.h"
#include "sxt/proof/sumcheck/mle_utility.h"
#include "sxt/scalar25/operation/mul.h"
#include "sxt/scalar25/operation/muladd.h"
#include "sxt/scalar25/operation/sub.h"
#include "sxt/scalar25/type/element.h"
#include "sxt/scalar25/type/literal.h"

namespace sxt::prfsk {
//--------------------------------------------------------------------------------------------------
// fold_gpu
//--------------------------------------------------------------------------------------------------
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-parameter"
xena::future<> fold_gpu(basct::span<s25t::element> mles, unsigned stride, unsigned n,
                        const s25t::element& r) noexcept {
  return {};
}
#pragma clang diagnostic pop
} // namespace sxt::prfsk
