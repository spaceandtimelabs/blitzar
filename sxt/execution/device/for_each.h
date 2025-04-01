/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2023-present Space and Time Labs, Inc.
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

#include <functional>
#include <optional>

#include "sxt/base/device/stream.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/async/shared_future.h"
#include "sxt/execution/device/chunk_context.h"

namespace sxt::basit {
class index_range;
class index_range_iterator;
} // namespace sxt::basit

namespace sxt::xendv {
//--------------------------------------------------------------------------------------------------
// concurrent_for_each
//--------------------------------------------------------------------------------------------------
/**
 * Invoke the function f on the range of chunks provided, splitting the work across available
 * devices.
 */
xena::future<>
concurrent_for_each(basit::index_range_iterator first, basit::index_range_iterator last,
                    std::function<xena::future<>(const basit::index_range&)> f) noexcept;

/**
 * Invoke the function f on chunks of the provided index range and try to split the
 * work across the number of available devices.
 */
xena::future<>
concurrent_for_each(basit::index_range rng,
                    std::function<xena::future<>(const basit::index_range&)> f) noexcept;

//--------------------------------------------------------------------------------------------------
// for_each_device
//--------------------------------------------------------------------------------------------------
/**
 * Invoke the function f on the range of chunks provided, splitting the work across available
 * devices.
 */
xena::future<> for_each_device(
    basit::index_range_iterator first, basit::index_range_iterator last,
    std::function<xena::future<>(const chunk_context& ctx, basit::index_range)> f) noexcept;
} // namespace sxt::xendv
