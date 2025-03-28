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
#pragma once

#include "sxt/execution/async/shared_future.h"

namespace sxt::xendv {
//--------------------------------------------------------------------------------------------------
// chunk_context
//--------------------------------------------------------------------------------------------------
/**
 * Give context for an individual chunk of a chunked computation
 */
struct chunk_context {
  // a counter tracking the processing index for the given chunk
  unsigned chunk_index = 0;

  // the device used to process the chunk
  unsigned device_index = 0;

  // the total number of devices used to process the collection of chunks
  unsigned num_devices_used = 0;

  // When two chunks are scheduled for the same device, alt_future gives
  // a handle to the asynchronous computation associated with the other
  // chunk.
  //
  // alt_future can be used to overlap memory transfer with kernel computation. For
  // example, a functor to process chunks might look something like this
  //    f(const chunk_context& ctx, index_range rng) noexcept -> xena::future<> {
  //        ...
  //        async_copy_memory(stream, ...);
  //
  //        co_await ctx.alt_future;
  //            // wait for the other future to finish so that we don't oversubscribe the GPU
  //
  //         launch_kernel(stream, ...);
  //         co_await synchronize_stream(stream);
  //    }
  xena::shared_future<> alt_future;
};
} // namespace sxt::xendv
