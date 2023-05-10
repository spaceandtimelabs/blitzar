/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2023 Space and Time Labs, Inc.
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

#include "sxt/base/type/raw_stream.h"

namespace sxt::basdv {
struct stream_handle;

//--------------------------------------------------------------------------------------------------
// stream
//--------------------------------------------------------------------------------------------------
/**
 * Wrapper around a pooled CUDA stream.
 */
class stream {
public:
  stream() noexcept;
  stream(stream&& other) noexcept;

  ~stream() noexcept;

  stream(const stream&) = delete;
  stream& operator=(const stream&) = delete;
  stream& operator=(stream&& other) noexcept;

  stream_handle* release_handle() noexcept;

  bast::raw_stream_t raw_stream() const noexcept;

  operator bast::raw_stream_t() const noexcept { return this->raw_stream(); }

private:
  stream_handle* handle_;
};
} // namespace sxt::basdv
