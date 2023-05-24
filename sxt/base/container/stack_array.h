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

#include <alloca.h>

#include <type_traits>

#include "sxt/base/container/span.h"

/**
 * Construct a dynamically-sized array on the stack.
 *
 * Warning: only use when n is not large.
 */
#define SXT_STACK_ARRAY(name, n, ...)                                                              \
  static_assert(std::is_trivially_destructible_v<__VA_ARGS__>);                                    \
  sxt::basct::span<__VA_ARGS__> name {                                                             \
    static_cast<__VA_ARGS__*>(::alloca((n) * sizeof(__VA_ARGS__))), n                              \
  }
