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

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <memory_resource>
#include <type_traits>
#include <utility>
#include <vector>

#include "sxt/base/error/assert.h"

namespace sxt::bastst {
//--------------------------------------------------------------------------------------------------
// check_equality
//--------------------------------------------------------------------------------------------------
namespace detail {
template <class T>
auto check_equality(const T& lhs, const T& rhs) noexcept -> decltype(lhs == rhs) {
  SXT_RELEASE_ASSERT(lhs == rhs, "lhs and rhs should be equal");

  return true;
}

template <class... Tx> void check_equality(const Tx&...) noexcept {
  // there's no equality operator so do nothing
}
} // namespace detail

//--------------------------------------------------------------------------------------------------
// exercise_allocator_aware_operations
//--------------------------------------------------------------------------------------------------
/**
 * walk through standard allocator aware operations and check correctness
 */
template <class T> void exercise_allocator_aware_operations(const T& obj) noexcept {
  std::pmr::monotonic_buffer_resource r1;

  // default construction
  {
    T t1;

    T t2{&r1};

    detail::check_equality(t1, t2);
  }

  // copy-construction
  {
    T t1{obj};
    detail::check_equality(t1, obj);

    T t2{obj, &r1};
    detail::check_equality(t2, obj);
  }

  // move-construction
  {
    T t1{obj};

    T t2{std::move(t1)};
    detail::check_equality(t2, obj);

    T t3{std::move(t2), &r1};
    detail::check_equality(t3, obj);
  }

  // copy-assignment
  {
    T t1;
    t1 = obj;
    detail::check_equality(t1, obj);

    T t2{t1};
    t2 = obj;
    detail::check_equality(t2, obj);

    T t3{t1, &r1};
    t3 = obj;
    detail::check_equality(t3, obj);
  }

  // move-assignment
  {
    T t1{obj};

    T t2;
    t2 = std::move(t1);
    detail::check_equality(t2, obj);

    t1 = obj;
    T t3{obj};
    t3 = std::move(t1);
    detail::check_equality(t3, obj);

    t1 = obj;
    T t4{obj, &r1};
    t4 = std::move(t1);
    detail::check_equality(t4, obj);
  }

  // composition
  {
    std::pmr::vector<T> v{&r1};
    v.emplace_back(obj);
    detail::check_equality(v[0], obj);
  }
}
} // namespace sxt::bastst
