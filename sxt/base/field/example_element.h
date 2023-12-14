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

namespace sxt::basfld {
//--------------------------------------------------------------------------------------------------
// element1
//--------------------------------------------------------------------------------------------------
/**
 * Provides a minimal implementation of the field element concent that can be used for writing
 * tests.
 *
 * element1 is a one limb field element that uses modulus 97.
 */
struct element1 {
  static constexpr size_t num_limbs_v = 1;

  element1() noexcept = default;

  constexpr element1(uint64_t x1) noexcept : data_{x1} {}

  constexpr const uint64_t& operator[](int index) const noexcept { return data_[index]; }

  constexpr uint64_t& operator[](int index) noexcept { return data_[index]; }

  constexpr const uint64_t* data() const noexcept { return data_; }

  constexpr uint64_t* data() noexcept { return data_; }

  static constexpr element1 modulus() noexcept { return element1{97}; }

  bool operator==(const element1&) const noexcept = default;

private:
  uint64_t data_[num_limbs_v];
};

//--------------------------------------------------------------------------------------------------
// element6
//--------------------------------------------------------------------------------------------------
/**
 * Provides a multi-limbed implementation of the field element concept that can be used for writing
 * tests.
 *
 * element6 is a six limbed field element that uses the bls12-381 modulus.
 */
struct element6 {
  static constexpr size_t num_limbs_v = 6;

  element6() noexcept = default;

  constexpr element6(uint64_t x1, uint64_t x2, uint64_t x3, uint64_t x4, uint64_t x5,
                     uint64_t x6) noexcept
      : data_{x1, x2, x3, x4, x5, x6} {}

  constexpr const uint64_t& operator[](int index) const noexcept { return data_[index]; }

  constexpr uint64_t& operator[](int index) noexcept { return data_[index]; }

  constexpr const uint64_t* data() const noexcept { return data_; }

  constexpr uint64_t* data() noexcept { return data_; }

  static constexpr element6 modulus() noexcept {
    return element6{0xb9feffffffffaaab, 0x1eabfffeb153ffff, 0x6730d2a0f6b0f624,
                    0x64774b84f38512bf, 0x4b1ba7b6434bacd7, 0x1a0111ea397fe69a};
  }

  bool operator==(const element6&) const noexcept = default;

private:
  uint64_t data_[num_limbs_v];
};
} // namespace sxt::basfld
