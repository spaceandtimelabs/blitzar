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
#include <print>

#include "benchmark/primitives/curve_addition_bls12_381.h"
#include "benchmark/primitives/field_addition_bls12_381.h"

using namespace sxt;

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::print("Usage: benchmark <vector_size> <repetitions>\n");
    return -1;
  }

  auto vector_size = std::atoi(argv[1]);
  auto repetitions = std::atoi(argv[2]);

  std::print("===== benchmark results =====\n");
  std::print("backend : GPU\n");
  std::print("vector size : {}\n", vector_size);
  std::print("repetitions : {}\n", repetitions);
  std::print("*****************************\n");

  add_bls12_381_g1_curve_elements(vector_size, repetitions);
  add_bls12_381_g1_field_elements(vector_size, repetitions);

  std::print("******************************\n");
  std::print("===== benchmark complete =====\n");
  std::print("******************************\n");

  return 0;
}
