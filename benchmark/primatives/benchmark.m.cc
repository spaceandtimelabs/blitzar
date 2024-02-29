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
#include <iostream>

#include "benchmark/primatives/curve_addition_bls12_381.h"

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: benchmark <vector_size> <repetitions>\n";
    return -1;
  }

  auto vector_size = std::atoi(argv[2]);
  auto repetitions = std::atoi(argv[3]);

  sxt::add(vector_size, repetitions);

  return 0;
}
