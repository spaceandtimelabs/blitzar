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
#include "benchmark/primitives/field_ops_bls12_381.h"

using namespace sxt;

int main(int argc, char* argv[]) {
  if (argc < 5) {
    std::print("Usage: benchmark <curve> <op> <n_elements> <repetitions> <optional - n_threads> <optional - n_executions>\n");
    return -1;
  }

  const std::string curve = argv[1];
  const std::string op = argv[2];
  auto n_elements = std::atoi(argv[3]);
  auto repetitions = std::atoi(argv[4]);
  auto n_threads = (argc > 6) ? std::atoi(argv[5]) : 256;
  auto n_executions = (argc > 7) ? std::atoi(argv[6]) : 10;

  std::println("===== benchmark results =====");
  std::println("backend : GPU");
  std::println("curve : {}", curve);
  std::println("operation : {}", op);
  std::println("Number of elements : {}", n_elements);
  std::println("Repetitions : {}", repetitions);
  std::println("Max threads per block : {}", n_threads);
  std::println("Number of executions : {}", n_executions);
  std::println("*****************************");

  if (op == "curve") {
    if (curve == "bls12_381") {
      add_bls12_381_g1_curve_elements(n_elements, repetitions,  n_threads, n_executions);
    }
  } else if (op == "field") {
    if (curve == "bls12_381") {
      add_bls12_381_field_elements(n_elements, repetitions, n_threads, n_executions);
      mul_bls12_381_field_elements(n_elements, repetitions, n_threads, n_executions);
    }
  }

  std::println("******************************");
  std::println("===== benchmark complete =====");
  std::println("******************************");

  return 0;
}
