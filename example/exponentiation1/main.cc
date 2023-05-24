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

#include "example/exponentiation1/exponentiate_cpu.h"
#include "example/exponentiation1/exponentiate_gpu.h"
#include "sxt/curve21/type/element_p3.h"

using namespace sxt;

int main() {
  const int n = 10;
  c21t::element_p3 elements_gpu[n];
  exponentiate_gpu(elements_gpu, n);

  c21t::element_p3 elements_cpu[n];
  exponentiate_cpu(elements_cpu, n);

  for (int i = 0; i < n; ++i) {
    std::cout << elements_gpu[i] << " \t" << elements_cpu[i] << "\n";
  }
  return 0;
}
