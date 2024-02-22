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
#include "sxt/field51/type/element.h"

#include <iostream>

using namespace sxt::f51t;

int main() {
  element e1{0, 0, 0, 0, 0};
  std::cout << "e1 = " << e1 << "\n";

  element e2{1, 0, 0, 0, 0};
  std::cout << "e2 = " << e2 << "\n";

  element e3{10, 0, 0, 0, 0};
  std::cout << "e3 = " << e3 << "\n";

  element e4{16, 0, 0, 0, 0};
  std::cout << "e4 = " << e4 << "\n";

  element e5{0x100, 0, 0, 0, 0};
  std::cout << "e5 = " << e5 << "\n";

  return 0;

}
