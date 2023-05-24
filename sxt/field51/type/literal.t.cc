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
#include "sxt/field51/type/literal.h"

#include <iostream>

using namespace sxt::f51t;

int main() {
  std::cout << "e1 = " << 0x0_f51 << std::endl;
  std::cout << "e2 = " << 0x1_f51 << std::endl;
  std::cout << "e3 = " << 0xa_f51 << std::endl;
  std::cout << "e4 = " << 0x10_f51 << std::endl;
  std::cout << "e6 = " << 0x3b86191f4f2865cc462f08daa6d911c0df283b53cb3b8f7d6027666f4c94e38_f51
            << std::endl;
  return 0;
}
