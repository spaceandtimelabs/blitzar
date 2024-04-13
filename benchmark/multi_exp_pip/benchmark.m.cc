/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2024-present Space and Time Labs, Inc.
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
#include <charconv>
#include <print>
#include <string_view>
#include <memory>

#include "sxt/seqcommit/generator/base_element.h"
#include "sxt/curve21/operation/add.h"
#include "sxt/curve21/operation/double.h"
#include "sxt/curve21/operation/neg.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/multiexp/pippenger2/in_memory_partition_table_accessor_utility.h"

using namespace sxt;

static std::unique_ptr<mtxpp2::partition_table_accessor<c21t::element_p3>>
make_partition_table_accessor(unsigned n) noexcept {
  std::vector<c21t::element_p3> generators(n);
  for (unsigned i=0; i<n; ++i) {
    sqcgn::compute_base_element(generators[i], i); 
  }
  return mtxpp2::make_in_memory_partition_table_accessor<c21t::element_p3>(generators);
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::println("Usage: benchmark <num_outputs> <n>");
    return -1;
  }
  std::string_view num_outputs_str{argv[1]};
  std::string_view n_str{argv[2]};
  unsigned num_outputs, n;
  if (std::from_chars(num_outputs_str.begin(), num_outputs_str.end(), num_outputs).ec !=
      std::errc{}) {
    std::println("invalid argument: {}\n", num_outputs_str);
    return -1;
  }
  if (std::from_chars(n_str.begin(), n_str.end(), n).ec != std::errc{}) {
    std::println("invalid argument: {}\n", n_str);
    return -1;
  }
  std::println("n = {}", n);
  auto accessor = make_partition_table_accessor(n);
  std::println("accessor created");
  (void)accessor;
  (void)argc;
  (void)argv;
  return 0;
}
