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
#include <cerrno>
#include <charconv>
#include <cstring>
#include <fstream>
#include <print>
#include <string_view>
#include <vector>

#include "sxt/base/error/panic.h"
#include "sxt/base/num/divide_up.h"
#include "sxt/base/num/fast_random_number_generator.h"
#include "sxt/curve21/operation/add.h"
#include "sxt/curve21/operation/double.h"
#include "sxt/curve21/operation/neg.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/multiexp/pippenger2/partition_table.h"
#include "sxt/ristretto/random/element.h"

using namespace sxt;

//--------------------------------------------------------------------------------------------------
// make_partition_table
//--------------------------------------------------------------------------------------------------
static void make_partition_table(std::string_view filename, unsigned n) noexcept {
  n = basn::divide_up(n, 16u) * 16u;
  std::print("creating table {} {}\n", filename, n);

  // make generators
  basn::fast_random_number_generator rng{1u, 2u};
  std::vector<c21t::element_p3> generators(n);
  rstrn::generate_random_elements(generators, rng);

  // compute precomputed partition values
  auto num_elements = 1u << 16u;
  std::vector<c21t::element_p3> sums(num_elements * n / 16u);
  mtxpp2::compute_partition_table<c21t::element_p3>(sums, generators);

  // write table
  std::ofstream out{filename};
  if (!out.good()) {
    baser::panic("failed to open {}: {}", filename, std::strerror(errno));
  }
  out.write(reinterpret_cast<const char*>(sums.data()), sums.size() * sizeof(sums[0]));
}

//--------------------------------------------------------------------------------------------------
// main
//--------------------------------------------------------------------------------------------------
int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::print(stderr, "Usage: blitzar <cmd> <arg1> ... <argN>\n");
    return -1;
  }
  std::string_view cmd{argv[1]};
  if (cmd == "make-partition-table") {
    // create a table of precomputed values for Pippenger's partition algorithm
    if (argc != 4) {
      std::print(stderr, "Usage: blitzar make-partition-table <filename> <n>\n");
      return -1;
    }
    std::string_view filename{argv[2]};
    std::string_view n_str{argv[3]};
    unsigned n;
    if (auto err = std::from_chars(n_str.begin(), n_str.end(), n); err.ec != std::errc{}) {
      std::print(stderr, "invalid number: {}\n", n_str);
      return -1;
    }
    make_partition_table(filename, n);
  }
  return 0;
}
