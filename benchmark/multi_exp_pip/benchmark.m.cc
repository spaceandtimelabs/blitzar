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
#include <chrono>
#include <iostream>
#include <limits>
#include <memory>
#include <print>
#include <random>
#include <string_view>

#include "sxt/curve21/operation/add.h"
#include "sxt/curve21/operation/double.h"
#include "sxt/curve21/operation/neg.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/execution/schedule/scheduler.h"
#include "sxt/memory/resource/pinned_resource.h"
#include "sxt/multiexp/pippenger2/in_memory_partition_table_accessor_utility.h"
#include "sxt/multiexp/pippenger2/multiexponentiation.h"
#include "sxt/ristretto/operation/compression.h"
#include "sxt/ristretto/type/compressed_element.h"
#include "sxt/seqcommit/generator/base_element.h"

using namespace sxt;

//--------------------------------------------------------------------------------------------------
// make_partition_table_accessor
//--------------------------------------------------------------------------------------------------
static std::unique_ptr<mtxpp2::partition_table_accessor<c21t::element_p3>>
make_partition_table_accessor(unsigned n) noexcept {
  std::vector<c21t::element_p3> generators(n);
  for (unsigned i = 0; i < n; ++i) {
    sqcgn::compute_base_element(generators[i], i);
  }
  return mtxpp2::make_in_memory_partition_table_accessor<c21t::element_p3>(generators);
}

//--------------------------------------------------------------------------------------------------
// fill_exponents
//--------------------------------------------------------------------------------------------------
static void fill_exponents(memmg::managed_array<uint8_t>& exponents, unsigned num_outputs,
                           unsigned n) noexcept {
  unsigned element_num_bytes = 32;
  exponents.resize(num_outputs * n * element_num_bytes);
  std::mt19937 rng{0};
  std::uniform_int_distribution<uint8_t> dist{0, std::numeric_limits<uint8_t>::max()};
  for (unsigned output_index = 0; output_index < num_outputs; ++output_index) {
    for (unsigned i = 0; i < n; ++i) {
      for (unsigned byte_index = 0; byte_index < element_num_bytes; ++byte_index) {
        exponents[byte_index + element_num_bytes * output_index +
                  element_num_bytes * num_outputs * i] = dist(rng);
      }
    }
  }
}

//--------------------------------------------------------------------------------------------------
// print_elements
//--------------------------------------------------------------------------------------------------
static void print_elements(basct::cspan<c21t::element_p3> elements) noexcept {
  rstt::compressed_element r;
  size_t index = 0;
  for (auto& e : elements) {
    rsto::compress(r, e);
    std::cout << index++ << ": " << r << "\n";
  }
}

//--------------------------------------------------------------------------------------------------
// main
//--------------------------------------------------------------------------------------------------
int main(int argc, char* argv[]) {
  if (argc != 4) {
    // Usage: benchmark <cpu|gpu> <n> <num_samples> <num_commitments> <element_nbytes> <verbose>
    std::println("Usage: benchmark <n> <num_samples> <num_outputs>");
    return -1;
  }

  // read arguments
  std::string_view n_str{argv[1]};
  std::string_view num_samples_str{argv[2]};
  std::string_view num_outputs_str{argv[3]};
  unsigned n, num_samples, num_outputs;
  if (std::from_chars(n_str.begin(), n_str.end(), n).ec != std::errc{}) {
    std::println("invalid argument: {}\n", n_str);
    return -1;
  }
  if (std::from_chars(num_samples_str.begin(), num_samples_str.end(), num_samples).ec !=
      std::errc{}) {
    std::println("invalid argument: {}\n", num_samples_str);
    return -1;
  }
  if (std::from_chars(num_outputs_str.begin(), num_outputs_str.end(), num_outputs).ec !=
      std::errc{}) {
    std::println("invalid argument: {}\n", num_outputs_str);
    return -1;
  }

  // set up data
  std::println("num_outputs = {}", num_outputs);
  std::println("n = {}", n);
  auto accessor = make_partition_table_accessor(n);
  std::println("accessor created");

  memmg::managed_array<uint8_t> exponents;
  fill_exponents(exponents, num_outputs, n);

  memmg::managed_array<c21t::element_p3> res{num_outputs, memr::get_pinned_resource()};
  auto fut = mtxpp2::multiexponentiate<c21t::element_p3>(res, *accessor, 32, exponents);
  xens::get_scheduler().run();

  // run benchmark
  double times = 0;
  for (unsigned i = 0; i < num_samples; ++i) {
    auto t1 = std::chrono::steady_clock::now();
    auto fut = mtxpp2::multiexponentiate<c21t::element_p3>(res, *accessor, 32, exponents);
    xens::get_scheduler().run();
    auto t2 = std::chrono::steady_clock::now();
    auto elapse = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
    times += elapse.count() / 1e3;
    // std::cout << "elapse: " << elapse.count() / 1e3 << "\n";
  }
  std::cout << "compute duration (s): " << times / num_samples << "\n";

  // print_elements(res);

  return 0;
}
