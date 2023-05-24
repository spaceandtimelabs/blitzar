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
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

#include "sxt/base/container/span.h"
#include "sxt/base/profile/callgrind.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/multiexp/curve21/multiproduct_cpu_driver.h"
#include "sxt/multiexp/index/index_table.h"
#include "sxt/multiexp/pippenger_multiprod/multiproduct.h"
#include "sxt/multiexp/random/random_multiproduct_descriptor.h"
#include "sxt/multiexp/random/random_multiproduct_generation.h"
#include "sxt/multiexp/test/curve21_arithmetic.h"
#include "sxt/ristretto/random/element.h"

using namespace sxt;

int main(int argc, char* argv[]) {
  if (argc < 7) {
    std::cout << "Usage: benchmark <use_naive> <sequence_length> "
                 "<num_sequences> <max_num_inputs> <num_samples> <verbose> \n";
    return -1;
  }

  size_t seed = 0;
  std::mt19937 rng{seed};
  bool use_naive = std::atoi(argv[1]);
  auto sequence_length = static_cast<size_t>(std::atoi(argv[2]));
  auto num_sequences = static_cast<size_t>(std::atoi(argv[3]));
  auto max_num_inputs = static_cast<size_t>(std::atoi(argv[4]));
  auto num_samples = std::atoi(argv[5]);
  bool verbose = std::atoi(argv[6]);

  mtxrn::random_multiproduct_descriptor descriptor{
      .min_sequence_length = sequence_length,
      .max_sequence_length = sequence_length,
      .min_num_sequences = num_sequences,
      .max_num_sequences = num_sequences,
      .max_num_inputs = max_num_inputs,
  };

  mtxi::index_table products;
  size_t num_inputs, num_entries;
  mtxc21::multiproduct_cpu_driver drv;

  std::vector<double> elapsed_times;
  double mean_elapsed_time = 0;

  // we populate the products table with random values
  mtxrn::generate_random_multiproduct(products, num_inputs, num_entries, rng, descriptor);

  memmg::managed_array<c21t::element_p3> inout(num_entries);

  // we populate the inout array with `num_inputs` random curve21 elements
  rstrn::generate_random_elements(basct::span<c21t::element_p3>{inout.data(), num_inputs}, rng);

  size_t num_rows = products.num_rows();
  memmg::managed_array<c21t::element_p3> expected_result(products.num_rows());

  // we benchmark multiple times to reduce the randomness of each execution
  for (auto sample = 0; sample < num_samples; ++sample) {
    mtxi::index_table products_aux(products);

    auto begin_time = std::chrono::steady_clock::now();

    if (use_naive) {
      SXT_TOGGLE_COLLECT;
      mtxtst::sum_curve21_elements(expected_result, products_aux.cheader(), inout);
      SXT_TOGGLE_COLLECT;
    } else {
      SXT_TOGGLE_COLLECT;
      mtxpmp::compute_multiproduct(inout, products_aux, drv, num_inputs);
      SXT_TOGGLE_COLLECT;
    }

    auto end_time = std::chrono::steady_clock::now();

    auto elapsed_time =
        std::chrono::duration_cast<std::chrono::microseconds>(end_time - begin_time).count() / 1e6;

    elapsed_times.push_back(elapsed_time);
    mean_elapsed_time += elapsed_time / num_samples;
  }

  double std_deviation = 0;

  for (int i = 0; i < num_samples; ++i) {
    std_deviation += pow(elapsed_times[i] - mean_elapsed_time, 2.);
  }

  std_deviation = sqrt(std_deviation / num_samples);

  double data_throughput = num_entries / mean_elapsed_time;

  std::cout << "===== benchmark input" << std::endl;
  std::cout << "std::mt19937 rng seed: " << seed << std::endl;
  std::cout << "use naive multiprod : " << (use_naive ? "yes" : "no") << std::endl;
  std::cout << "num_samples (loop) : " << num_samples << std::endl;
  std::cout << "sequence_length : " << sequence_length << std::endl;
  std::cout << "num_sequences : " << num_sequences << std::endl;
  std::cout << "max_num_inputs : " << max_num_inputs << std::endl;

  std::cout << "===== benchmark results" << std::endl;
  std::cout << "num_inputs : " << num_inputs << std::endl;
  std::cout << "num_entries : " << num_entries << std::endl;

  std::cout << "compute elapsed time (s) : " << std::fixed << mean_elapsed_time << std::endl;
  std::cout << "compute std time deviation (s) : " << std::fixed << std_deviation << std::endl;

  std::cout << "throughput (num_entries / s) : " << std::scientific << data_throughput << std::endl;

  if (verbose) {
    auto& results = (use_naive ? expected_result : inout);

    for (size_t i = 0; i < num_rows; ++i) {
      std::cout << results.data()[i] << std::endl;
    }
  }

  return 0;
}
