/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2023 Space and Time Labs, Inc.
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
#include <chrono>
#include <iostream>
#include <memory_resource>
#include <string_view>
#include <vector>

#include "params.h"
#include "sxt/base/container/span.h"
#include "sxt/base/num/ceil_log2.h"
#include "sxt/base/num/fast_random_number_generator.h"
#include "sxt/base/profile/callgrind.h"
#include "sxt/cbindings/backend/computational_backend.h"
#include "sxt/curve21/operation/overload.h"
#include "sxt/proof/inner_product/proof_descriptor.h"
#include "sxt/proof/inner_product/random_product_generation.h"
#include "sxt/proof/transcript/transcript.h"
#include "sxt/ristretto/type/compressed_element.h"
#include "sxt/scalar25/operation/overload.h"
#include "sxt/scalar25/type/element.h"

using namespace sxt;
using namespace sxt::bncip;

//--------------------------------------------------------------------------------------------------
// print_result
//--------------------------------------------------------------------------------------------------
static void print_result(std::string_view name, const std::vector<double>& durations,
                         size_t n) noexcept {
  double sum = 0;
  for (auto duration : durations) {
    sum += duration;
  }
  auto mean = sum / durations.size();
  std::cout << name << " mean elapse (s): " << mean << "\n";
  std::cout << name << " mean throughput (1/s): " << (n / mean) << "\n";
}

//--------------------------------------------------------------------------------------------------
// main
//--------------------------------------------------------------------------------------------------
int main(int argc, char* argv[]) {
  params params;
  read_params(params, argc, argv);

  std::cout << "===== benchmark results" << std::endl;
  std::cout << "backend : " << params.backend_name << std::endl;
  std::cout << "n length : " << params.n << std::endl;
  std::cout << "iterations : " << params.iterations << std::endl;
  std::cout << "********************************************" << std::endl;

  std::pmr::monotonic_buffer_resource alloc;
  basn::fast_random_number_generator rng{1, 2};
  prfip::proof_descriptor descriptor;
  basct::cspan<s25t::element> a_vector;
  generate_random_product(descriptor, a_vector, rng, &alloc, params.n);

  auto num_rounds = static_cast<size_t>(basn::ceil_log2(params.n));
  std::vector<rstt::compressed_element> l_vector(num_rounds), r_vector(num_rounds);
  s25t::element ap_value;
  s25t::element product = a_vector[0] * descriptor.b_vector[0];
  c21t::element_p3 a_commit = a_vector[0] * descriptor.g_vector[0];
  for (size_t i = 1; i < params.n; ++i) {
    product = product + a_vector[i] * descriptor.b_vector[i];
    a_commit = a_commit + a_vector[i] * descriptor.g_vector[i];
  }

  std::vector<double> prove_durations(params.iterations), verify_durations(params.iterations);

  for (size_t i = 0; i < params.iterations; ++i) {
    // prove
    {
      prft::transcript transcript{"abc"};
      auto t1 = std::chrono::steady_clock::now();
      params.backend->prove_inner_product(l_vector, r_vector, ap_value, transcript, descriptor,
                                          a_vector);
      auto t2 = std::chrono::steady_clock::now();
      prove_durations[i] =
          std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1.0e6;
    }

    // verify
    {
      prft::transcript transcript{"abc"};
      auto t1 = std::chrono::steady_clock::now();
      auto res = params.backend->verify_inner_product(transcript, descriptor, product, a_commit,
                                                      l_vector, r_vector, ap_value);
      if (!res) {
        std::cerr << "verification failed\n";
        return -1;
      }
      auto t2 = std::chrono::steady_clock::now();
      verify_durations[i] =
          std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1.0e6;
    }
  }

  print_result("prove", prove_durations, params.n);
  print_result("verify", verify_durations, params.n);

  return 0;
}
