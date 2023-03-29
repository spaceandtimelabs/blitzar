#include <chrono>
#include <iostream>
#include <memory_resource>
#include <vector>

#include "params.h"
#include "sxt/base/container/span.h"
#include "sxt/base/num/ceil_log2.h"
#include "sxt/base/num/fast_random_number_generator.h"
#include "sxt/base/profile/callgrind.h"
#include "sxt/cbindings/backend/computational_backend.h"
#include "sxt/proof/inner_product/proof_descriptor.h"
#include "sxt/proof/inner_product/random_product_generation.h"
#include "sxt/proof/transcript/transcript.h"
#include "sxt/ristretto/type/compressed_element.h"
#include "sxt/scalar25/type/element.h"

using namespace sxt;
using namespace sxt::bncip;

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
  prft::transcript transcript{"abc"};

  std::vector<double> durations(params.iterations);

  for (size_t i = 0; i < params.iterations; ++i) {
    auto t1 = std::chrono::steady_clock::now();
    params.backend->prove_inner_product(l_vector, r_vector, ap_value, transcript, descriptor,
                                        a_vector);
    auto t2 = std::chrono::steady_clock::now();
    durations[i] = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1.0e6;
  }

  double sum = 0;
  for (auto duration : durations) {
    sum += duration;
  }
  auto mean = sum / durations.size();
  std::cout << "mean elapse (s): " << mean << "\n";
  std::cout << "mean throughput (1/s): " << (params.n / mean) << "\n";
  return 0;
}
