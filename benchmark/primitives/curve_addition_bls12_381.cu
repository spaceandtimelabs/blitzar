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
#include <chrono>
#include <print>

#include "sxt/base/num/divide_up.h"
#include "sxt/base/num/fast_random_number_generator.h"
#include "sxt/curve_g1/operation/add.h"
#include "sxt/curve_g1/random/element_p2.h"
#include "sxt/curve_g1/type/element_p2.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/device_resource.h"

namespace sxt {
//--------------------------------------------------------------------------------------------------
// vector_add_impl 
//--------------------------------------------------------------------------------------------------
__global__ void vector_add_impl(cg1t::element_p2* __restrict__ vec_ret,
                                const cg1t::element_p2* __restrict__ vec_a,
                                unsigned n_elements,
                                unsigned repetitions) {
  auto tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < n_elements) {
    cg1t::element_p2 x = vec_a[tid];
    auto res = x;
    for (unsigned i = 0; i < repetitions; ++i) {
      cg1o::add(res, res, x);
    }
    vec_ret[tid] = res;
  }
}

//--------------------------------------------------------------------------------------------------
// vector_add
//--------------------------------------------------------------------------------------------------
void vector_add(cg1t::element_p2* vec_result, const cg1t::element_p2* vec_a, unsigned n_elements, unsigned repetitions, unsigned n_threads) {
    const unsigned threads_per_block = n_threads;
    const unsigned num_blocks = basn::divide_up(n_elements, threads_per_block);

    vector_add_impl<<<num_blocks, threads_per_block>>>(vec_result, vec_a, n_elements, repetitions);
}


//--------------------------------------------------------------------------------------------------
// init_random_array_impl 
//--------------------------------------------------------------------------------------------------
__global__ void init_random_array_impl(cg1t::element_p2* __restrict__ rand, unsigned n_elements) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n_elements)
    {
        basn::fast_random_number_generator rng{static_cast<uint64_t>(tid + 1),
                                               static_cast<uint64_t>(n_elements + 1)};
        cg1rn::generate_random_element(rand[tid], rng);
    }
}

//--------------------------------------------------------------------------------------------------
// init_random_array
//--------------------------------------------------------------------------------------------------
void init_random_array(cg1t::element_p2* rand, unsigned n_elements, unsigned n_threads) {
    const unsigned threads_per_block = n_threads;
    const unsigned num_blocks = basn::divide_up(n_elements, threads_per_block);

    init_random_array_impl<<<num_blocks, threads_per_block>>>(rand, n_elements);
}

//--------------------------------------------------------------------------------------------------
// add_bls12_381_g1_curve_elements
//--------------------------------------------------------------------------------------------------
void add_bls12_381_g1_curve_elements(unsigned n_elements, unsigned repetitions, unsigned n_threads) noexcept {
  std::print("add_bls12_381_g1_curve_elements\n");
  
  // Allocate memory for the input and output vectors
  memmg::managed_array<cg1t::element_p2> a(n_elements, memr::get_device_resource());
  memmg::managed_array<cg1t::element_p2> ret(n_elements, memr::get_device_resource());

  // Populate the input vectors with random curve elements
  init_random_array(a.data(), n_elements, n_threads);

  // Warm-up loop
  vector_add(ret.data(), a.data(), n_elements, repetitions, n_threads);
  cudaDeviceSynchronize();

  // Report any errors from the warmup loop
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::print("CUDA error: {}\n", cudaGetErrorString(err));
  }

  // Benchmarking loop
  auto start_time = std::chrono::steady_clock::now();
  vector_add(ret.data(), a.data(), n_elements, repetitions, n_threads);
  cudaDeviceSynchronize();
  auto end_time = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

  // Report any errors from the benchmark loop
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::print("CUDA error: {}\n", cudaGetErrorString(err));
  }

  // Report data
  std::print("Elapsed time: {} milliseconds\n", duration.count());
  auto GMPS = 1.0e-9 * repetitions * n_elements / (1.0e-3 * duration.count());
  std::print("Performance: {} Giga curve additions Per Second\n", GMPS);
}
}
