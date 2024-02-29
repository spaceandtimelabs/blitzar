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
#include <chrono>

#include "sxt/base/num/fast_random_number_generator.h"
#include "sxt/curve_g1/operation/add.h"
#include "sxt/curve_g1/random/element_p2.h"
#include "sxt/curve_g1/type/element_p2.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/device_resource.h"

#define MAX_THREADS_PER_BLOCK 256

namespace sxt {
//--------------------------------------------------------------------------------------------------
// vector_add_impl 
//--------------------------------------------------------------------------------------------------
__global__ void vector_add_impl(cg1t::element_p2* vec_a, cg1t::element_p2* vec_b, cg1t::element_p2* vec_r, size_t n_elments) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n_elments) {
      cg1o::add(vec_r[tid], vec_a[tid], vec_b[tid]);
    }
}

//--------------------------------------------------------------------------------------------------
// vector_add 
//--------------------------------------------------------------------------------------------------
void vector_add(cg1t::element_p2* vec_b, cg1t::element_p2* vec_a, cg1t::element_p2* vec_result, size_t n_elments) {
    const unsigned int num_blocks = static_cast<unsigned int>(ceil(static_cast<float>(n_elments) / MAX_THREADS_PER_BLOCK));
    const unsigned int threads_per_block = MAX_THREADS_PER_BLOCK;

    vector_add_impl<<<num_blocks, threads_per_block>>>(vec_a, vec_b, vec_result, n_elments);
}

//--------------------------------------------------------------------------------------------------
// init_random_array_impl 
//--------------------------------------------------------------------------------------------------
__global__ static void init_random_array_impl(cg1t::element_p2* rand, size_t n_elments) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n_elments)
    {
        basn::fast_random_number_generator rng{static_cast<uint64_t>(tid + 1),
                                               static_cast<uint64_t>(n_elments + 1)};
        cg1rn::generate_random_element(rand[tid], rng);
    }
}

//--------------------------------------------------------------------------------------------------
// init_random_array
//--------------------------------------------------------------------------------------------------
void init_random_array(cg1t::element_p2* rand, size_t n_elments) {
    const unsigned int num_blocks = static_cast<unsigned int>(ceil(static_cast<float>(n_elments) / MAX_THREADS_PER_BLOCK));
    const unsigned int threads_per_block = MAX_THREADS_PER_BLOCK;

    init_random_array_impl<<<num_blocks, threads_per_block>>>(rand, n_elments);
}

//--------------------------------------------------------------------------------------------------
// add
//--------------------------------------------------------------------------------------------------
void add(size_t vector_size, size_t repetitions) noexcept {
  std::cout << "Blitzar benchmark primative: curve element addition" << std::endl;
  std::cout << "Vector size = " << vector_size << std::endl;
  std::cout << "Repetitions = " << repetitions << std::endl;
  
  // Allocate memory for the input and output vectors
  memmg::managed_array<cg1t::element_p2> a(vector_size, memr::get_device_resource());
  memmg::managed_array<cg1t::element_p2> b(vector_size, memr::get_device_resource());
  memmg::managed_array<cg1t::element_p2> ret(vector_size, memr::get_device_resource());

  // Populate the input vectors with random curve elements
  init_random_array(a.data(), vector_size);
  init_random_array(b.data(), vector_size);

  // Warm-up loop
  std::cout << "Starting warm-up" << std::endl;
  for (int i = 0; i < repetitions; i++) {
    vector_add(a.data(), b.data(), ret.data(), vector_size);
  }

  // Benchmarking loop
  std::cout << "Starting benchmarking" << std::endl;
  auto start_time = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < repetitions; i++) {
    vector_add(a.data(), b.data(), ret.data(), vector_size);
  }
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

  // Report data
  std::cout << "Elapsed time: " << duration.count() << " microseconds" << std::endl;
  double GMPS = 1.0e-9 * repetitions * vector_size / (1.0e-6 * duration.count()) ;
  std::cout << "Performance: " << GMPS << " Giga curve additions Per Second" << std::endl;

  // Copy the result back to the host. Not necessary for performance measurements.
  // memmg::managed_array<cg1t::element_p2> res(vector_size);
  // cudaMemcpy(res.data(), ret.data(), ret.num_bytes(), cudaMemcpyDeviceToHost);
}
}
