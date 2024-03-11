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
#include <format>
#include <print>

#include "stats.h"

#include "sxt/base/num/divide_up.h"
#include "sxt/base/num/fast_random_number_generator.h"
#include "sxt/field12/operation/add.h"
#include "sxt/field12/operation/mul.h"
#include "sxt/field12/random/element.h"
#include "sxt/field12/type/element.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/device_resource.h"

namespace sxt {
//--------------------------------------------------------------------------------------------------
// vector_add_impl 
//--------------------------------------------------------------------------------------------------
template <typename T>
__global__ void vector_add_impl(T* __restrict__ vec_ret,
                                const T* __restrict__ vec_a,
                                unsigned n_elements,
                                unsigned repetitions) {
  auto tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < n_elements) {
    T x = vec_a[tid];
    auto res = x;
    for (unsigned i = 0; i < repetitions; ++i) {
      f12o::add(res, res, x);
    }
    vec_ret[tid] = res;
  }
}

//--------------------------------------------------------------------------------------------------
// vector_mul_impl 
//--------------------------------------------------------------------------------------------------
template <typename T>
__global__ void vector_mul_impl(T* __restrict__ vec_ret,
                                const T* __restrict__ vec_a,
                                unsigned n_elements,
                                unsigned repetitions) {
  auto tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < n_elements) {
    T x = vec_a[tid];
    auto res = x;
    for (unsigned i = 0; i < repetitions; ++i) {
      f12o::mul(res, res, x);
    }
    vec_ret[tid] = res;
  }
}

//--------------------------------------------------------------------------------------------------
// vector_add 
//--------------------------------------------------------------------------------------------------
void vector_add(f12t::element* vec_result, const f12t::element* vec_a, unsigned n_elements, unsigned repetitions, unsigned n_threads) {
    const unsigned threads_per_block = n_threads;
    const unsigned num_blocks = basn::divide_up(n_elements, threads_per_block);

    vector_add_impl<<<num_blocks, threads_per_block>>>(vec_result, vec_a, n_elements, repetitions);
}

//--------------------------------------------------------------------------------------------------
// vector_mul
//--------------------------------------------------------------------------------------------------
void vector_mul(f12t::element* vec_result, const f12t::element* vec_a, unsigned n_elements, unsigned repetitions, unsigned n_threads) {
    const unsigned threads_per_block = n_threads;
    const unsigned num_blocks = basn::divide_up(n_elements, threads_per_block);

    vector_mul_impl<<<num_blocks, threads_per_block>>>(vec_result, vec_a, n_elements, repetitions);
}

//--------------------------------------------------------------------------------------------------
// init_random_array_impl 
//--------------------------------------------------------------------------------------------------
__global__ void init_random_array_impl(f12t::element* __restrict__ rand, unsigned n_elements) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n_elements)
    {
        basn::fast_random_number_generator rng{static_cast<uint64_t>(tid + 1),
                                               static_cast<uint64_t>(n_elements + 1)};
        f12rn::generate_random_element(rand[tid], rng);
    }
}

//--------------------------------------------------------------------------------------------------
// init_random_array
//--------------------------------------------------------------------------------------------------
void init_random_array(f12t::element* rand, unsigned n_elements, unsigned n_threads) {
    const unsigned threads_per_block = n_threads;
    const unsigned num_blocks = basn::divide_up(n_elements, threads_per_block);

    init_random_array_impl<<<num_blocks, threads_per_block>>>(rand, n_elements);
}

//--------------------------------------------------------------------------------------------------
// add_bls12_381_g1_field_elements
//--------------------------------------------------------------------------------------------------
void add_bls12_381_field_elements(unsigned n_elements, unsigned repetitions, unsigned n_threads, unsigned n_executions) noexcept {
  std::println("add_bls12_381_field_elements");

  // Allocate memory for the input and output vectors
  memmg::managed_array<f12t::element> a(n_elements, memr::get_device_resource());
  memmg::managed_array<f12t::element> ret(n_elements, memr::get_device_resource());

  // Populate the input vectors with random curve elements
  init_random_array(a.data(), n_elements, n_threads);

  // Warm-up loop
  vector_add(ret.data(), a.data(), n_elements, repetitions, n_threads);
  cudaDeviceSynchronize();

  // Benchmarking loop
  std::vector<double> elapsed_times;
  for (unsigned i = 0; i < n_executions; ++i) {
    auto start_time = std::chrono::steady_clock::now();
    vector_add(ret.data(), a.data(), n_elements, repetitions, n_threads);
    cudaDeviceSynchronize();
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    elapsed_times.push_back(duration.count());
  }

  // Report data
  std::println("Final benchmarks over {} executions", n_executions);
  std::println("...Median : {} ms", sxt::median(elapsed_times));
  std::println("...Min    : {} ms", sxt::min(elapsed_times));
  std::println("...Max    : {} ms", sxt::max(elapsed_times));
  std::println("...Mean   : {} ms", sxt::mean(elapsed_times));
  std::println("...STD    : {} ms", sxt::std_dev(elapsed_times));
  std::println("...Performance : {} Giga Field Additions Per Second", sxt::gmps(elapsed_times, repetitions, n_elements));
  std::println("");
}

//--------------------------------------------------------------------------------------------------
// mul_bls12_381_field_elements
//--------------------------------------------------------------------------------------------------
void mul_bls12_381_field_elements(unsigned n_elements, unsigned repetitions, unsigned n_threads, unsigned n_executions) noexcept {
  std::println("mul_bls12_381_field_elements");

  // Allocate memory for the input and output vectors
  memmg::managed_array<f12t::element> a(n_elements, memr::get_device_resource());
  memmg::managed_array<f12t::element> ret(n_elements, memr::get_device_resource());

  // Populate the input vectors with random curve elements
  init_random_array(a.data(), n_elements, n_threads);

  // Warm-up loop
  vector_mul(ret.data(), a.data(), n_elements, repetitions, n_threads);
  cudaDeviceSynchronize();

  // Benchmarking loop
  std::vector<double> elapsed_times;
  for (unsigned i = 0; i < n_executions; ++i) {
    auto start_time = std::chrono::steady_clock::now();
    vector_mul(ret.data(), a.data(), n_elements, repetitions, n_threads);
    cudaDeviceSynchronize();
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    elapsed_times.push_back(duration.count());
  }

  // Report data
  std::println("Final benchmarks over {} executions", n_executions);
  std::println("...Median : {} ms", sxt::median(elapsed_times));
  std::println("...Min    : {} ms", sxt::min(elapsed_times));
  std::println("...Max    : {} ms", sxt::max(elapsed_times));
  std::println("...Mean   : {} ms", sxt::mean(elapsed_times));
  std::println("...STD    : {} ms", sxt::std_dev(elapsed_times));
  std::println("...Performance : {} Giga Field Additions Per Second", sxt::gmps(elapsed_times, repetitions, n_elements));
  std::println("");
}
}
