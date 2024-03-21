#include <iostream>
#include <random>
#include <chrono>

#include "sxt/curve21/operation/add.h"
#include "sxt/curve21/operation/double.h"
#include "sxt/curve21/operation/neg.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/curve21/type/literal.h"
#include "sxt/execution/schedule/scheduler.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/multiexp/pippenger2/multiexponentiation.h"
using namespace sxt;
using c21t::operator""_c21;

using E = c21t::element_p3;
const auto element_num_bytes = 32u;

//--------------------------------------------------------------------------------------------------
// make_lookup_arrray
//--------------------------------------------------------------------------------------------------
static memmg::managed_array<E> make_lookup_array(unsigned n) {
  memmg::managed_array<E> res((1u << 16u) * basn::divide_up(n, 16u));
  E ex[] = {0x123_c21, 0x456_c21, 0x789_c21};
  for (unsigned i = 0; i < res.size(); ++i) {
    res[i] = ex[i % 3];
  }
  return res;
}

//--------------------------------------------------------------------------------------------------
// generate_scalars
//--------------------------------------------------------------------------------------------------
static void generate_scalars(memmg::managed_array<const uint8_t*>& scalars,
                             memmg::managed_array<uint8_t>& data_table, unsigned num_outputs,
                             unsigned n) {
  scalars.resize(num_outputs);
  data_table.resize(element_num_bytes * num_outputs * n);

  std::mt19937 gen{0};
  std::uniform_int_distribution<uint8_t> distribution{0, std::numeric_limits<uint8_t>::max()};

  for (size_t i = 0; i < data_table.size(); ++i) {
    data_table[i] = distribution(gen);
  }

  for (unsigned output_index = 0; output_index < num_outputs; ++output_index) {
    scalars[output_index] = data_table.data() + output_index * n * element_num_bytes;
  }
}

//--------------------------------------------------------------------------------------------------
// main
//--------------------------------------------------------------------------------------------------
int main() {
  /* auto n = 16u * 100; */
  auto n = 1024u;
  auto num_outputs = 1024u;
  auto num_iterations = 5u;

  // make lookup table
  auto table = make_lookup_array(n);

  // make scalars
  memmg::managed_array<const uint8_t*> scalars;
  memmg::managed_array<uint8_t> data;
  generate_scalars(scalars, data, num_outputs, n);

  // compute multi-exponentiation
  memmg::managed_array<E> res(num_outputs);
  double duration = 0.0;
  for (unsigned i = 0; i < num_iterations; ++i) {
    auto t1 = std::chrono::steady_clock::now();
    auto fut = mtxpp2::multiexponentiate<E>(res, table, scalars, element_num_bytes, n);
    xens::get_scheduler().run();
    auto t2 = std::chrono::steady_clock::now();
    duration += std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
  }
  duration /= num_iterations;
  std::cout << "duration (s): " << duration / 1000.0 << std::endl;;

  return 0;
}
