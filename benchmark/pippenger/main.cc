#include <iostream>
#include <random>

#include "sxt/curve21/type/element_p3.h"
#include "sxt/curve21/type/literal.h"
#include "sxt/multiexp/pippenger2/multiexponentiation.h"
#include "sxt/memory/management/managed_array.h"
using namespace sxt;
using c21t::operator""_c21;

using E = c21t::element_p3;

//--------------------------------------------------------------------------------------------------
// make_lookup_arrray
//--------------------------------------------------------------------------------------------------
static memmg::managed_array<E> make_lookup_array(unsigned n) {
  memmg::managed_array<E> res((1u << 16u) * n / 16u);
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
                             memmg::managed_array<uint8_t> data_table, unsigned num_outputs,
                             unsigned n) {
  const auto element_nbytes = 32u;
  scalars.resize(num_outputs);
  data_table.resize(element_nbytes * num_outputs * n);

  std::mt19937 gen{0};
  std::uniform_int_distribution<uint8_t> distribution{0, std::numeric_limits<uint8_t>::max()};

  for (size_t i = 0; i < data_table.size(); ++i) {
    data_table[i] = distribution(gen);
  }

  for (unsigned output_index = 0; output_index < num_outputs; ++output_index) {
    scalars[output_index] = data_table.data() + output_index * n * element_nbytes;
  }
}

//--------------------------------------------------------------------------------------------------
// main
//--------------------------------------------------------------------------------------------------
int main() {
  auto n = 1024u;
  auto num_outputs = 1024u;

  std::cout << "arf\n";
  // make lookup table
  auto table = make_lookup_array(1024);

  // make scalars
  memmg::managed_array<const uint8_t*> scalars;
  memmg::managed_array<uint8_t> data_table;
  generate_scalars(scalars, data_table, num_outputs, n);
  std::cout << "nuf\n";
  (void)generate_scalars;
/* template <bascrv::element T> */
/* xena::future<> multiexponentiate(basct::span<T> res, basct::cspan<T> partition_table, */
/*                                  basct::cspan<const uint8_t*> exponents, */
/*                                  unsigned element_num_bytes, */
/*                                  unsigned n) noexcept { */
  return 0;
}
