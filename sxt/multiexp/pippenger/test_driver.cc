#include "sxt/multiexp/pippenger/test_driver.h"

#include <cassert>
#include <cstdint>
#include <iostream>

#include "sxt/base/bit/iteration.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/multiexp/index/index_table.h"

namespace sxt::mtxpi {
//--------------------------------------------------------------------------------------------------
// compute_multiproduct_inputs
//--------------------------------------------------------------------------------------------------
void test_driver::compute_multiproduct_inputs(
    memmg::managed_array<void>& inout,
    basct::cspan<basct::cspan<size_t>> powers,
    size_t radix_log2) const noexcept {
  assert(inout.size() == powers.size());
  size_t num_terms = 0;
  for (auto& subpowers : powers) {
    num_terms += subpowers.size();
  }
  basct::cspan<uint64_t> inputs{static_cast<uint64_t*>(inout.data()),
                                inout.size()};
  memmg::managed_array<uint64_t> terms{num_terms, inout.get_allocator()};
  size_t term_index = 0;
  for(size_t input_index=0; input_index<inputs.size(); ++input_index) {
    for (auto power : powers[input_index]) {
      terms[term_index++] = (1 << power * radix_log2) * inputs[input_index];
    }
  }
  inout = std::move(terms);
}

//--------------------------------------------------------------------------------------------------
// compute_multiproduct
//--------------------------------------------------------------------------------------------------
void test_driver::compute_multiproduct(
       memmg::managed_array<void>& inout,
       mtxi::index_table& multiproduct_table) const noexcept {
  basct::cspan<uint64_t> inputs{static_cast<uint64_t*>(inout.data()),
                                inout.size()};
  memmg::managed_array<uint64_t> outputs{multiproduct_table.num_rows(),
                                         inout.get_allocator()};
  for (size_t row_index = 0; row_index < multiproduct_table.num_rows();
       ++row_index) {
    auto products = multiproduct_table.header()[row_index];
    assert(!products.empty());
    auto& output = outputs[row_index];
    output = 0;
    for (auto input : products) {
      assert(input < inputs.size());
      output += inputs[input];
    }
  }
  inout = std::move(outputs);
}

//--------------------------------------------------------------------------------------------------
// combine_multiproduct_outputs
//--------------------------------------------------------------------------------------------------
void test_driver::combine_multiproduct_outputs(
    memmg::managed_array<void>& inout,
    basct::cspan<uint8_t> output_digit_or_all) const noexcept {
  basct::cspan<uint64_t> inputs{static_cast<uint64_t*>(inout.data()),
                                inout.size()};
  memmg::managed_array<uint64_t> outputs{output_digit_or_all.size(),
                                         inout.get_allocator()};
  size_t input_index = 0;
  for (size_t output_index = 0; output_index < output_digit_or_all.size();
       ++output_index) {
    auto& output = outputs[output_index];
    output = 0;
    uint64_t digit_or_all = output_digit_or_all[output_index];
    while (digit_or_all != 0) {
      auto pos = basbt::consume_next_bit(digit_or_all);
      output += (1 << pos) * inputs[input_index++];
    }
  }
  inout = std::move(outputs);
}
}  // namespace sxt::mtxpi
