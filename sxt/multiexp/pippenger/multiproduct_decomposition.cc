#include "sxt/multiexp/pippenger/multiproduct_decomposition.h"

#include <algorithm>
#include <array>
#include <iterator>

#include "sxt/base/bit/bit_position.h"
#include "sxt/base/container/blob_array.h"
#include "sxt/base/num/constexpr_switch.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/multiexp/base/exponent_sequence.h"

namespace sxt::mtxpi {
//--------------------------------------------------------------------------------------------------
// count_max_products_and_indexes
//--------------------------------------------------------------------------------------------------
static void
count_max_products_and_indexes(size_t& max_products, size_t& max_indexes,
                               size_t& max_single_indexes, size_t& max_element_nbytes,
                               basct::cspan<mtxb::exponent_sequence> exponents) noexcept {
  max_products = 0;
  max_indexes = 0;
  max_single_indexes = 0;
  max_element_nbytes = 0;
  for (auto& sequence : exponents) {
    max_element_nbytes = std::max(max_element_nbytes, size_t{sequence.element_nbytes});
    size_t t = sequence.element_nbytes * 8;
    max_products += t;
    t *= sequence.n;
    max_single_indexes = std::max(max_single_indexes, t);
    max_indexes += t;
  }
}

//--------------------------------------------------------------------------------------------------
// compute_output_digit_or_all
//--------------------------------------------------------------------------------------------------
static void compute_output_digit_or_all(basct::blob_array& output_digit_or_all,
                                        basct::cspan<unsigned> product_sizes,
                                        basct::cspan<mtxb::exponent_sequence> exponents) noexcept {
  size_t bit_index = 0;
  for (size_t output_index = 0; output_index < exponents.size(); ++output_index) {
    auto or_all = output_digit_or_all[output_index];
    auto& sequence = exponents[output_index];
    for (size_t byte_index = 0; byte_index < sequence.element_nbytes; ++byte_index) {
      uint8_t val = 0;
      for (int i = 0; i < 8; ++i) {
        val |= static_cast<uint8_t>(product_sizes[bit_index++] > 0) << i;
      }
      or_all[byte_index] = val;
    }
  }
}

//--------------------------------------------------------------------------------------------------
// decompose_multiexponentiation
//--------------------------------------------------------------------------------------------------
template <unsigned NBytes>
static void decompose_multiexponentiation(unsigned*& indexes, unsigned*& counts,
                                          basct::span<unsigned> workspace,
                                          const mtxb::exponent_sequence& sequence) noexcept {
  constexpr size_t NumBits = NBytes * 8;
  std::fill_n(counts, NumBits, 0);
  basbt::compute_bit_positions(workspace, {sequence.data, NBytes * sequence.n});
  for (auto pos : workspace) {
    ++counts[pos % NumBits];
  }
  std::array<unsigned*, NumBits> index_table;
  for (size_t i = 0; i < index_table.size(); ++i) {
    index_table[i] = indexes;
    indexes += counts[i];
  }
  for (auto pos : workspace) {
    *index_table[pos % NumBits]++ = pos / NumBits;
  }
  counts += NumBits;
}

//--------------------------------------------------------------------------------------------------
// compute_multiproduct_decomposition
//--------------------------------------------------------------------------------------------------
void compute_multiproduct_decomposition(memmg::managed_array<unsigned>& indexes,
                                        memmg::managed_array<unsigned>& product_sizes,
                                        basct::blob_array& output_digit_or_all,
                                        basct::cspan<mtxb::exponent_sequence> exponents) noexcept {
  size_t max_products, max_indexes, max_single_indexes, max_element_nbytes;
  count_max_products_and_indexes(max_products, max_indexes, max_single_indexes, max_element_nbytes,
                                 exponents);
  indexes = memmg::managed_array<unsigned>{
      max_indexes,
      indexes.get_allocator(),
  };
  product_sizes = memmg::managed_array<unsigned>{
      max_products,
      product_sizes.get_allocator(),
  };
  output_digit_or_all.resize(exponents.size(), max_element_nbytes);
  memmg::managed_array<unsigned> workspace(max_single_indexes);
  auto indexes_out = indexes.data();
  auto counts_out = product_sizes.data();
  for (auto& sequence : exponents) {
    basn::constexpr_switch<1, 33>(
        sequence.element_nbytes, [&]<unsigned N>(std::integral_constant<unsigned, N>) noexcept {
          decompose_multiexponentiation<N>(indexes_out, counts_out, workspace, sequence);
        });
  }
  indexes.shrink(static_cast<size_t>(std::distance(indexes.data(), indexes_out)));
  compute_output_digit_or_all(output_digit_or_all, product_sizes, exponents);
  auto iter = std::remove(product_sizes.begin(), product_sizes.end(), 0);
  product_sizes.shrink(std::distance(product_sizes.begin(), iter));
}
} // namespace sxt::mtxpi
