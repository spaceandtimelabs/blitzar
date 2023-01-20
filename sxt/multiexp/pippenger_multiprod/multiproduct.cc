#include "sxt/multiexp/pippenger_multiprod/multiproduct.h"

#include <cstdlib>
#include <limits>
#include <vector>

#include "sxt/base/container/span_void.h"
#include "sxt/base/error/assert.h"
#include "sxt/base/iterator/counting_iterator.h"
#include "sxt/multiexp/index/index_table.h"
#include "sxt/multiexp/pippenger_multiprod/clump_inputs.h"
#include "sxt/multiexp/pippenger_multiprod/clump_outputs.h"
#include "sxt/multiexp/pippenger_multiprod/driver.h"
#include "sxt/multiexp/pippenger_multiprod/multiproduct_params.h"
#include "sxt/multiexp/pippenger_multiprod/multiproduct_params_computation.h"
#include "sxt/multiexp/pippenger_multiprod/partition_inputs.h"
#include "sxt/multiexp/pippenger_multiprod/product_table_normalization.h"
#include "sxt/multiexp/pippenger_multiprod/prune.h"
#include "sxt/multiexp/pippenger_multiprod/reduction_stats.h"

namespace sxt::mtxpmp {
//--------------------------------------------------------------------------------------------------
// prune_and_permute_products
//--------------------------------------------------------------------------------------------------
static void prune_and_permute_products(basct::span_void inout,
                                       basct::span<basct::span<uint64_t>> products,
                                       size_t& num_inactive_outputs, size_t& num_inactive_inputs,
                                       const driver& drv, size_t num_active_inputs) noexcept {
  std::vector<uint64_t> permutation{basit::counting_iterator<uint64_t>{0},
                                    basit::counting_iterator<uint64_t>{num_active_inputs}};
  prune_rows(products, permutation, num_inactive_outputs, num_inactive_inputs);
  drv.permute_inputs(inout, permutation);
}

//--------------------------------------------------------------------------------------------------
// compute_multiproduct
//--------------------------------------------------------------------------------------------------
void compute_multiproduct(basct::span_void inout, basct::span<basct::span<uint64_t>> products,
                          const driver& drv, size_t num_inputs) noexcept {
  size_t num_inactive_outputs = 0;
  size_t num_inactive_inputs = 0;
  prune_and_permute_products(inout, products, num_inactive_outputs, num_inactive_inputs, drv,
                             num_inputs);
  multiproduct_params params;
  compute_multiproduct_params(params, products.size() - num_inactive_outputs,
                              num_inputs - num_inactive_inputs);
  if (params.partition_size > 0) {
    reduction_stats stats;
    auto num_inactive_inputs_p = num_inactive_inputs;
    partition_inputs(inout.subspan(num_inactive_inputs), stats, products, num_inactive_outputs,
                     num_inactive_inputs_p, drv, params.partition_size);
    auto num_active_inputs = stats.num_terms - (num_inactive_inputs_p - num_inactive_inputs);
    num_inactive_inputs = num_inactive_inputs_p;
    prune_and_permute_products(inout.subspan(num_inactive_inputs), products, num_inactive_outputs,
                               num_inactive_inputs, drv, num_active_inputs);
  }
  while (num_inactive_outputs < products.size()) {
    reduction_stats stats;
    size_t num_inactive_inputs_p = num_inactive_inputs;
    clump_inputs(inout.subspan(num_inactive_inputs), stats, products, num_inactive_outputs,
                 num_inactive_inputs_p, drv, params.input_clump_size);
    if (stats.prev_num_terms == stats.num_terms) {
      break;
    }
    auto num_active_inputs = stats.num_terms - (num_inactive_inputs_p - num_inactive_inputs);
    num_inactive_inputs = num_inactive_inputs_p;

    mtxi::index_table clumped_output_table;
    std::vector<uint64_t> output_clumps;
    if (!compute_clumped_output_table(clumped_output_table, output_clumps,
                                      products.subspan(num_inactive_outputs), num_active_inputs,
                                      params.output_clump_size)) {
      break;
    }
    compute_multiproduct(inout.subspan(num_inactive_inputs), clumped_output_table.header(), drv,
                         num_active_inputs);
    rewrite_multiproducts_with_output_clumps(products.subspan(num_inactive_outputs), output_clumps,
                                             params.output_clump_size);
    num_active_inputs = output_clumps.size();
    prune_and_permute_products(inout.subspan(num_inactive_inputs), products, num_inactive_outputs,
                               num_inactive_inputs, drv, num_active_inputs);
  }

  drv.compute_naive_multiproduct(inout, products, num_inactive_inputs);
}

void compute_multiproduct(basct::span_void inout, mtxi::index_table& products, const driver& drv,
                          size_t num_inputs) noexcept {
  SXT_DEBUG_ASSERT(inout.size() >= num_inputs && inout.size() >= products.num_rows());
  normalize_product_table(products, inout.size());
  compute_multiproduct(inout, products.header(), drv, num_inputs);
}
} // namespace sxt::mtxpmp
