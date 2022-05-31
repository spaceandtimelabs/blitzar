#include "sxt/seqcommit/cbindings/pedersen_gpu_backend.h"

#include "sxt/seqcommit/base/commitment.h"
#include "sxt/multiexp/base/exponent_sequence.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/seqcommit/generator/gpu_generator.h"
#include "sxt/seqcommit/naive/commitment_computation_gpu.h"

namespace sxt::sqccb {
//--------------------------------------------------------------------------------------------------
// pre_initialize_gpu
//--------------------------------------------------------------------------------------------------
static void pre_initialize_gpu() {
  // initialization of dummy variables
  memmg::managed_array<uint8_t> data_table_fake(1); // 1 col, 1 row, 1 bytes per data
  memmg::managed_array<sqcb::commitment> commitments_per_col_fake(1);
  memmg::managed_array<mtxb::exponent_sequence> data_cols_fake(1);
  basct::span<sqcb::commitment> commitments_fake(commitments_per_col_fake.data(), 1);
  basct::cspan<mtxb::exponent_sequence> value_sequences_fake(data_cols_fake.data(), 1);

  data_table_fake[0] = 1;

  auto &data_col = data_cols_fake[0];

  data_col.n = 1;
  data_col.element_nbytes = 1;
  data_col.data = data_table_fake.data();

  // A small dummy computation to avoid the future cost of JIT compiling PTX code
  sqcnv::compute_commitments_gpu(commitments_fake, value_sequences_fake);
}

//--------------------------------------------------------------------------------------------------
// pedersen_gpu_backend
//--------------------------------------------------------------------------------------------------
pedersen_gpu_backend::pedersen_gpu_backend() {
  pre_initialize_gpu();
}

//--------------------------------------------------------------------------------------------------
// compute_commitments
//--------------------------------------------------------------------------------------------------
void pedersen_gpu_backend::compute_commitments(
    basct::span<sqcb::commitment> commitments,
    basct::cspan<mtxb::exponent_sequence> value_sequences) noexcept {
    
    sqcnv::compute_commitments_gpu(commitments, value_sequences);
}

//--------------------------------------------------------------------------------------------------
// get_generators
//--------------------------------------------------------------------------------------------------
void pedersen_gpu_backend::get_generators(
    basct::span<sqcb::commitment> generators,
    uint64_t offset_generators) noexcept {

    sqcgn::gpu_get_generators(generators, offset_generators);
}

//--------------------------------------------------------------------------------------------------
// get_pedersen_gpu_backend
//--------------------------------------------------------------------------------------------------
pedersen_gpu_backend* get_pedersen_gpu_backend() {
    // see https://isocpp.org/wiki/faq/ctors#static-init-order-on-first-use
    static pedersen_gpu_backend* backend = new pedersen_gpu_backend{};
    return backend;
}
}
