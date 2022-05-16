#include "sxt/seqcommit/cbindings/pedersen.h"

#include <iostream>
#include <cuda_runtime.h>

#include "sxt/base/container/span.h"
#include "sxt/seqcommit/base/commitment.h"
#include "sxt/multiexp/base/exponent_sequence.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/seqcommit/naive/commitment_computation_cpu.h"
#include "sxt/seqcommit/naive/commitment_computation_gpu.h"

using namespace sxt;

using bench_fn = void(*)(
    basct::span<sqcb::commitment> commitments,
    basct::cspan<mtxb::exponent_sequence> value_sequences) noexcept;

using span_value_sequences = basct::cspan<mtxb::exponent_sequence>;

static bench_fn backend_func = nullptr;

//--------------------------------------------------------------------------------------------------
// pre_initialize
//--------------------------------------------------------------------------------------------------
static void pre_initialize_gpu(bench_fn func) {
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

  func(commitments_fake, value_sequences_fake);
}

int sxt_init(const sxt_config* config) {
  if (config == nullptr) return 1;

  if (config->backend == SXT_BACKEND_CPU) {
    backend_func = sqcnv::compute_commitments_cpu;
    return 0;
  } else if (config->backend == SXT_BACKEND_GPU) {
    int num_devices;

    auto rcode = cudaGetDeviceCount(&num_devices);

    if (rcode != cudaSuccess) {
      num_devices = 0; 

      std::cout << "cudaGetDeviceCount failed: " << cudaGetErrorString(rcode)
                << "\n";
    }

    if (num_devices > 0) {
      backend_func = sqcnv::compute_commitments_gpu;

      pre_initialize_gpu(backend_func);
    } else {
      backend_func = sqcnv::compute_commitments_cpu;

      std::cout << "WARN: Using 'compute_commitments_cpu'. " << std::endl;
    }

    return 0;
  }

  return 1;
}

static int valid_descriptor(const sxt_sequence_descriptor *descriptor) {
  // verify if data pointers are valid
  if (descriptor->dense.n > 0 && descriptor->dense.data == nullptr) return 1;

  // verify if word size is inside the correct range (1 to 32)
  if (descriptor->dense.element_nbytes == 0 || descriptor->dense.element_nbytes > 32) return 1;

  return 0;
}

static int valid_sequence_descriptor(uint32_t num_sequences, const sxt_sequence_descriptor* descriptors) {
  // invalid pointers
  if (descriptors == nullptr) return 1;

  // verify if each descriptor is inside the ranges
  for (uint32_t commit_index = 0; commit_index < num_sequences; ++commit_index) {
    // verify if sequence type is already implemented
    if (descriptors->sequence_type != SXT_DENSE_SEQUENCE_TYPE) return 1;

    if (valid_descriptor(descriptors + commit_index)) return 1;
  }

  return 0;
}

int sxt_compute_pedersen_commitments(
    sxt_commitment* commitments, uint32_t num_sequences,
    const sxt_sequence_descriptor* descriptors) {

  if (num_sequences == 0) return 0;

  // backend not initialized (sxt_init not called correctly)
  if (backend_func == nullptr) return 1;

  // verify if input is valid
  if (commitments == nullptr
        || valid_sequence_descriptor(num_sequences, descriptors)) return 1;

  static_assert(sizeof(sqcb::commitment) == 
      sizeof(sxt_commitment), "types must be ABI compatible");

  basct::span<sqcb::commitment> commitments_result(
            reinterpret_cast<sqcb::commitment *>(commitments), num_sequences);

  memmg::managed_array<mtxb::exponent_sequence> sequences(num_sequences);

  static_assert(sizeof(mtxb:: exponent_sequence) == 
      sizeof(sxt_dense_sequence_descriptor), "types must be ABI compatible");

  // populating sequence object
  for (uint64_t i = 0; i < num_sequences; ++i) {
    sequences[i] = *(reinterpret_cast<const mtxb::exponent_sequence *>(&descriptors[i].dense));
  }

  basct::cspan<mtxb::exponent_sequence> value_sequences(sequences.data(),
                                                        num_sequences);
  
  backend_func(commitments_result, value_sequences);

  return 0;
}
