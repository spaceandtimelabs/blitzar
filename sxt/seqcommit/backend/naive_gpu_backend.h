#pragma once

#include <cinttypes>

#include "sxt/base/container/span.h"
#include "sxt/seqcommit/backend/pedersen_backend.h"

namespace sxt::sqcb {
struct indexed_exponent_sequence;
}
namespace sxt::rstt {
class compressed_element;
}

namespace sxt::sqcbck {

//--------------------------------------------------------------------------------------------------
// naive_gpu_backend
//--------------------------------------------------------------------------------------------------
class naive_gpu_backend final : public pedersen_backend {
public:
  naive_gpu_backend();

  void compute_commitments(basct::span<rstt::compressed_element> commitments,
                           basct::cspan<sqcb::indexed_exponent_sequence> value_sequences,
                           basct::span<rstt::compressed_element> generators) noexcept override;

  void get_generators(basct::span<rstt::compressed_element> generators,
                      uint64_t offset_generators) noexcept override;
};

//--------------------------------------------------------------------------------------------------
// get_naive_gpu_backend
//--------------------------------------------------------------------------------------------------
naive_gpu_backend* get_naive_gpu_backend();

} // namespace sxt::sqcbck
