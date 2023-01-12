#pragma once

#include <cinttypes>

#include "sxt/base/container/span.h"
#include "sxt/cbindings/backend/computational_backend.h"

namespace sxt::sqcb {
struct indexed_exponent_sequence;
}
namespace sxt::rstt {
class compressed_element;
}
namespace sxt::c21t {
struct element_p3;
}

namespace sxt::cbnbck {

//--------------------------------------------------------------------------------------------------
// gpu_backend
//--------------------------------------------------------------------------------------------------
class gpu_backend final : public computational_backend {
public:
  gpu_backend();

  void compute_commitments(basct::span<rstt::compressed_element> commitments,
                           basct::cspan<sqcb::indexed_exponent_sequence> value_sequences,
                           basct::cspan<c21t::element_p3> generators,
                           uint64_t length_longest_sequence,
                           bool has_sparse_sequence) noexcept override;

  void get_generators(basct::span<c21t::element_p3> generators,
                      uint64_t offset_generators) noexcept override;
};

//--------------------------------------------------------------------------------------------------
// get_gpu_backend
//--------------------------------------------------------------------------------------------------
gpu_backend* get_gpu_backend();

} // namespace sxt::cbnbck
