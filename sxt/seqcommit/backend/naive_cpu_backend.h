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
namespace sxt::c21t {
struct element_p3;
}

namespace sxt::sqcbck {

//--------------------------------------------------------------------------------------------------
// naive_cpu_backend
//--------------------------------------------------------------------------------------------------
class naive_cpu_backend final : public sqcbck::pedersen_backend {
public:
  void compute_commitments(basct::span<rstt::compressed_element> commitments,
                           basct::cspan<sqcb::indexed_exponent_sequence> value_sequences,
                           basct::cspan<c21t::element_p3> generators,
                           uint64_t length_longest_sequence,
                           bool has_sparse_sequence) noexcept override;

  void get_generators(basct::span<c21t::element_p3> generators,
                      uint64_t offset_generators) noexcept override;
};

//--------------------------------------------------------------------------------------------------
// get_naive_cpu_backend
//--------------------------------------------------------------------------------------------------
naive_cpu_backend* get_naive_cpu_backend();

} // namespace sxt::sqcbck
