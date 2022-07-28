#pragma once

#include <cinttypes>

#include "sxt/base/container/span.h"

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
// pedersen_backend
//--------------------------------------------------------------------------------------------------
class pedersen_backend {
public:
  virtual ~pedersen_backend() noexcept = default;

  virtual void compute_commitments(basct::span<rstt::compressed_element> commitments,
                                   basct::cspan<sqcb::indexed_exponent_sequence> value_sequences,
                                   basct::cspan<c21t::element_p3> generators,
                                   uint64_t longest_sequence,
                                   bool has_sparse_sequence) noexcept = 0;

  virtual void get_generators(basct::span<c21t::element_p3> generators,
                              uint64_t offset_generators) noexcept = 0;
};

} // namespace sxt::sqcbck
