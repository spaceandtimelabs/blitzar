#pragma once

#include <cinttypes>

#include "sxt/base/container/span.h"
#include "sxt/seqcommit/cbindings/pedersen_backend.h"

namespace sxt::sqcb { struct indexed_exponent_sequence; }
namespace sxt::sqcb { class commitment; }

namespace sxt::sqccb {

class pedersen_cpu_backend final : public sqccb::pedersen_backend {
public:
  void compute_commitments(
    basct::span<sqcb::commitment> commitments,
    basct::cspan<sqcb::indexed_exponent_sequence> value_sequences,
    basct::span<sqcb::commitment> generators) noexcept override;

  void get_generators(
    basct::span<sqcb::commitment> generators,
    uint64_t offset_generators) noexcept override;
};

pedersen_cpu_backend* get_pedersen_cpu_backend();

} // namespace sxt::sqccb
