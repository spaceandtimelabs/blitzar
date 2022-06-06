#pragma once

#include <cinttypes>

#include "sxt/base/container/span.h"

namespace sxt::mtxb { struct exponent_sequence; }
namespace sxt::sqcb { class commitment; }

namespace sxt::sqccb {

class pedersen_backend {
public:
  virtual ~pedersen_backend() noexcept = default;

  virtual void compute_commitments(
    basct::span<sqcb::commitment> commitments,
    basct::cspan<mtxb::exponent_sequence> value_sequences,
    basct::span<sqcb::commitment> generators) noexcept = 0;

  virtual void get_generators(
    basct::span<sqcb::commitment> generators,
    uint64_t offset_generators) noexcept = 0;
};

} // namespace sxt::sqccb
