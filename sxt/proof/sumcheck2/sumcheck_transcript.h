#pragma once

#include "sxt/base/field/element.h"
#include "sxt/base/container/span.h"

namespace sxt::prfsk2 {
//--------------------------------------------------------------------------------------------------
// sumcheck_transcript
//--------------------------------------------------------------------------------------------------
template <basfld::element T>
class sumcheck_transcript {
public:
  virtual ~sumcheck_transcript() noexcept = default;

  virtual void init(size_t num_variables, size_t round_degree) noexcept = 0;

  virtual void round_challenge(T& r, basct::cspan<T> polynomial) noexcept = 0;
};
} // namespace sxt::prfsk2
