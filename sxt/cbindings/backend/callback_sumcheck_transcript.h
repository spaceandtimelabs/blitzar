#pragma once

#include "sxt/proof/sumcheck/sumcheck_transcript.h"

namespace sxt::cbnbck {
//--------------------------------------------------------------------------------------------------
// callback_sumcheck_transcript
//--------------------------------------------------------------------------------------------------
class callback_sumcheck_transcript final : public prfsk::sumcheck_transcript {
public:
  using callback_t = void (*)(s25t::element* r, void* context, const s25t::element* polynomial,
                              unsigned polynomial_len);

  callback_sumcheck_transcript(callback_t f, void* context) noexcept : f_{f}, context_{context} {}

  void init(size_t /*num_variables*/, size_t /*round_degree*/) noexcept override {}

  void round_challenge(s25t::element& r, basct::cspan<s25t::element> polynomial) noexcept override {
    f_(&r, context_, polynomial.data(), static_cast<unsigned>(polynomial.size()));
  }

private:
  callback_t f_;
  void* context_;
};
} // namespace sxt::cbnbck
