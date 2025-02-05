#pragma once  

#include <cstddef>

#include "sxt/base/container/span.h"

namespace sxt::s25t {
class element;
}

namespace sxt::prfsk {
//--------------------------------------------------------------------------------------------------
// sumcheck_transcript 
//--------------------------------------------------------------------------------------------------
struct sumcheck_transcript {
  public:
    virtual ~sumcheck_transcript() noexcept = default;

    virtual void init(size_t num_variables, size_t round_degree) noexcept = 0;

    virtual void round_challenge(s25t::element& r,
                                 basct::cspan<s25t::element> polynomial) noexcept = 0;
};
} // namespace sxt::prfsk
