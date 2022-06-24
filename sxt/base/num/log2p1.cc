#include "sxt/base/num/log2p1.h"

#include <cmath>
#include <cassert>

namespace sxt::basn {
//--------------------------------------------------------------------------------------------------
// log2p1
//--------------------------------------------------------------------------------------------------
double log2p1(basct::cspan<uint8_t> x) noexcept {
    double res = 1;
    double power256 = 1.;

    // only numbers smaller than 127 bytes (1016 bits) are allowed
    assert(x.size() <= 127);
    
    for (size_t i = 0; i < x.size(); ++i) {
        res += static_cast<double>(x[i]) * power256;
        power256 *= 256.;
    }

    return std::log2(res);
}
}
