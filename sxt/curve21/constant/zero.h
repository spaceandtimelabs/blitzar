#pragma once

#include "sxt/curve21/type/element_cached.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/field51/constant/one.h"
#include "sxt/field51/constant/zero.h"

namespace sxt::c21cn {
//--------------------------------------------------------------------------------------------------
// zero_p3_v
//--------------------------------------------------------------------------------------------------
static constexpr c21t::element_p3 zero_p3_v{
    .X{f51cn::zero_v},
    .Y{f51cn::one_v},
    .Z{f51cn::one_v},
    .T{f51cn::zero_v},
};

//--------------------------------------------------------------------------------------------------
// zero_cached_v
//--------------------------------------------------------------------------------------------------
static constexpr c21t::element_cached zero_cached_v{
    .YplusX{f51cn::one_v},
    .YminusX{f51cn::one_v},
    .Z{f51cn::one_v},
    .T2d{f51cn::zero_v},
};
} // namespace sxt::c21cn
