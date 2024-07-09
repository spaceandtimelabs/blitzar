/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2023-present Space and Time Labs, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/**
 * Adopted from zcash/librustzcash
 *
 * Copyright (c) 2017
 * Zcash Company
 *
 * See third_party/license/zcash.LICENSE
 */
#pragma once

#include "sxt/curve_gk/type/element_affine.h"
#include "sxt/curve_gk/type/element_p2.h"
#include "sxt/fieldgk/constant/one.h"
#include "sxt/fieldgk/type/element.h"

namespace sxt::cgkcn {
//--------------------------------------------------------------------------------------------------
// generator_x_v
//--------------------------------------------------------------------------------------------------
/**
 * x = 1 in Montgomery form
 */
static constexpr fgkt::element generator_x_v{fgkcn::one_v};

//--------------------------------------------------------------------------------------------------
// generator_y_v
//--------------------------------------------------------------------------------------------------
/**
 * y = sqrt(-16) in Montgomery form
 */
static constexpr fgkt::element generator_y_v{0x11b2dff1448c41d8, 0x23d3446f21c77dc3,
                                             0xaa7b8cf435dfafbb, 0x14b34cf69dc25d68};

//--------------------------------------------------------------------------------------------------
// generator_affine_v
//--------------------------------------------------------------------------------------------------
/**
 * Generator of G1 (x, y) = (1, sqrt(-16))
 */
static constexpr cgkt::element_affine generator_affine_v{generator_x_v, generator_y_v, false};

//--------------------------------------------------------------------------------------------------
// generator_p2_v
//--------------------------------------------------------------------------------------------------
/**
 * Generator of G1 (x, y, z) = (1, sqrt(-16), 1)
 */
static constexpr cgkt::element_p2 generator_p2_v{generator_x_v, generator_y_v, fgkcn::one_v};
} // namespace sxt::cgkcn
