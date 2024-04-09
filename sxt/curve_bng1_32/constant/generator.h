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

#include "sxt/curve_bng1_32/type/element_affine.h"
#include "sxt/curve_bng1_32/type/element_p2.h"
#include "sxt/field32/constant/one.h"
#include "sxt/field32/type/element.h"

namespace sxt::cn3cn {
//--------------------------------------------------------------------------------------------------
// generator_x_v
//--------------------------------------------------------------------------------------------------
/**
 * The generators of G1 is computed by finding the lexicographically smallest valid x coordinate,
 * and its lexicographically smallest y coordinate and multiplying it by the cofactor such that the
 * result is nonzero.
 *
 * Generator of G1 (x, y) = (1, 2).
 * Cofactor of G1 is 1.
 */
static constexpr f32t::element generator_x_v{f32cn::one_v};

//--------------------------------------------------------------------------------------------------
// generator_y_v
//--------------------------------------------------------------------------------------------------
// static constexpr f32t::element generator_y_v{0xa6ba871b8b1e1b3a, 0x14f1d651eb8e167b,
//                                              0xccdd46def0f28c58, 0x1c14ef83340fbe5e};
static constexpr f32t::element generator_y_v{0x8b1e1b3a, 0xa6ba871b, 0xeb8e167b, 0x14f1d651,
                                             0xf0f28c58, 0xccdd46de, 0x340fbe5e, 0x1c14ef83};

//--------------------------------------------------------------------------------------------------
// generator_affine_v
//--------------------------------------------------------------------------------------------------
static constexpr cn3t::element_affine generator_affine_v{generator_x_v, generator_y_v, false};

//--------------------------------------------------------------------------------------------------
// generator_p2_v
//--------------------------------------------------------------------------------------------------
static constexpr cn3t::element_p2 generator_p2_v{generator_x_v, generator_y_v, f32cn::one_v};
} // namespace sxt::cn3cn
