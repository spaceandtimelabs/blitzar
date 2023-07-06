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
/*
 * Adopted from zcash/librustzcash
 *
 * Copyright (c) 2017
 * Zcash Company
 *
 * See third_party/license/zcash.LICENSE
 */
#pragma once

#include "sxt/curve_g1/type/element_affine.h"
#include "sxt/curve_g1/type/element_p2.h"
#include "sxt/field12/constant/one.h"
#include "sxt/field12/type/element.h"

namespace sxt::cg1cn {
//--------------------------------------------------------------------------------------------------
// generator_x_v
//--------------------------------------------------------------------------------------------------
/*
 The generators of G1/G2 are computed by finding the lexicographically smallest valid x coordinate,
 and its lexicographically smallest y coordinate and multiplying it by the cofactor such that the
 result is nonzero.

 Generator of G1
 x =
 3685416753713387016781088315183077757961620795782546409894578378688607592378376318836054947676345821548104185464507
 y =
 1339506544944476473020471379941921221584933875938349620426543736416511423956333506472724655353366534992391756441569
 */
static constexpr f12t::element generator_x_v{0x5cb38790fd530c16, 0x7817fc679976fff5,
                                             0x154f95c7143ba1c1, 0xf0ae6acdf3d0e747,
                                             0xedce6ecc21dbf440, 0x120177419e0bfb75};

//--------------------------------------------------------------------------------------------------
// generator_y_v
//--------------------------------------------------------------------------------------------------
static constexpr f12t::element generator_y_v{0xbaac93d50ce72271, 0x8c22631a7918fd8e,
                                             0xdd595f13570725ce, 0x51ac582950405194,
                                             0x0e1c8c3fad0059c0, 0x0bbc3efc5008a26a};

//--------------------------------------------------------------------------------------------------
// generator_affine_v
//--------------------------------------------------------------------------------------------------
static constexpr cg1t::element_affine generator_affine_v{
    .X{generator_x_v}, .Y{generator_y_v}, .infinity{false}};

//--------------------------------------------------------------------------------------------------
// generator_p2_v
//--------------------------------------------------------------------------------------------------
static constexpr cg1t::element_p2 generator_p2_v{
    .X{generator_x_v}, .Y{generator_y_v}, .Z{f12cn::one_v}};
} // namespace sxt::cg1cn
