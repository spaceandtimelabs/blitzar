/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2023 Space and Time Labs, Inc.
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
#pragma once

#include "sxt/field51/type/element.h"

namespace sxt::f51cn {
//--------------------------------------------------------------------------------------------------
// d_v
//--------------------------------------------------------------------------------------------------
/* 37095705934669439343138083508754565189542113879843219016388785533085940283555 */
static constexpr f51t::element d_v = {929955233495203, 466365720129213, 1662059464998953,
                                      2033849074728123, 1442794654840575};

//--------------------------------------------------------------------------------------------------
// d2_v
//--------------------------------------------------------------------------------------------------
/* 2 * d =
 * 16295367250680780974490674513165176452449235426866156013048779062215315747161
 */
static constexpr f51t::element d2_v = {1859910466990425, 932731440258426, 1072319116312658,
                                       1815898335770999, 633789495995903};

//--------------------------------------------------------------------------------------------------
// onemsqd_v
//--------------------------------------------------------------------------------------------------
/* 1 - d ^ 2 */
static constexpr f51t::element onemsqd_v = {1136626929484150, 1998550399581263, 496427632559748,
                                            118527312129759, 45110755273534};

//--------------------------------------------------------------------------------------------------
// sqdmone_v
//--------------------------------------------------------------------------------------------------
/* (d - 1) ^ 2 */
static constexpr f51t::element sqdmone_v = {1507062230895904, 1572317787530805, 683053064812840,
                                            317374165784489, 1572899562415810};

//--------------------------------------------------------------------------------------------------
// sqrtadm1_v
//--------------------------------------------------------------------------------------------------
/* sqrt(ad - 1) with a = -1 (mod p) */
static constexpr f51t::element sqrtadm1_v = {2241493124984347, 425987919032274, 2207028919301688,
                                             1220490630685848, 974799131293748};
} // namespace sxt::f51cn
