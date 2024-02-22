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
#pragma once

#include "sxt/field32/type/element.h"

namespace sxt::f32cn {
//--------------------------------------------------------------------------------------------------
// d_v
//--------------------------------------------------------------------------------------------------
/* (-121665/121666) % p_v
 * 37095705934669439343138083508754565189542113879843219016388785533085940283555 
 * 0x52036cee2b6ffe738cc740797779e89800700a4d4141d8ab75eb4dca135978a3
 * EDWARDS_D
 * https://github.com/dalek-cryptography/curve25519-dalek/blob/a62e4a5c573ca9a68503a6fbe47e3f189a4765b0/curve25519-dalek/src/backend/serial/u32/constants.rs#L33-L36
 */
static constexpr f32t::element d_v = {56195235, 13857412, 51736253, 6949390, 114729,
                                      24766616, 60832955, 30306712, 48412415, 21499315};

//--------------------------------------------------------------------------------------------------
// d2_v
//--------------------------------------------------------------------------------------------------
/* (2 * d_v) * p_v
 * 16295367250680780974490674513165176452449235426866156013048779062215315747161
 * 0x2406d9dc56dffce7198e80f2eef3d13000e0149a8283b156ebd69b9426b2f159
 * EDWARDS_D2
 * https://github.com/dalek-cryptography/curve25519-dalek/blob/a62e4a5c573ca9a68503a6fbe47e3f189a4765b0/curve25519-dalek/src/backend/serial/u32/constants.rs#L38-L41
 */
static constexpr f32t::element d2_v = {45281625, 27714825, 36363642, 13898781, 229458,
                                       15978800, 54557047, 27058993, 29715967, 9444199};

//--------------------------------------------------------------------------------------------------
// onemsqd_v
//--------------------------------------------------------------------------------------------------
/* (1-(d_v^2)) % p_v
 * 1159843021668779879193775521855586647937357759715417654439879720876111806838
 * 0x29072a8b2b3e0d79994abddbe70dfe42c81a138cd5e350fe27c09c1945fc176
 * ONE_MINUS_EDWARDS_D_SQUARED
 * https://github.com/dalek-cryptography/curve25519-dalek/blob/a62e4a5c573ca9a68503a6fbe47e3f189a4765b0/curve25519-dalek/src/backend/serial/u32/constants.rs#L43-L46
 */
static constexpr f32t::element onemsqd_v = {6275446, 16937061, 44170319, 29780721, 11667076,
                                            7397348, 39186143, 1766194, 42675006, 672202};

//--------------------------------------------------------------------------------------------------
// sqdmone_v
//--------------------------------------------------------------------------------------------------
/* (d - 1) ^ 2 
 * ((d_v-1)^2)%p_v
 * 40440834346308536858101042469323190826248399146238708352240133220865137265952
 * 0x5968b37af66c22414cdcd32f529b4eebd29e4a2cb01e199931ad5aaa44ed4d20
 * EDWARDS_D_MINUS_ONE_SQUARED
 * https://github.com/dalek-cryptography/curve25519-dalek/blob/a62e4a5c573ca9a68503a6fbe47e3f189a4765b0/curve25519-dalek/src/backend/serial/u32/constants.rs#L48-L52 
 */
static constexpr f32t::element sqdmone_v = {15551776, 22456977, 53683765, 23429360, 55212328,
                                            10178283, 40474537, 4729243, 61826754, 23438029};

//--------------------------------------------------------------------------------------------------
// sqrtadm1_v
//--------------------------------------------------------------------------------------------------
/* sqrt(a*d_v - 1)%p_v with a = -1 % p_v
 * 0x376931bf2b8348ac0f3cfcc931f5d1fdaf9d8e0c1b7854bd7e97f6a0497b2e1b
 * SQRT_AD_MINUS_ONE
 * https://github.com/dalek-cryptography/curve25519-dalek/blob/a62e4a5c573ca9a68503a6fbe47e3f189a4765b0/curve25519-dalek/src/backend/serial/u32/constants.rs#L54-L58
 */
static constexpr f32t::element sqrtadm1_v = {24849947, 33400850, 43495378, 6347714, 46036536,
                                             32887293, 41837720, 18186727, 66238516, 14525638};
} // namespace sxt::f32cn
