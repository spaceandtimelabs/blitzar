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
 * Adopted from zkcrypto/bls12_381
 *
 * Copyright (c) 2021
 * Sean Bowe <ewillbefull@gmail.com>
 * Jack Grigg <thestr4d@gmail.com>
 *
 * See third_party/license/zkcrypto.LICENSE
 */
#pragma once

#include "sxt/base/container/span.h"

namespace sxt::cg1t {
class compressed_element;
struct element_p2;
} // namespace sxt::cg1t

namespace sxt::cg1o {
//--------------------------------------------------------------------------------------------------
// compress
//--------------------------------------------------------------------------------------------------
/*
 * Serializes a point on the BLS12-381 curve to compressed form using the zkcrypto/bls12_381
 * projects serialization guidance.
 * https://github.com/zkcrypto/bls12_381/blob/4df45188913e9d66ef36ae12825865347eed4e1b/src/notes/serialization.rs
 */

void compress(cg1t::compressed_element& e_c, const cg1t::element_p2& e_p) noexcept;

//--------------------------------------------------------------------------------------------------
// batch_compress
//--------------------------------------------------------------------------------------------------
void batch_compress(basct::span<cg1t::compressed_element> ex_c,
                    basct::cspan<cg1t::element_p2> ex_p) noexcept;
} // namespace sxt::cg1o
