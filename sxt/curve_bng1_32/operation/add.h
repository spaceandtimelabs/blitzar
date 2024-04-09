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

#include "sxt/base/macro/cuda_callable.h"
#include "sxt/curve_bng1_32/operation/mul_by_3b.h"
#include "sxt/curve_bng1_32/type/element_p2.h"
#include "sxt/field32/operation/add.h"
#include "sxt/field32/operation/mul.h"
#include "sxt/field32/operation/sub.h"

namespace sxt::cn3t {
struct element_affine;
}

namespace sxt::cn3o {
//--------------------------------------------------------------------------------------------------
// add_inplace
//--------------------------------------------------------------------------------------------------
/**
 * p = p + q
 */
CUDA_CALLABLE inline void add_inplace(cn3t::element_p2& p, const cn3t::element_p2& q) noexcept {
  f32t::element t0, t1, t2, t3, t4;
  const f32t::element px{p.X};

  f32o::mul(t0, p.X, q.X);
  f32o::mul(t1, p.Y, q.Y);
  f32o::mul(t2, p.Z, q.Z);
  f32o::add(t3, p.X, p.Y);
  f32o::add(t4, q.X, q.Y);
  f32o::mul(t3, t3, t4);
  f32o::add(t4, t0, t1);
  f32o::sub(t3, t3, t4);
  f32o::add(t4, p.Y, p.Z);
  f32o::add(p.X, q.Y, q.Z);
  f32o::mul(t4, t4, p.X);
  f32o::add(p.X, t1, t2);
  f32o::sub(t4, t4, p.X);
  f32o::add(p.X, px, p.Z);
  f32o::add(p.Y, q.X, q.Z);
  f32o::mul(p.X, p.X, p.Y);
  f32o::add(p.Y, t0, t2);
  f32o::sub(p.Y, p.X, p.Y);
  f32o::add(p.X, t0, t0);
  f32o::add(t0, p.X, t0);
  mul_by_3b(t2, t2);
  f32o::add(p.Z, t1, t2);
  f32o::sub(t1, t1, t2);
  mul_by_3b(p.Y, p.Y);
  f32o::mul(p.X, t4, p.Y);
  f32o::mul(t2, t3, t1);
  f32o::sub(p.X, t2, p.X);
  f32o::mul(p.Y, p.Y, t0);
  f32o::mul(t1, t1, p.Z);
  f32o::add(p.Y, t1, p.Y);
  f32o::mul(t0, t0, t3);
  f32o::mul(p.Z, p.Z, t4);
  f32o::add(p.Z, p.Z, t0);
}

//--------------------------------------------------------------------------------------------------
// add
//--------------------------------------------------------------------------------------------------
/**
 * Algorithm 7, https://eprint.iacr.org/2015/1060.pdf
 */
CUDA_CALLABLE
void inline add(cn3t::element_p2& h, const cn3t::element_p2& p,
                const cn3t::element_p2& q) noexcept {
  h = p;
  add_inplace(h, q);
}

//--------------------------------------------------------------------------------------------------
// add
//--------------------------------------------------------------------------------------------------
/**
 * Algorithm 8, https://eprint.iacr.org/2015/1060.pdf
 */
CUDA_CALLABLE
void add(cn3t::element_p2& h, const cn3t::element_p2& p, const cn3t::element_affine& q) noexcept;
} // namespace sxt::cn3o
