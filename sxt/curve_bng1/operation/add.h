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
#include "sxt/curve_bng1/operation/mul_by_3b.h"
#include "sxt/curve_bng1/type/element_p2.h"
#include "sxt/field25/operation/add.h"
#include "sxt/field25/operation/mul.h"
#include "sxt/field25/operation/sub.h"

namespace sxt::cn1t {
struct element_affine;
}

namespace sxt::cn1o {
//--------------------------------------------------------------------------------------------------
// add_inplace
//--------------------------------------------------------------------------------------------------
/**
 * p = p + q
 */
CUDA_CALLABLE inline void add_inplace(cn1t::element_p2& p, const cn1t::element_p2& q) noexcept {
  f25t::element t0, t1, t2, t3, t4;
  const f25t::element px{p.X};

  f25o::mul(t0, p.X, q.X);
  f25o::mul(t1, p.Y, q.Y);
  f25o::mul(t2, p.Z, q.Z);
  f25o::add(t3, p.X, p.Y);
  f25o::add(t4, q.X, q.Y);
  f25o::mul(t3, t3, t4);
  f25o::add(t4, t0, t1);
  f25o::sub(t3, t3, t4);
  f25o::add(t4, p.Y, p.Z);
  f25o::add(p.X, q.Y, q.Z);
  f25o::mul(t4, t4, p.X);
  f25o::add(p.X, t1, t2);
  f25o::sub(t4, t4, p.X);
  f25o::add(p.X, px, p.Z);
  f25o::add(p.Y, q.X, q.Z);
  f25o::mul(p.X, p.X, p.Y);
  f25o::add(p.Y, t0, t2);
  f25o::sub(p.Y, p.X, p.Y);
  f25o::add(p.X, t0, t0);
  f25o::add(t0, p.X, t0);
  mul_by_3b(t2, t2);
  f25o::add(p.Z, t1, t2);
  f25o::sub(t1, t1, t2);
  mul_by_3b(p.Y, p.Y);
  f25o::mul(p.X, t4, p.Y);
  f25o::mul(t2, t3, t1);
  f25o::sub(p.X, t2, p.X);
  f25o::mul(p.Y, p.Y, t0);
  f25o::mul(t1, t1, p.Z);
  f25o::add(p.Y, t1, p.Y);
  f25o::mul(t0, t0, t3);
  f25o::mul(p.Z, p.Z, t4);
  f25o::add(p.Z, p.Z, t0);
}

//--------------------------------------------------------------------------------------------------
// add
//--------------------------------------------------------------------------------------------------
/**
 * Algorithm 7, https://eprint.iacr.org/2015/1060.pdf
 */
CUDA_CALLABLE
void inline add(cn1t::element_p2& h, const cn1t::element_p2& p,
                const cn1t::element_p2& q) noexcept {
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
void add(cn1t::element_p2& h, const cn1t::element_p2& p, const cn1t::element_affine& q) noexcept;
} // namespace sxt::cn1o
