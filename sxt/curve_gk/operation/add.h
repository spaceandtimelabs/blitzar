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
#include "sxt/curve_gk/operation/mul_by_3b.h"
#include "sxt/curve_gk/type/element_p2.h"
#include "sxt/fieldgk/operation/add.h"
#include "sxt/fieldgk/operation/mul.h"
#include "sxt/fieldgk/operation/sub.h"

namespace sxt::cgkt {
struct element_affine;
}

namespace sxt::cgko {
//--------------------------------------------------------------------------------------------------
// add_inplace
//--------------------------------------------------------------------------------------------------
/**
 * p = p + q
 */
CUDA_CALLABLE inline void add_inplace(cgkt::element_p2& p, const cgkt::element_p2& q) noexcept {
  fgkt::element t0, t1, t2, t3, t4;
  const fgkt::element px{p.X};

  fgko::mul(t0, p.X, q.X);
  fgko::mul(t1, p.Y, q.Y);
  fgko::mul(t2, p.Z, q.Z);
  fgko::add(t3, p.X, p.Y);
  fgko::add(t4, q.X, q.Y);
  fgko::mul(t3, t3, t4);
  fgko::add(t4, t0, t1);
  fgko::sub(t3, t3, t4);
  fgko::add(t4, p.Y, p.Z);
  fgko::add(p.X, q.Y, q.Z);
  fgko::mul(t4, t4, p.X);
  fgko::add(p.X, t1, t2);
  fgko::sub(t4, t4, p.X);
  fgko::add(p.X, px, p.Z);
  fgko::add(p.Y, q.X, q.Z);
  fgko::mul(p.X, p.X, p.Y);
  fgko::add(p.Y, t0, t2);
  fgko::sub(p.Y, p.X, p.Y);
  fgko::add(p.X, t0, t0);
  fgko::add(t0, p.X, t0);
  mul_by_3b(t2, t2);
  fgko::add(p.Z, t1, t2);
  fgko::sub(t1, t1, t2);
  mul_by_3b(p.Y, p.Y);
  fgko::mul(p.X, t4, p.Y);
  fgko::mul(t2, t3, t1);
  fgko::sub(p.X, t2, p.X);
  fgko::mul(p.Y, p.Y, t0);
  fgko::mul(t1, t1, p.Z);
  fgko::add(p.Y, t1, p.Y);
  fgko::mul(t0, t0, t3);
  fgko::mul(p.Z, p.Z, t4);
  fgko::add(p.Z, p.Z, t0);
}

//--------------------------------------------------------------------------------------------------
// add
//--------------------------------------------------------------------------------------------------
/**
 * Algorithm 7, https://eprint.iacr.org/2015/1060.pdf
 */
CUDA_CALLABLE
void inline add(cgkt::element_p2& h, const cgkt::element_p2& p,
                const cgkt::element_p2& q) noexcept {
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
void add(cgkt::element_p2& h, const cgkt::element_p2& p, const cgkt::element_affine& q) noexcept;
} // namespace sxt::cgko
