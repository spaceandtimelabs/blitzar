#pragma once

#include "sxt/base/macro/cuda_callable.h"
#include "sxt/base/num/fast_random_number_generator.h"
#include "sxt/curve21/ristretto/point_formation.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/field51/random/element.h"
#include "sxt/field51/type/element.h"

namespace sxt::c21rn {
//--------------------------------------------------------------------------------------------------
// generate_random_ristretto_element
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
void generate_random_ristretto_element(
    c21t::element_p3& p, basn::fast_random_number_generator& rng) noexcept {
  f51t::element r0, r1;
  f51rn::generate_random_element(r0, rng);
  f51rn::generate_random_element(r1, rng);
  c21rs::form_ristretto_point(p, r0, r1);
}
} // namespace sxt::c21rn
