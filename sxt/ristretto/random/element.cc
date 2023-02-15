#include "sxt/ristretto/random/element.h"

#include "sxt/base/macro/cuda_warning.h"
#include "sxt/base/num/fast_random_number_generator.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/field51/random/element.h"
#include "sxt/field51/type/element.h"
#include "sxt/ristretto/base/byte_conversion.h"
#include "sxt/ristretto/base/point_formation.h"
#include "sxt/ristretto/type/compressed_element.h"

namespace sxt::rstrn {
//--------------------------------------------------------------------------------------------------
// generate_random_element_impl
//--------------------------------------------------------------------------------------------------
CUDA_DISABLE_HOSTDEV_WARNING
template <class Rng>
static CUDA_CALLABLE void generate_random_element_impl(c21t::element_p3& p, Rng& rng) noexcept {
  f51t::element r0, r1;
  f51rn::generate_random_element(r0, rng);
  f51rn::generate_random_element(r1, rng);
  rstb::form_ristretto_point(p, r0, r1);
}

//--------------------------------------------------------------------------------------------------
// generate_random_elements_impl
//--------------------------------------------------------------------------------------------------
template <class Rng>
static void generate_random_elements_impl(basct::span<c21t::element_p3> px, Rng& rng) noexcept {
  for (auto& pi : px) {
    generate_random_element(pi, rng);
  }
}

//--------------------------------------------------------------------------------------------------
// generate_random_element
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
void generate_random_element(c21t::element_p3& p,
                             basn::fast_random_number_generator& rng) noexcept {
  generate_random_element_impl(p, rng);
}

void generate_random_element(c21t::element_p3& p, std::mt19937& rng) noexcept {
  generate_random_element_impl(p, rng);
}

void generate_random_element(rstt::compressed_element& p,
                             basn::fast_random_number_generator& rng) noexcept {
  c21t::element_p3 pp;
  generate_random_element(pp, rng);
  rstb::to_bytes(p.data(), pp);
}

//--------------------------------------------------------------------------------------------------
// generate_random_elements
//--------------------------------------------------------------------------------------------------
void generate_random_elements(basct::span<c21t::element_p3> px,
                              basn::fast_random_number_generator& rng) noexcept {
  generate_random_elements_impl(px, rng);
}

void generate_random_elements(basct::span<c21t::element_p3> px, std::mt19937& rng) noexcept {
  generate_random_elements_impl(px, rng);
}

void generate_random_elements(basct::span<rstt::compressed_element> px,
                              basn::fast_random_number_generator& rng) noexcept {
  for (auto& pi : px) {
    generate_random_element(pi, rng);
  }
}
} // namespace sxt::rstrn
