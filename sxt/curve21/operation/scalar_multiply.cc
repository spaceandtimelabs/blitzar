#include "sxt/curve21/operation/scalar_multiply.h"

#include <cassert>
#include <cstring>

#include "sxt/curve21/constant/zero.h"
#include "sxt/curve21/operation/add.h"
#include "sxt/curve21/operation/cmov.h"
#include "sxt/curve21/operation/double.h"
#include "sxt/curve21/type/conversion_utility.h"
#include "sxt/curve21/type/element_cached.h"
#include "sxt/curve21/type/element_p1p1.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/scalar25/base/reduce.h"
#include "sxt/scalar25/type/element.h"

namespace sxt::c21o {
//--------------------------------------------------------------------------------------------------
// fill_exponent
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
static void fill_exponent(s25t::element& a, basct::cspan<uint8_t> data) noexcept {
  std::memcpy(a.data(), data.data(), data.size());

  if (a.data()[31] > 127) {
    s25b::reduce32(a); // a_i = a_i % (2^252 + 27742317777372353535851937790883648493)
  }
}

//--------------------------------------------------------------------------------------------------
// scalar_multiply255
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
void scalar_multiply255(c21t::element_p3& h, const unsigned char* a,
                        const c21t::element_p3& p) noexcept {
  c21t::element_p1p1 r;
  c21t::element_p2 s;
  c21t::element_p1p1 t2, t3, t4, t5, t6, t7, t8;
  c21t::element_p3 p2, p3, p4, p5, p6, p7, p8;
  c21t::element_cached pi[8];
  c21t::element_cached t;
  int i;

  c21t::to_element_cached(pi[1 - 1], p); /* p */

  double_element(t2, p);
  c21t::to_element_p3(p2, t2);
  c21t::to_element_cached(pi[2 - 1], p2); /* 2p = 2*p */

  add(t3, p, pi[2 - 1]);
  c21t::to_element_p3(p3, t3);
  c21t::to_element_cached(pi[3 - 1], p3); /* 3p = 2p+p */

  double_element(t4, p2);
  c21t::to_element_p3(p4, t4);
  c21t::to_element_cached(pi[4 - 1], p4); /* 4p = 2*2p */

  add(t5, p, pi[4 - 1]);
  c21t::to_element_p3(p5, t5);
  c21t::to_element_cached(pi[5 - 1], p5); /* 5p = 4p+p */

  double_element(t6, p3);
  c21t::to_element_p3(p6, t6);
  c21t::to_element_cached(pi[6 - 1], p6); /* 6p = 2*3p */

  add(t7, p, pi[6 - 1]);
  c21t::to_element_p3(p7, t7);
  c21t::to_element_cached(pi[7 - 1], p7); /* 7p = 6p+p */

  double_element(t8, p4);
  c21t::to_element_p3(p8, t8);
  c21t::to_element_cached(pi[8 - 1], p8); /* 8p = 2*4p */

  signed char e[64];
  for (i = 0; i < 32; ++i) {
    e[2 * i + 0] = (a[i] >> 0) & 15;
    e[2 * i + 1] = (a[i] >> 4) & 15;
  }
  /* each e[i] is between 0 and 15 */
  /* e[63] is between 0 and 7 */

  signed char carry = 0;
  for (i = 0; i < 63; ++i) {
    e[i] += carry;
    carry = e[i] + 8;
    carry >>= 4;
    e[i] -= carry * ((signed char)1 << 4);
  }
  e[63] += carry;
  /* each e[i] is between -8 and 8 */

  h = c21cn::zero_p3_v;

  for (i = 63; i != 0; i--) {
    cmov8(t, pi, e[i]);
    add(r, h, t);

    c21t::to_element_p2(s, r);
    double_element(r, s);
    c21t::to_element_p2(s, r);
    double_element(r, s);
    c21t::to_element_p2(s, r);
    double_element(r, s);
    c21t::to_element_p2(s, r);
    double_element(r, s);

    c21t::to_element_p3(h, r); /* *16 */
  }
  cmov8(t, pi, e[i]);
  add(r, h, t);

  c21t::to_element_p3(h, r);
}

//--------------------------------------------------------------------------------------------------
// scalar_multiply
//--------------------------------------------------------------------------------------------------
/*
 h = a * p
 where a = a[0]+256*a[1]+...+256^31 a[31]
 */
CUDA_CALLABLE
void scalar_multiply(c21t::element_p3& h, basct::cspan<uint8_t> a,
                     const c21t::element_p3& p) noexcept {
  assert(a.size() <= 32);
  s25t::element a_p{};
  fill_exponent(a_p, a);
  scalar_multiply255(h, a_p.data(), p);
}
} // namespace sxt::c21o
