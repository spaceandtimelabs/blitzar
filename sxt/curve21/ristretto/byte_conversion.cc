/**
 * Adopted from libsodium
 *
 * Copyright (c) 2013-2022
 * Frank Denis <j at pureftpd dot org>
 *
 * See third_party/license/libsodium.LICENSE
 */

#include "sxt/curve21/ristretto/byte_conversion.h"
#include "sxt/curve21/ristretto/sqrt_ratio_m1.h"

#include "sxt/curve21/type/element_p3.h"

#include "sxt/field51/type/element.h"

#include "sxt/field51/operation/add.h"
#include "sxt/field51/operation/sub.h"
#include "sxt/field51/operation/mul.h"
#include "sxt/field51/operation/square.h"
#include "sxt/field51/operation/cmov.h"
#include "sxt/field51/operation/cneg.h"
#include "sxt/field51/operation/abs.h"
#include "sxt/field51/operation/pow22523.h"

#include "sxt/field51/base/byte_conversion.h"

#include "sxt/field51/constant/one.h"
#include "sxt/field51/constant/sqrtm1.h"
#include "sxt/field51/constant/invsqrtamd.h"

#include "sxt/field51/property/sign.h"

namespace sxt::c21rs {

//--------------------------------------------------------------------------------------------------
// to_bytes
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
void to_bytes(uint8_t s[32], const c21t::element_p3& p) noexcept {
    f51t::element den1, den2;
    f51t::element den_inv;
    f51t::element eden;
    f51t::element inv_sqrt;
    f51t::element ix, iy;
    f51t::element one{f51cn::one_v};
    f51t::element s_;
    f51t::element t_z_inv;
    f51t::element u1, u2;
    f51t::element u1_u2u2;
    f51t::element x_, y_;
    f51t::element x_z_inv;
    f51t::element z_inv;
    f51t::element zmy;
    int     rotate;

    f51o::add(u1, p.Z, p.Y);       /* u1 = Z+Y */
    f51o::sub(zmy, p.Z, p.Y);      /* zmy = Z-Y */
    f51o::mul(u1, u1, zmy);          /* u1 = (Z+Y)*(Z-Y) */
    f51o::mul(u2, p.X, p.Y);       /* u2 = X*Y */

    f51o::square(u1_u2u2, u2);           /* u1_u2u2 = u2^2 */
    f51o::mul(u1_u2u2, u1, u1_u2u2); /* u1_u2u2 = u1*u2^2 */

    c21rs::compute_sqrt_ratio_m1(inv_sqrt, one, u1_u2u2);

    f51o::mul(den1, inv_sqrt, u1);   /* den1 = inv_sqrt*u1 */
    f51o::mul(den2, inv_sqrt, u2);   /* den2 = inv_sqrt*u2 */
    f51o::mul(z_inv, den1, den2);    /* z_inv = den1*den2 */
    f51o::mul(z_inv, z_inv, p.T);   /* z_inv = den1*den2*T */

    f51o::mul(ix, p.X, f51t::element{f51cn::sqrtm1_v});       /* ix = X*sqrt(-1) */
    f51o::mul(iy, p.Y, f51t::element{f51cn::sqrtm1_v});       /* iy = Y*sqrt(-1) */
    f51o::mul(eden, den1, f51t::element{f51cn::invsqrtamd}); /* eden = den1/sqrt(a-d) */

    f51o::mul(t_z_inv, p.T, z_inv); /* t_z_inv = T*z_inv */
    rotate = f51p::is_negative(t_z_inv);

    x_ = p.X;
    y_ = p.Y;
    den_inv = den2;

    f51o::cmov(x_, iy, rotate);
    f51o::cmov(y_, ix, rotate);
    f51o::cmov(den_inv, eden, rotate);

    f51o::mul(x_z_inv, x_, z_inv);
    f51o::cneg(y_, y_, f51p::is_negative(x_z_inv));

    f51o::sub(s_, p.Z, y_);
    f51o::mul(s_, den_inv, s_);
    f51o::abs(s_, s_);

    f51b::to_bytes(s, s_.data());
}

}  // namespace sxt::c21rs
