#pragma once

#include "sxt/base/concept/field.h"
#include "sxt/scalar25/operation/add.h"
#include "sxt/scalar25/operation/mul.h"
#include "sxt/scalar25/operation/muladd.h"
#include "sxt/scalar25/operation/neg.h"
#include "sxt/scalar25/operation/sub.h"
#include "sxt/scalar25/type/element.h"

static_assert(
    sxt::bascpt::field<sxt::s25t::element>);
