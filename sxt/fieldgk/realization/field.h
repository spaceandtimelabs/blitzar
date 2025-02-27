#pragma once

#include "sxt/base/field/element.h"
#include "sxt/fieldgk/operation/add.h"
#include "sxt/fieldgk/operation/mul.h"
#include "sxt/fieldgk/operation/muladd.h"
#include "sxt/fieldgk/operation/neg.h"
#include "sxt/fieldgk/operation/sub.h"
#include "sxt/fieldgk/type/element.h"

static_assert(sxt::basfld::element<sxt::fgkt::element>);
