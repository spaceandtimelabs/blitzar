#pragma once

#include <type_traits>

#include "sxt/scalar25/type/element.h"
#include "sxt/base/error/panic.h"
#include "sxt/cbindings/base/field_id.h"

namespace sxt::cbnb {
//--------------------------------------------------------------------------------------------------
// switch_field_type
//--------------------------------------------------------------------------------------------------
template <class F> void switch_field_type(field_id_t id, F f) {
  switch (id) {
  case field_id_t::scalar25519:
    f(std::type_identity<s25t::element>{});
    break;
  default:
    baser::panic("unsupported field id {}", static_cast<unsigned>(id));
  }
}
} // namespace sxt::cbnb
