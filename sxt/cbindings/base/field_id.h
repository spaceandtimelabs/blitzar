#pragma once

namespace sxt::cbnb {
//--------------------------------------------------------------------------------------------------
// field_id_t
//--------------------------------------------------------------------------------------------------
/**
 * Ids for the various fields we support.
 *
 * Note: The values should match those in blitzar_api.h.
 */
enum class field_id_t : unsigned {
  scalar25519 = 0,
};
} // namespace sxt::cbnb
