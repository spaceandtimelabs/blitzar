#include "cbindings/fixed_pedersen.h"

#include <memory>

#include "cbindings/backend.h"
#include "sxt/cbindings/base/multiexp_handle.h"
using namespace sxt;

struct sxt_multiexp_handle* sxt_multiexp_handle_new(unsigned curve_id, const void* generators,
                                                unsigned n) {
  auto res = std::make_unique<cbnb::multiexp_handle>();
  res->curve_id = static_cast<cbnb::curve_id_t>(curve_id);
  auto backend = cbn::get_backend();
  res->partition_table_accessor = backend->make_partition_table_accessor(res->curve_id, generators, n);
  return reinterpret_cast<sxt_multiexp_handle*>(res.release());
}

void sxt_multiexp_handle_free(struct sxt_multiexp_handle* handle) {
  (void)handle;
}
