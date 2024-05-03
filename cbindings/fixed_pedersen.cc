#include "cbindings/fixed_pedersen.h"

struct sxt_multiexp_handle* sxt_multiexp_handle_new(unsigned curve_id, const void* generators,
                                                unsigned n) {
  (void)curve_id;
  (void)generators;
  (void)n;
  return nullptr;
}

void sxt_multiexp_handle_free(struct sxt_multiexp_handle* handle) {
  (void)handle;
}
