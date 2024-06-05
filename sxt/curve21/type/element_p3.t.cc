#include "sxt/curve21/type/element_p3.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/field51/type/literal.h"
using namespace sxt;
using namespace sxt::c21t;
using sxt::f51t::operator""_f51;

TEST_CASE("todo") {
  c21t::element_p3 e{0x3b6f8891960f6ad45776d1e1213c1bd9de44f888163a76921515e6cf9f3fd67e_f51,
                     0x336d9ece4cdb30925921f40f14dab827d6e156675107378db6d34c9a874a007e_f51,
                     0x59e4ea1a52a20ea2fd9cb81712f675b450b27bff31b598ba722d5b0bf61c8608_f51,
                     0x1f6e08da2d298daafc6ea6fedd5e07c172749500483d139bc532c7e392cad989_f51};
  ;
#if 0
  res = {.X = 0x3b6f8891960f6ad45776d1e1213c1bd9de44f888163a76921515e6cf9f3fd67e_f51,
         .Y = 0x336d9ece4cdb30925921f40f14dab827d6e156675107378db6d34c9a874a007e_f51,
         .Z = 0x59e4ea1a52a20ea2fd9cb81712f675b450b27bff31b598ba722d5b0bf61c8608_f51,
         .T = 0x1f6e08da2d298daafc6ea6fedd5e07c172749500483d139bc532c7e392cad989_f51};
#endif
}
