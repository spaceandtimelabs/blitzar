#include "sxt/proof/transcript/strobe128.h"

#include <array>

#include "sxt/base/test/unit_test.h"

using namespace sxt::prft;

TEST_CASE("conformance test protocol") {
  strobe128 s1 = strobe128("Conformance Test Protocol");

  std::array<uint8_t, 1024> msg;
  memset(msg.data(), 99u, msg.size());

  {
    uint8_t label_data[] = "ms";
    s1.meta_ad({label_data, sizeof(label_data) - 1}, false);
  }

  {
    uint8_t label_data[] = "g";
    s1.meta_ad({label_data, sizeof(label_data) - 1}, true);
    s1.ad(msg, false);
  }

  std::array<uint8_t, 32> prf1;

  {
    uint8_t label_data[] = "prf";
    s1.meta_ad({label_data, sizeof(label_data) - 1}, false);
    s1.prf(prf1, false);
  }

  std::array<uint8_t, 32> expected_prf1 = {180, 142, 100, 92, 161, 124, 102, 127, 213, 32, 107,
                                           165, 122, 106, 34, 141, 114, 216, 225, 144, 56, 20,
                                           211, 241, 127, 98, 41,  150, 215, 207, 239, 176};

  REQUIRE(prf1 == expected_prf1);

  {
    uint8_t label_data[] = "key";
    s1.meta_ad({label_data, sizeof(label_data) - 1}, false);
    s1.key(prf1, false);
  }

  std::array<uint8_t, 32> prf2;

  {
    uint8_t label_data[] = "prf";
    s1.meta_ad({label_data, sizeof(label_data) - 1}, false);
    s1.prf(prf2, false);
  }

  std::array<uint8_t, 32> expected_prf2 = {
      7u,  228u, 92u,  206u, 128u, 120u, 206u, 226u, 89u, 227u, 227u, 117u, 187u, 133u, 215u, 86u,
      16u, 226u, 209u, 225u, 32u,  28u,  95,   100u, 80u, 69u,  161u, 148u, 237u, 212u, 159u, 248u};

  REQUIRE(prf2 == expected_prf2);
}
