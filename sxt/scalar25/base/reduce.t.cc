#include "sxt/scalar25/base/reduce.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/scalar25/type/element.h"
#include "sxt/scalar25/type/literal.h"

using namespace sxt::s25b;
using namespace sxt::s25t;

TEST_CASE("we correctly reduces arrays with 32 bytes") {
  SECTION("we do not reduce A when A < L (L = the field order)") {
    element s = 0x1000000000000000000000000000000014def9dea2f79cd65812631a5cf5d3ec_s25;
    element expected_s = s;
    reduce32(s);
    REQUIRE(s == expected_s);
  }

  SECTION("we correctly reduce A when A = L (L = the field order)") {
    element s = 0x1000000000000000000000000000000014def9dea2f79cd65812631a5cf5d3ed_s25;
    reduce32(s);
    REQUIRE(s == 0x0_s25);
  }

  SECTION("we correctly reduce A when A = L + 103 (L = the field order)") {
    element s = 0x1000000000000000000000000000000014def9dea2f79cd65812631a5cf5d454_s25;
    reduce32(s);
    REQUIRE(s == 0x67_s25);
  }

  SECTION("we correctly reduce A when A is the biggest 256bits integer") {
    element s = 0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff_s25;
    reduce32(s);
    REQUIRE(s == 0xffffffffffffffffffffffffffffffec6ef5bf4737dcf70d6ec31748d98951c_s25);
  }
}

TEST_CASE("we correctly reduces arrays with 33 bytes") {
  SECTION("we do not reduce A when A < L (L = the field order)") {
    element s = 0x1000000000000000000000000000000014def9dea2f79cd65812631a5cf5d3ec_s25;
    element expected_s = s;
    reduce33(s, 0);
    REQUIRE(s == expected_s);
  }

  SECTION("we correctly reduce A when A = L + 103 (L = the field order)") {
    element s = 0x1000000000000000000000000000000014def9dea2f79cd65812631a5cf5d454_s25;
    reduce33(s, 0);
    REQUIRE(s == 0x67_s25);
  }

  SECTION("we correctly reduce A when A is the biggest 264bits integer (33 full bytes)") {
    element s = 0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff_s25;
    reduce33(s, static_cast<uint8_t>(0xff));
    REQUIRE(s == 0xffffffffffffffffffffffffffffeb225410faf292a375531e0bd4affb703ec_s25);
  }
}

TEST_CASE("we correctly reduces arrays with 64 bytes") {
  SECTION("we do not reduce A when A < L (L = the field order)") {
    element s = 0x1000000000000000000000000000000014def9dea2f79cd65812631a5cf5d3ec_s25;
    element expected_s = s;
    uint8_t s_data[64] = {};
    std::memcpy(s_data, s.data(), 32); // we copy the 32 bytes to the begginning of s_data
    reduce64(s, s_data);
    REQUIRE(s == expected_s);
  }

  SECTION("we correctly reduce A when A = L (L = the field order)") {
    element s = 0x1000000000000000000000000000000014def9dea2f79cd65812631a5cf5d3ed_s25;
    uint8_t s_data[64] = {};
    std::memcpy(s_data, s.data(), 32); // we copy the 32 bytes to the begginning of s_data
    reduce64(s, s_data);
    REQUIRE(s == 0x0_s25);
  }

  SECTION("we correctly reduce A when A = L + 103 (L = the field order)") {
    element s = 0x1000000000000000000000000000000014def9dea2f79cd65812631a5cf5d454_s25;
    uint8_t s_data[64] = {};
    std::memcpy(s_data, s.data(), 32); // we copy the 32 bytes to the begginning of s_data
    reduce64(s, s_data);
    REQUIRE(s == 0x67_s25);
  }

  SECTION("we correctly reduce A when A is the biggest 256bits integer") {
    element s = 0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff_s25;
    uint8_t s_data[64] = {};
    std::memcpy(s_data, s.data(), 32); // we copy the 32 bytes to the begginning of s_data
    reduce64(s, s_data);
    REQUIRE(s == 0xffffffffffffffffffffffffffffffec6ef5bf4737dcf70d6ec31748d98951c_s25);
  }

  SECTION("we correctly reduce A when A is the biggest 512bits integer") {
    element s = 0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff_s25;
    uint8_t s_data[64] = {};
    std::memcpy(s_data, s.data(), 32);      // we copy the 32 bytes to the begginning of s_data
    std::memcpy(s_data + 32, s.data(), 32); // we copy the 32 bytes to the end of s_data
    reduce64(s, s_data);
    REQUIRE(s == 0x399411b7c309a3dceec73d217f5be65d00e1ba768859347a40611e3449c0f00_s25);
  }
}
