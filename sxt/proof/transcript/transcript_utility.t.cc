#include "sxt/proof/transcript/transcript_utility.h"

#include <cstring>
#include <string>

#include "sxt/base/test/unit_test.h"
#include "sxt/ristretto/type/compressed_element.h"
#include "sxt/scalar25/operation/reduce.h"
#include "sxt/scalar25/type/element.h"

using namespace sxt;
using namespace sxt::prft;

template <class T> static void test_append(T val) {
  char val_label[] = "test1";
  char input_label[] = "input_test";

  SECTION("we can pass a string_view label and a cspan<uint8_t> value to append") {
    transcript trans(input_label);
    transcript expected_trans(input_label);

    T val_array[2] = {val, val};
    sxt::basct::cspan<T> val_span{val_array, 2};
    sxt::basct::cspan<uint8_t> bytes_span{reinterpret_cast<const uint8_t*>(val_array),
                                          sizeof(val_array)};
    append_values(trans, val_label, val_span);
    expected_trans.append_message(val_label, bytes_span);
    REQUIRE(trans == expected_trans);
  }

  SECTION("we can pass a string_view label and a T value to append ") {
    transcript trans(input_label);
    transcript expected_trans(input_label);

    // append val to transcript
    append_value(trans, val_label, val);

    // append the same val above to expected_transcript
    expected_trans.append_message(val_label, {reinterpret_cast<const uint8_t*>(&val), sizeof(val)});

    REQUIRE(trans == expected_trans);
  }
}

TEST_CASE("We can append integer values to the transcript") {
  test_append(1234);
  test_append(123u);
  test_append(123ULL);
}

TEST_CASE("We can append a compressed_ristretto point to the transcript") {
  test_append(sxt::rstt::compressed_element{123u});
}

TEST_CASE("we can get challenge values from a transcript") {
  transcript trans{"abc"};

  SECTION("challenge values aren't equal") {
    s25t::element x1, x2;
    challenge_value(x1, trans, "xyz");
    challenge_value(x2, trans, "123");
    REQUIRE(x1 != x2);
  }

  SECTION("challenge values are reduced") {
    s25t::element x;
    challenge_value(x, trans, "xyz");
    auto xp = x;
    s25o::reduce32(xp);
    REQUIRE(std::memcmp(xp.data(), x.data(), sizeof(s25t::element)) == 0);
  }

  SECTION("we can challenge an array of values") {
    s25t::element xx[2];
    challenge_values(xx, trans, "123");
    REQUIRE(xx[0] != xx[1]);
    for (const auto& xi : xx) {
      auto xip = xi;
      s25o::reduce32(xip);
      REQUIRE(std::memcmp(xip.data(), xi.data(), sizeof(s25t::element)) == 0);
    }
  }
}
