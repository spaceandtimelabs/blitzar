#include "sxt/proof/transcript/transcript_utility.h"

#include <string>

#include "sxt/base/test/unit_test.h"
#include "sxt/ristretto/type/compressed_element.h"

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
    append(trans, val_label, val_span);
    expected_trans.append_message(val_label, bytes_span);
    REQUIRE(trans == expected_trans);
  }

  SECTION("we can pass a string_view label and a T value to append ") {
    transcript trans(input_label);
    transcript expected_trans(input_label);

    // append val to transcript
    append(trans, val_label, val);

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
