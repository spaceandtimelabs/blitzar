#include "sxt/execution/async/future_value_storage.h"

#include <memory>

#include "sxt/base/test/unit_test.h"

using namespace sxt;
using namespace sxt::xena;

TEST_CASE("future_value_storage manages the storage of a xena::future") {
  SECTION("void is supported") {
    future_value_storage<void> val;
    val.consume_value();
  }

  SECTION("we can manage PODs") {
    future_value_storage<int> val{123};
    REQUIRE(val.consume_value() == 123);
  }

  SECTION("we can manage move-only types") {
    future_value_storage<std::unique_ptr<int>> val{std::make_unique<int>(123)};
    REQUIRE(*val.consume_value() == 123);
  }
}
