#include "sxt/multiexp/multiproduct_gpu/multiproduct.h"

#include <numeric>

#include "sxt/algorithm/reduction/test_reducer.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/execution/schedule/scheduler.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/managed_device_resource.h"

using namespace sxt;
using namespace sxt::mtxmpg;

TEST_CASE("we can compute multiproducts using the GPU") {
  size_t max_generators = 10'000;
  memmg::managed_array<uint64_t> generators{max_generators, memr::get_managed_device_resource()};
  std::iota(generators.begin(), generators.end(), 1);

  SECTION("we can compute a single product with a single element") {
    memmg::managed_array<unsigned> indexes = {0};
    memmg::managed_array<unsigned> product_sizes = {1};
    auto res = compute_multiproduct<algr::test_add_reducer>({generators.data(), 1},
                                                            std::move(indexes), product_sizes);
    xens::get_scheduler().run();
    memmg::managed_array<uint64_t> expected = {1};
    REQUIRE(res.value() == expected);
  }

  SECTION("we can compute a single product with two terms") {
    memmg::managed_array<unsigned> indexes = {0, 2};
    memmg::managed_array<unsigned> product_sizes = {2};
    auto res = compute_multiproduct<algr::test_add_reducer>({generators.data(), 1},
                                                            std::move(indexes), product_sizes);
    xens::get_scheduler().run();
    memmg::managed_array<uint64_t> expected = {
        generators[0] + generators[2],
    };
    REQUIRE(res.value() == expected);
  }

  SECTION("we can compute multiple products") {
    memmg::managed_array<unsigned> indexes = {1, 0, 2, 0, 1, 2};
    memmg::managed_array<unsigned> product_sizes = {1, 2, 3};
    auto res = compute_multiproduct<algr::test_add_reducer>({generators.data(), 1},
                                                            std::move(indexes), product_sizes);
    xens::get_scheduler().run();
    memmg::managed_array<uint64_t> expected = {
        generators[1],
        generators[0] + generators[2],
        generators[0] + generators[1] + generators[2],
    };
    REQUIRE(res.value() == expected);
  }

  SECTION("we can compute products with many terms") {
    unsigned n = 1'000;
    memmg::managed_array<unsigned> indexes(n);
    std::iota(indexes.begin(), indexes.end(), 0u);
    memmg::managed_array<unsigned> product_sizes = {n};
    auto res = compute_multiproduct<algr::test_add_reducer>({generators.data(), 1},
                                                            std::move(indexes), product_sizes);
    xens::get_scheduler().run();
    memmg::managed_array<uint64_t> expected = {
        n * (n + 1) / 2,
    };
    REQUIRE(res.value() == expected);
  }
}
