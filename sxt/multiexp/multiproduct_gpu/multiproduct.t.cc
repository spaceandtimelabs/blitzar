#include "sxt/multiexp/multiproduct_gpu/multiproduct.h"

#include <numeric>

#include "sxt/algorithm/reduction/test_reducer.h"
#include "sxt/base/container/blob_array.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/managed_device_resource.h"
#include "sxt/multiexp/index/index_table.h"

using namespace sxt;
using namespace sxt::mtxmpg;

TEST_CASE("we can compute multiproducts using the GPU") {
  constexpr uint64_t ignore_v = 0;

  size_t max_generators = 10'000;
  memmg::managed_array<uint64_t> generators{max_generators, memr::get_managed_device_resource()};
  std::iota(generators.begin(), generators.end(), 1);

  SECTION("we can compute a single product with a single element") {
    mtxi::index_table products = {{ignore_v, ignore_v, 0}};
    basct::blob_array masks(1, 1);
    masks[0][0] = 1;
    auto res = compute_multiproduct<algr::test_add_reducer>({generators.data(), masks.size()},
                                                            products.cheader(), masks, 1)
                   .await_result()
                   .as_array<uint64_t>();
    memmg::managed_array<uint64_t> expected = {1};
    REQUIRE(res == expected);
  }

  SECTION("we can compute a single product with two terms") {
    mtxi::index_table products = {{ignore_v, ignore_v, 0, 1}};
    basct::blob_array masks(3, 1);
    masks[0][0] = 1;
    masks[1][0] = 0;
    masks[2][0] = 1;
    auto res = compute_multiproduct<algr::test_add_reducer>({generators.data(), masks.size()},
                                                            products.cheader(), masks, 2)
                   .await_result()
                   .as_array<uint64_t>();
    memmg::managed_array<uint64_t> expected = {
        generators[0] + generators[2],
    };
    REQUIRE(res == expected);
  }

  SECTION("we can compute multiple products") {
    mtxi::index_table products = {
        {ignore_v, ignore_v, 1},
        {ignore_v, ignore_v, 0, 2},
        {ignore_v, ignore_v, 0, 1, 2},
    };
    basct::blob_array masks(3, 1);
    masks[0][0] = 1;
    masks[1][0] = 1;
    masks[2][0] = 1;
    auto res = compute_multiproduct<algr::test_add_reducer>({generators.data(), masks.size()},
                                                            products.cheader(), masks, 3)
                   .await_result()
                   .as_array<uint64_t>();
    memmg::managed_array<uint64_t> expected = {
        generators[1],
        generators[0] + generators[2],
        generators[0] + generators[1] + generators[2],
    };
    REQUIRE(res == expected);
  }

  SECTION("we can compute products with many terms") {
    size_t n = 1'000;
    mtxi::index_table products{1, n + 2};
    auto& row = products.header()[0];
    row = {products.entry_data(), n + 2};
    row[0] = ignore_v;
    row[1] = ignore_v;
    auto rest = row.subspan(2);
    std::iota(rest.begin(), rest.end(), 0);
    basct::blob_array masks(n, 1);
    for (size_t i = 0; i < n; ++i) {
      masks[i][0] = 1;
    }
    auto res = compute_multiproduct<algr::test_add_reducer>({generators.data(), masks.size()},
                                                            products.cheader(), masks, n)
                   .await_result()
                   .as_array<uint64_t>();
    memmg::managed_array<uint64_t> expected = {
        n * (n + 1) / 2,
    };
    REQUIRE(res == expected);
  }
}
