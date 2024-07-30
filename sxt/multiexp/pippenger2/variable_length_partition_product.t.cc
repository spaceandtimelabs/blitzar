#include "sxt/multiexp/pippenger2/variable_length_partition_product.h"

#include <random>

#include "sxt/base/curve/example_element.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/managed_device_resource.h"
#include "sxt/multiexp/pippenger2/in_memory_partition_table_accessor.h"

using namespace sxt;
using namespace sxt::mtxpp2;

TEST_CASE("we can compute partition products of variable length") {
  using E = bascrv::element97;
  memmg::managed_array<E> products{8, memr::get_managed_device_resource()};
  std::vector<uint8_t> scalars(1);
  memmg::managed_array<E> expected(8);
  for (auto& e : expected) {
    e = 0u;
  }

  auto partition_table_size = 1u << 16;
  memmg::managed_array<E> partition_table(partition_table_size * 10);
  std::mt19937 rng{0};
  for (unsigned i = 0; i < partition_table.size(); ++i) {
    if (i % (1u << 16u) == 0) {
      partition_table[i] = 0u;
    } else {
      partition_table[i] = std::uniform_int_distribution<unsigned>{0, 96}(rng);
    }
  }
  in_memory_partition_table_accessor accessor{memmg::managed_array<E>{partition_table}, 16};

#if 0
  SECTION("we handle a product with a single scalar") {
    scalars[0] = 1;
    auto fut = async_partition_product<E>(products, accessor, scalars, 0);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    basdv::synchronize_device();
    expected[0] = partition_table[1];
    REQUIRE(products == expected);
  }

  SECTION("we can compute a multiproduct where the number of products is not a multiple of 8") {
    scalars = {1u, 3u};
    products.resize(2);
    auto fut = async_partition_product<E>(products, accessor, scalars, 0);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    basdv::synchronize_device();
    expected = {
        partition_table[3],
        partition_table[2],
    };
    REQUIRE(products == expected);
  }

  SECTION("we handle a product with an offset") {
    scalars[0] = 1;
    auto fut = async_partition_product<E>(products, accessor, scalars, 16);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    basdv::synchronize_device();
    expected[0] = partition_table[partition_table_size + 1];
    REQUIRE(products == expected);
  }

  SECTION("we handle a product with two scalars") {
    scalars = {1u, 3u};
    auto fut = async_partition_product<E>(products, accessor, scalars, 0);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    basdv::synchronize_device();
    expected[0] = partition_table[3];
    expected[1] = partition_table[2];
    REQUIRE(products == expected);
  }

  SECTION("we handle a product with more 16 scalars") {
    scalars.resize(16);
    scalars[0] = 1u;
    scalars[15] = 1u;
    auto fut = async_partition_product<E>(products, accessor, scalars, 0);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    basdv::synchronize_device();
    expected[0] = partition_table[1 + (1u << 15u)].value;
    REQUIRE(products == expected);
  }

  SECTION("we handle a product with more than 16 scalars") {
    scalars.resize(32);
    scalars[0] = 1u;
    scalars[16] = 1u;
    auto fut = async_partition_product<E>(products, accessor, scalars, 0);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    basdv::synchronize_device();
    expected[0] = partition_table[1].value + partition_table[partition_table_size + 1].value;
    REQUIRE(products == expected);
  }

  SECTION("we can compute products on the host") {
    scalars.resize(32);
    scalars[0] = 1u;
    scalars[16] = 1u;
    partition_product<E>(products, accessor, scalars, 0);
    expected[0] = partition_table[1].value + partition_table[partition_table_size + 1].value;
    REQUIRE(products == expected);
  }
}

TEST_CASE("we can compute the product of partitions with different bit widths") {
  using E = bascrv::element97;
  memmg::managed_array<E> products{8, memr::get_managed_device_resource()};
  std::vector<uint8_t> scalars(1);
  memmg::managed_array<E> expected(8);
  for (auto& e : expected) {
    e = 0u;
  }

  memmg::managed_array<E> partition_table((1u << 2) * 10);
  std::mt19937 rng{0};
  for (unsigned i = 0; i < partition_table.size(); ++i) {
    if (i % (1u << 2) == 0) {
      partition_table[i] = 0u;
    } else {
      partition_table[i] = std::uniform_int_distribution<unsigned>{0, 96}(rng);
    }
  }
  in_memory_partition_table_accessor accessor{memmg::managed_array<E>{partition_table}, 2};

  SECTION("we handle a product with a single scalar") {
    scalars[0] = 1;
    auto fut = async_partition_product<E>(products, accessor, scalars, 0);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    basdv::synchronize_device();
    expected[0] = partition_table[1];
    REQUIRE(products == expected);
  }

  SECTION("we can compute a multiproduct where the number of products is not a multiple of 8") {
    scalars = {1u, 3u};
    products.resize(2);
    auto fut = async_partition_product<E>(products, accessor, scalars, 0);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    basdv::synchronize_device();
    expected = {
        partition_table[3],
        partition_table[2],
    };
    REQUIRE(products == expected);
  }
#endif
}
