#include "sxt/multiexp/bucket_method/reduction.h"

#include <iostream>

#include "sxt/base/curve/example_element.h"
#include "sxt/base/device/synchronization.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/memory/resource/managed_device_resource.h"
using namespace sxt;
using namespace sxt::mtxbk;

using E = bascrv::element97;

TEST_CASE("we can reduce bucket sums") {
  basdv::stream stream;

  memmg::managed_array<E> bucket_sums{memr::get_managed_device_resource()};
  memmg::managed_array<E> reductions(1);

  SECTION("we can reduce 3 bucket sums") {
    bucket_sums = {3, 5, 2};
    reduce_buckets<E>(reductions, stream, bucket_sums, 2, 1);
    basdv::synchronize_stream(stream);
    E expected = 3u * 1 + 5u * 2u + 2u * 3u;
    REQUIRE(reductions[0] == expected);
  }

  SECTION("we can reduce two groups of 3 bucket sums") {
    bucket_sums = {3, 5, 2, 7, 1, 9};
    reduce_buckets<E>(reductions, stream, bucket_sums, 2, 1);
    basdv::synchronize_stream(stream);
    unsigned expected = 3u * 1u + 5u * 2u + 2u * 3u;
    expected += 4u * (7u * 1u + 1u * 2u + 9u * 3u);
    REQUIRE(reductions[0] == expected);
  }

  SECTION("we can reduce multiple outputs") {
    reductions.resize(2);
    bucket_sums = {3, 5, 2, 7, 1, 9};
    reduce_buckets<E>(reductions, stream, bucket_sums, 2, 1);
    basdv::synchronize_stream(stream);
    unsigned expected = 3u * 1u + 5u * 2u + 2u * 3u;
    REQUIRE(reductions[0] == expected);
    expected = 7u * 1u + 1u * 2u + 9u * 3u;
    REQUIRE(reductions[1] == expected);
  }
}
