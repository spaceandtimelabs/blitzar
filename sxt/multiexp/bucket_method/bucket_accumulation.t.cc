#include "sxt/multiexp/bucket_method/bucket_accumulation.h"

#include "sxt/base/curve/example_element.h"
#include "sxt/base/device/synchronization.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/execution/schedule/scheduler.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/managed_device_resource.h"
using namespace sxt;
using namespace sxt::mtxbk;

TEST_CASE("we can perform a bucket accumulation pass") {
  using E = bascrv::element97;
  memmg::managed_array<E> bucket_sums(255 * 32);

  SECTION("we handle the empty case") {
    bucket_sums.reset();
    auto fut = accumulate_buckets<E>(bucket_sums, {}, {});    
    REQUIRE(fut.ready());
  }

  SECTION("we handle a case with a single zero element") {
    uint8_t scalar[32] = {};
    const uint8_t* scalars[] = {
      scalar
    };
    E generators[] = {7};
    auto fut = accumulate_buckets<E>(bucket_sums, generators, scalars);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    for (auto val : bucket_sums) {
      REQUIRE(val == 0);
    }
  }

  SECTION("we handle a case with a single element of 1") {
    uint8_t scalar[32] = {};
    scalar[0] = 1;
    const uint8_t* scalars[] = {
      scalar
    };
    E generators[] = {7};
    auto fut = accumulate_buckets<E>(bucket_sums, generators, scalars);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    for (size_t i=0; i<bucket_sums.size(); ++i) {
      auto val = bucket_sums[i];
      if (i == 0) {
        REQUIRE(val == 7);
      } else {
        REQUIRE(val == 0);
      }
    }
  }

  SECTION("we handle two scalars of different values") {
    uint8_t scalar_data[64] = {};
    scalar_data[0] = 1;
    scalar_data[32] = 2;
    const uint8_t* scalars[] = {
      scalar_data
    };
    E generators[] = {7, 5};
    auto fut = accumulate_buckets<E>(bucket_sums, generators, scalars);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    for (size_t i=0; i<bucket_sums.size(); ++i) {
      auto val = bucket_sums[i];
      if (i == 0) {
        REQUIRE(val == 7);
      } else if (i == 1) {
        REQUIRE(val == 5);
      } else {
        REQUIRE(val == 0);
      }
    }
  }

  SECTION("we handle two scalars of the same value") {
    uint8_t scalar_data[64] = {};
    scalar_data[0] = 2;
    scalar_data[32] = 2;
    const uint8_t* scalars[] = {
      scalar_data
    };
    E generators[] = {7, 5};
    auto fut = accumulate_buckets<E>(bucket_sums, generators, scalars);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    for (size_t i=0; i<bucket_sums.size(); ++i) {
      auto val = bucket_sums[i];
      if (i == 1) {
        REQUIRE(val == 12);
      } else {
        REQUIRE(val == 0);
      }
    }
  }

  SECTION("we handle multiple outputs") {
    bucket_sums = memmg::managed_array<E>(255 * 32 * 2);
    uint8_t scalar_data1[32] = {};
    scalar_data1[0] = 2;
    uint8_t scalar_data2[32] = {};
    scalar_data2[0] = 2;
    const uint8_t* scalars[] = {
      scalar_data1,
      scalar_data2,
    };
    E generators[] = {7};
    auto fut = accumulate_buckets<E>(bucket_sums, generators, scalars);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    for (size_t i=0; i<bucket_sums.size(); ++i) {
      auto val = bucket_sums[i];
      if (i == 1 || i == 255 * 32 + 1) {
        REQUIRE(val == 7);
      } else {
        REQUIRE(val == 0);
      }
    }
  }
}

TEST_CASE("we can perform a bucket accumulation pass 2") {
  using E = bascrv::element97;
  memmg::managed_array<E> bucket_sums{255 * 32, memr::get_managed_device_resource()};

  SECTION("we handle the empty case") {
    bucket_sums.reset();
    auto fut = accumulate_buckets2<E>(bucket_sums, {}, {});    
    REQUIRE(fut.ready());
  }

  SECTION("we handle a case with a single zero element") {
    uint8_t scalar[32] = {};
    const uint8_t* scalars[] = {
      scalar
    };
    E generators[] = {7};
    auto fut = accumulate_buckets<E>(bucket_sums, generators, scalars);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    basdv::synchronize_device();
    for (auto val : bucket_sums) {
      REQUIRE(val == 0);
    }
  }

  SECTION("we handle a case with a single element of 1") {
    uint8_t scalar[32] = {};
    scalar[0] = 1;
    const uint8_t* scalars[] = {
      scalar
    };
    E generators[] = {7};
    auto fut = accumulate_buckets2<E>(bucket_sums, generators, scalars);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    basdv::synchronize_device();
    for (size_t i=0; i<bucket_sums.size(); ++i) {
      auto val = bucket_sums[i];
      if (i == 0) {
        REQUIRE(val == 7);
      } else {
        REQUIRE(val == 0);
      }
    }
  }

  SECTION("we handle two scalars of different values") {
    uint8_t scalar_data[64] = {};
    scalar_data[0] = 1;
    scalar_data[32] = 2;
    const uint8_t* scalars[] = {
      scalar_data
    };
    E generators[] = {7, 5};
    auto fut = accumulate_buckets2<E>(bucket_sums, generators, scalars);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    basdv::synchronize_device();
    for (size_t i=0; i<bucket_sums.size(); ++i) {
      auto val = bucket_sums[i];
      if (i == 0) {
        REQUIRE(val == 7);
      } else if (i == 1) {
        REQUIRE(val == 5);
      } else {
        REQUIRE(val == 0);
      }
    }
  }

  SECTION("we handle two scalars of the same value") {
    uint8_t scalar_data[64] = {};
    scalar_data[0] = 2;
    scalar_data[32] = 2;
    const uint8_t* scalars[] = {
      scalar_data
    };
    E generators[] = {7, 5};
    auto fut = accumulate_buckets2<E>(bucket_sums, generators, scalars);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    basdv::synchronize_device();
    for (size_t i=0; i<bucket_sums.size(); ++i) {
      auto val = bucket_sums[i];
      if (i == 1) {
        REQUIRE(val == 12);
      } else {
        REQUIRE(val == 0);
      }
    }
  }

  SECTION("we handle multiple outputs") {
    bucket_sums = memmg::managed_array<E>(255 * 32 * 2);
    uint8_t scalar_data1[32] = {};
    scalar_data1[0] = 2;
    uint8_t scalar_data2[32] = {};
    scalar_data2[0] = 2;
    const uint8_t* scalars[] = {
      scalar_data1,
      scalar_data2,
    };
    E generators[] = {7};
    auto fut = accumulate_buckets2<E>(bucket_sums, generators, scalars);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    basdv::synchronize_device();
    for (size_t i=0; i<bucket_sums.size(); ++i) {
      auto val = bucket_sums[i];
      if (i == 1 || i == 255 * 32 + 1) {
        REQUIRE(val == 7);
      } else {
        REQUIRE(val == 0);
      }
    }
  }
}
