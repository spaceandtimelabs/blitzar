#include "sxt/algorithm/reduction/reduction.h"

#include <random>

#include "sxt/algorithm/base/identity_mapper.h"
#include "sxt/algorithm/reduction/test_reducer.h"
#include "sxt/base/device/memory_utility.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/device_resource.h"

using namespace sxt;
using namespace sxt::algr;

static void make_dataset(memmg::managed_array<uint64_t>& cpu_data,
                         memmg::managed_array<uint64_t>& gpu_data, std::mt19937& rng,
                         unsigned int n) noexcept {
  cpu_data = memmg::managed_array<uint64_t>(n);
  gpu_data = memmg::managed_array<uint64_t>(n, memr::get_device_resource());
  for (size_t i = 0; i < n; ++i) {
    cpu_data[i] = rng();
  }
  basdv::memcpy_host_to_device(gpu_data.data(), cpu_data.data(), n * sizeof(uint64_t));
}

static int64_t compute_sum(const memmg::managed_array<uint64_t>& data) noexcept {
  uint64_t res = 0;
  for (auto x : data) {
    res += x;
  }
  return res;
}

TEST_CASE("we can perform reductions on the GPU") {
  std::mt19937 rng{0};

  memmg::managed_array<uint64_t> cpu_data;
  memmg::managed_array<uint64_t> gpu_data{memr::get_device_resource()};

  SECTION("we can reduce a single element") {
    make_dataset(cpu_data, gpu_data, rng, 1);
    auto res = reduce<test_add_reducer>(algb::identity_mapper{gpu_data.data()}, 1);
    REQUIRE(res.await_result() == cpu_data[0]);
  }

  SECTION("we can reduce two elements") {
    make_dataset(cpu_data, gpu_data, rng, 2);
    auto res = reduce<test_add_reducer>(algb::identity_mapper{gpu_data.data()}, 2);
    REQUIRE(res.await_result() == cpu_data[0] + cpu_data[1]);
  }

  SECTION("we can reduce small data sets") {
    for (auto n : {4, 31, 32, 33, 42, 63, 64, 65}) {
      make_dataset(cpu_data, gpu_data, rng, n);
      auto res = reduce<test_add_reducer>(algb::identity_mapper{gpu_data.data()}, n);
      REQUIRE(res.await_result() == compute_sum(cpu_data));
    }
  }

  SECTION("we can reduce datasets that are a power of two") {
    for (unsigned int n = 2; n < (1u << 12u); n <<= 1u) {
      make_dataset(cpu_data, gpu_data, rng, n);
      auto res = reduce<test_add_reducer>(algb::identity_mapper{gpu_data.data()}, n);
      REQUIRE(res.await_result() == compute_sum(cpu_data));
    }
  }

  SECTION("we can reduce arbitrary random data sets") {
    std::uniform_int_distribution<unsigned int> dist{128, 1u << 15u};
    for (int i = 0; i < 10; ++i) {
      auto n = dist(rng);
      make_dataset(cpu_data, gpu_data, rng, n);
      auto res = reduce<test_add_reducer>(algb::identity_mapper{gpu_data.data()}, n);
      REQUIRE(res.await_result() == compute_sum(cpu_data));
    }
  }
}
