#include "sxt/seqcommit/cbindings/get_one_commit.h"

#include <algorithm>
#include <array>
#include <string>
#include <vector>

#include "sxt/base/test/unit_test.h"
#include "sxt/curve21/constant/zero.h"
#include "sxt/curve21/operation/add.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/seqcommit/cbindings/backend.h"
#include "sxt/seqcommit/cbindings/get_generators.h"

using namespace sxt;

static void initialize_backend(int backend, uint64_t precomputed_elements) {
  const sxt_config config = {backend, precomputed_elements};
  REQUIRE(sxt_init(&config) == 0);
}

static std::vector<c21t::element_p3> initialize_generators(int backend, uint64_t num_generators) {
  initialize_backend(backend, 0);

  std::vector<c21t::element_p3> generators(num_generators);
  REQUIRE(sxt_get_generators(reinterpret_cast<sxt_ristretto*>(generators.data()), num_generators,
                             0) == 0);

  sxt::sqccb::reset_backend_for_testing();

  return generators;
}

static void verify_one_commit(int backend, uint64_t num_precomputed_els, uint64_t n,
                              const c21t::element_p3& expected_element) {
  initialize_backend(backend, num_precomputed_els);

  sxt_ristretto one_commitment;
  REQUIRE(sxt_get_one_commit(&one_commitment, n) == 0);
  REQUIRE(reinterpret_cast<c21t::element_p3*>(&one_commitment)[0] == expected_element);

  sxt::sqccb::reset_backend_for_testing();
}

static void test_one_commit_with_given_backend(int backend) {
  auto generators = initialize_generators(backend, 10);

  SECTION("The first one_commit should be zero") {
    uint64_t num_precomputed_els = 0, one_commit_n = 0;
    verify_one_commit(backend, num_precomputed_els, one_commit_n, c21cn::zero_p3_v);
  }

  SECTION("The second one_commit should be equal to the first generator") {
    uint64_t num_precomputed_els = 0, one_commit_n = 1;
    verify_one_commit(backend, num_precomputed_els, one_commit_n, generators[0]);
  }

  SECTION("The third one_commit should be equal to the first + second generators") {
    c21t::element_p3 sum_gen_0_1;
    c21o::add(sum_gen_0_1, generators[0], generators[1]);

    uint64_t num_precomputed_els = 0, one_commit_n = 2;
    verify_one_commit(backend, num_precomputed_els, one_commit_n, sum_gen_0_1);
  }

  SECTION("The third one_commit should be equal to the first + second generators even when we "
          "precompute elements") {
    c21t::element_p3 sum_gen_0_1;
    c21o::add(sum_gen_0_1, generators[0], generators[1]);

    uint64_t num_precomputed_els = 1, one_commit_n = 2;
    verify_one_commit(backend, num_precomputed_els, one_commit_n, sum_gen_0_1);
  }
}

TEST_CASE("We can correctly fetch the one commit using the naive cpu backend") {
  test_one_commit_with_given_backend(SXT_NAIVE_BACKEND_CPU);
}

TEST_CASE("We can correctly fetch the one commit using the naive gpu backend") {
  test_one_commit_with_given_backend(SXT_NAIVE_BACKEND_GPU);
}

TEST_CASE("We can correctly fetch the one commit using the pippenger cpu backend") {
  test_one_commit_with_given_backend(SXT_PIPPENGER_BACKEND_CPU);
}
