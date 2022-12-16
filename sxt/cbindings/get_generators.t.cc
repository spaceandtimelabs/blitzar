#include "sxt/cbindings/get_generators.h"

#include <cassert>
#include <vector>

#include "sxt/base/test/unit_test.h"
#include "sxt/cbindings/backend.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/seqcommit/generator/base_element.h"

using namespace sxt;
using namespace sxt::cbn;

static void initialize_backend(int backend, uint64_t precomputed_elements) {
  const sxt_config config = {backend, precomputed_elements};
  REQUIRE(sxt_init(&config) == 0);
}

static std::vector<sxt_ristretto> initialize_generators(int backend,
                                                        uint64_t num_precomputed_elements,
                                                        uint64_t num_generators, uint64_t offset) {
  initialize_backend(backend, num_precomputed_elements);

  std::vector<sxt_ristretto> generators(num_generators);
  REQUIRE(sxt_get_generators(generators.data(), num_generators, offset) == 0);

  reset_backend_for_testing();

  return generators;
}

static void verify_generator(const std::vector<sxt_ristretto>& generators, uint64_t index,
                             uint64_t offset) {
  assert(generators.size() > index);

  c21t::element_p3 expected_gi;
  sqcgn::compute_base_element(expected_gi, index + offset);
  REQUIRE(expected_gi == reinterpret_cast<const c21t::element_p3*>(generators.data())[index]);
}

static void test_generators_with_given_backend(int backend) {
  SECTION("We cannot fetch more than zero generators when we have a null pointer input") {
    initialize_backend(backend, 0);
    uint64_t num_generators = 3, offset_generators = 0;
    REQUIRE(sxt_get_generators(nullptr, num_generators, offset_generators) != 0);
    reset_backend_for_testing();
  }

  SECTION("We can fetch zero generators with null pointer input") {
    initialize_backend(backend, 0);
    uint64_t num_generators = 0, offset_generators = 0;
    REQUIRE(sxt_get_generators(nullptr, num_generators, offset_generators) == 0);
    reset_backend_for_testing();
  }

  SECTION("We can fetch at least one generator using zero precomputed elements and zero offset") {
    uint64_t num_precomputed_els = 0, num_generators = 10, offset_generators = 0;
    auto generators =
        initialize_generators(backend, num_precomputed_els, num_generators, offset_generators);

    verify_generator(generators, 0, 0);
    verify_generator(generators, 1, 0);
    verify_generator(generators, 9, 0);
  }

  SECTION(
      "We can fetch at least one generator using non-zero precomputed elements, but zero offset") {
    uint64_t num_precomputed_els = 10, num_generators = 10, offset_generators = 0;
    auto generators =
        initialize_generators(backend, num_precomputed_els, num_generators, offset_generators);

    verify_generator(generators, 0, 0);
    verify_generator(generators, 1, 0);
    verify_generator(generators, 9, 0);
  }

  SECTION("We can correctly compute generators when offset is non-zero, but precomputed elements "
          "is zero") {
    uint64_t num_precomputed_els = 0, num_generators = 10, offset_generators = 15;
    auto generators =
        initialize_generators(backend, num_precomputed_els, num_generators, offset_generators);

    verify_generator(generators, 0, offset_generators);
    verify_generator(generators, 2, offset_generators);
    verify_generator(generators, 9, offset_generators);
  }

  SECTION("We can correctly compute generators when the offset is bigger than the number of "
          "precomputed elements") {
    uint64_t num_precomputed_generators = 10, num_generators = 10, offset_generators = 15;
    auto generators = initialize_generators(backend, num_precomputed_generators, num_generators,
                                            offset_generators);

    verify_generator(generators, 0, offset_generators);
    verify_generator(generators, 2, offset_generators);
    verify_generator(generators, 9, offset_generators);
  }
}

TEST_CASE("We can correctly get generators using the naive cpu backend") {
  test_generators_with_given_backend(SXT_NAIVE_BACKEND_CPU);
}

TEST_CASE("We can correctly get generators using the naive gpu backend") {
  test_generators_with_given_backend(SXT_NAIVE_BACKEND_GPU);
}

TEST_CASE("We can correctly get generators using the pippenger cpu backend") {
  test_generators_with_given_backend(SXT_PIPPENGER_BACKEND_CPU);
}
