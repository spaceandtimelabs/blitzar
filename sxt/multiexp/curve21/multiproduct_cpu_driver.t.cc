#include "sxt/multiexp/curve21/multiproduct_cpu_driver.h"

#include <random>

#include "sxt/base/container/span.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/multiexp/index/index_table.h"
#include "sxt/multiexp/pippenger_multiprod/multiproduct.h"
#include "sxt/multiexp/random/random_multiproduct_descriptor.h"
#include "sxt/multiexp/random/random_multiproduct_generation.h"
#include "sxt/multiexp/test/add_curve21_elements.h"
#include "sxt/ristretto/random/element.h"

using namespace sxt;
using namespace sxt::mtxc21;

static void verify_random_example(std::mt19937& rng,
                                  const mtxrn::random_multiproduct_descriptor& descriptor) {
  mtxi::index_table products;
  multiproduct_cpu_driver drv;
  size_t num_inputs, num_entries;

  mtxrn::generate_random_multiproduct(products, num_inputs, num_entries, rng, descriptor);

  memmg::managed_array<c21t::element_p3> inout(num_entries);
  rstrn::generate_random_elements(basct::span<c21t::element_p3>{inout.data(), num_inputs}, rng);

  memmg::managed_array<c21t::element_p3> expected_result(products.num_rows());
  mtxtst::add_curve21_elements(expected_result, products.cheader(), inout);

  mtxpmp::compute_multiproduct(inout, products, drv, num_inputs);

  for (size_t index = 0; index < products.num_rows(); ++index) {
    REQUIRE(inout[index] == expected_result[index]);
  }
}

TEST_CASE("we can compute curve21 multiproducts") {
  std::mt19937 rng{2022};
  multiproduct_cpu_driver drv;

  SECTION("we handle the empty case") {
    memmg::managed_array<c21t::element_p3> inout;
    mtxi::index_table products;

    mtxpmp::compute_multiproduct(inout, products, drv, 0);

    REQUIRE(inout.empty());
  }

  SECTION("we handle a multiproduct with a single term") {
    size_t num_inputs = 1;
    memmg::managed_array<c21t::element_p3> inout(1);
    rstrn::generate_random_elements(basct::span<c21t::element_p3>{inout.data(), num_inputs}, rng);

    mtxi::index_table products{{0}};

    memmg::managed_array<c21t::element_p3> expected_result(1);
    mtxtst::add_curve21_elements(expected_result, products.cheader(), inout);

    mtxpmp::compute_multiproduct(inout, products, drv, num_inputs);

    REQUIRE(inout == expected_result);
  }

  SECTION("we handle a single output with multiple terms") {
    size_t num_inputs = 2;
    memmg::managed_array<c21t::element_p3> inout(2);
    rstrn::generate_random_elements(basct::span<c21t::element_p3>{inout.data(), num_inputs}, rng);

    mtxi::index_table products{{0, 1}};

    memmg::managed_array<c21t::element_p3> expected_result(1);
    mtxtst::add_curve21_elements(expected_result, products.cheader(), inout);

    mtxpmp::compute_multiproduct(inout, products, drv, num_inputs);

    REQUIRE(inout[0] == expected_result[0]);
  }

  SECTION("we handle a multi-product with two outputs") {
    size_t num_inputs = 3;
    memmg::managed_array<c21t::element_p3> inout(5);
    rstrn::generate_random_elements(basct::span<c21t::element_p3>{inout.data(), num_inputs}, rng);

    mtxi::index_table products{{0, 1, 2}, {0, 2}};

    memmg::managed_array<c21t::element_p3> expected_result(2);
    mtxtst::add_curve21_elements(expected_result, products.cheader(), inout);

    mtxpmp::compute_multiproduct(inout, products, drv, num_inputs);

    REQUIRE(inout[0] == expected_result[0]);
    REQUIRE(inout[1] == expected_result[1]);
  }

  SECTION("we can handle random multiproducts with only two rows and few entries") {
    mtxrn::random_multiproduct_descriptor random_descriptor{
        .min_sequence_length = 1,
        .max_sequence_length = 4,
        .min_num_sequences = 2,
        .max_num_sequences = 2,
        .max_num_inputs = 8,
    };

    for (int i = 0; i < 100; ++i) {
      verify_random_example(rng, random_descriptor);
    }
  }

  SECTION("we handle random multiproducts with two rows and many entries") {
    mtxrn::random_multiproduct_descriptor random_descriptor{
        .min_sequence_length = 1,
        .max_sequence_length = 20,
        .min_num_sequences = 2,
        .max_num_sequences = 2,
        .max_num_inputs = 20,
    };

    for (int i = 0; i < 100; ++i) {
      verify_random_example(rng, random_descriptor);
    }
  }

  SECTION("we handle random multiproducts with three rows") {
    mtxrn::random_multiproduct_descriptor random_descriptor{
        .min_sequence_length = 1,
        .max_sequence_length = 20,
        .min_num_sequences = 3,
        .max_num_sequences = 3,
        .max_num_inputs = 20,
    };

    for (int i = 0; i < 100; ++i) {
      verify_random_example(rng, random_descriptor);
    }
  }

  SECTION("we handle random multiproducts with multiple rows") {
    mtxrn::random_multiproduct_descriptor random_descriptor{
        .min_sequence_length = 1,
        .max_sequence_length = 20,
        .min_num_sequences = 1,
        .max_num_sequences = 10,
        .max_num_inputs = 20,
    };

    for (int i = 0; i < 100; ++i) {
      verify_random_example(rng, random_descriptor);
    }
  }

  SECTION("we handle random multiproducts with many rows") {
    mtxrn::random_multiproduct_descriptor random_descriptor{
        .min_sequence_length = 1,
        .max_sequence_length = 100,
        .min_num_sequences = 100,
        .max_num_sequences = 1000,
        .max_num_inputs = 200,
    };

    for (int i = 0; i < 5; ++i) {
      verify_random_example(rng, random_descriptor);
    }
  }

  SECTION("we handle random multiproducts with many inputs") {
    mtxrn::random_multiproduct_descriptor random_descriptor{
        .min_sequence_length = 1000,
        .max_sequence_length = 2000,
        .min_num_sequences = 1,
        .max_num_sequences = 10,
        .max_num_inputs = 4000,
    };

    for (int i = 0; i < 10; ++i) {
      verify_random_example(rng, random_descriptor);
    }
  }
}
