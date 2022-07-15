#include "sxt/multiexp/ristretto/multiexponentiation_cpu_driver.h"

#include <iostream>
#include <memory_resource>
#include <random>
#include <vector>

#include "sxt/base/container/span.h"
#include "sxt/base/num/fast_random_number_generator.h"
#include "sxt/base/test/unit_test.h"

#include "sxt/memory/management/managed_array.h"

#include "sxt/multiexp/base/exponent_sequence.h"
#include "sxt/multiexp/pippenger/multiexponentiation.h"
#include "sxt/multiexp/random/random_multiexponentiation_descriptor.h"
#include "sxt/multiexp/random/random_multiexponentiation_generation.h"
#include "sxt/multiexp/test/compute_ristretto_muladd.h"
#include "sxt/multiexp/test/generate_ristretto_elements.h"

#include "sxt/ristretto/base/byte_conversion.h"
#include "sxt/ristretto/operation/add.h"
#include "sxt/ristretto/operation/scalar_multiply.h"
#include "sxt/ristretto/random/element.h"
#include "sxt/ristretto/type/compressed_element.h"

using namespace sxt;
using namespace sxt::mtxrs;

static void compute_random_test_case(std::mt19937& rng, size_t num_sequences,
                                     mtxrn::random_multiexponentiation_descriptor descriptor,
                                     const multiexponentiation_cpu_driver& drv) {

  uint64_t num_inputs = 0;

  std::pmr::monotonic_buffer_resource resource;
  std::vector<mtxb::exponent_sequence> sequences(num_sequences);

  mtxrn::generate_random_multiexponentiation(num_inputs, sequences, &resource, rng, descriptor);

  memmg::managed_array<rstt::compressed_element> inout(num_inputs);

  mtxtst::generate_ristretto_elements(inout, rng);

  memmg::managed_array<rstt::compressed_element> expected_result(num_sequences);

  mtxtst::compute_ristretto_muladd(expected_result, inout, sequences);

  mtxpi::compute_multiexponentiation(inout, drv, sequences);

  REQUIRE(inout == expected_result);
}

TEST_CASE("Pippenger Multiexponentiation using Ristretto Points") {
  multiexponentiation_cpu_driver drv;

  std::mt19937 rng{2022};
  rstt::compressed_element gs[4];
  mtxtst::generate_ristretto_elements(gs, rng);

  SECTION("We handle zero sequences") {
    memmg::managed_array<rstt::compressed_element> inout;

    std::vector<mtxb::exponent_sequence> sequences;

    mtxpi::compute_multiexponentiation(inout, drv, sequences);

    REQUIRE(inout.empty());
  }

  SECTION("We handle a single sequence") {
    SECTION("with zero exponents") {
      memmg::managed_array<rstt::compressed_element> inout;

      std::vector<mtxb::exponent_sequence> sequences = {
          {.element_nbytes = 0, .n = 0, .data = nullptr}};

      mtxpi::compute_multiexponentiation(inout, drv, sequences);

      REQUIRE(inout.size() == 1);
      REQUIRE(inout[0] == rstt::compressed_element{});
    }

    SECTION("with one exponent") {
      SECTION("being the zero multiplier cases") {
        std::vector<uint64_t> exponents = {0};

        memmg::managed_array<rstt::compressed_element> inout = {gs[0]};

        std::vector<mtxb::exponent_sequence> sequences = {
            mtxb::exponent_sequence{.element_nbytes = sizeof(exponents[0]),
                                    .n = 1,
                                    .data = reinterpret_cast<uint8_t*>(exponents.data())}};

        compute_multiexponentiation(inout, drv, sequences);

        REQUIRE(inout.size() == 1);
        REQUIRE(inout == memmg::managed_array{rstt::compressed_element{}});
      }

      SECTION("being the one multiplier case") {
        std::vector<uint32_t> exponents = {1};

        memmg::managed_array<rstt::compressed_element> inout = {gs[0]};

        std::vector<mtxb::exponent_sequence> sequences = {
            mtxb::exponent_sequence{.element_nbytes = sizeof(exponents[0]),
                                    .n = 1,
                                    .data = reinterpret_cast<uint8_t*>(exponents.data())}};

        compute_multiexponentiation(inout, drv, sequences);

        REQUIRE(inout.size() == 1);
        REQUIRE(inout[0] == gs[0]);
      }

      SECTION("being the two multiplier case") {
        std::vector<uint8_t> exponents = {2};

        memmg::managed_array<rstt::compressed_element> inout = {gs[0]};

        std::vector<mtxb::exponent_sequence> sequences = {
            mtxb::exponent_sequence{.element_nbytes = sizeof(exponents[0]),
                                    .n = 1,
                                    .data = reinterpret_cast<uint8_t*>(exponents.data())}};

        compute_multiexponentiation(inout, drv, sequences);

        rstt::compressed_element g_0_2rt;
        rsto::add(g_0_2rt, gs[0], gs[0]); // g_0_2 = g_0 + g_0

        REQUIRE(inout.size() == 1);
        REQUIRE(inout[0] == g_0_2rt);
      }

      SECTION("being the three multiplier case") {
        std::vector<uint16_t> exponents = {3};

        memmg::managed_array<rstt::compressed_element> inout = {gs[0]};

        std::vector<mtxb::exponent_sequence> sequences = {
            mtxb::exponent_sequence{.element_nbytes = sizeof(exponents[0]),
                                    .n = 1,
                                    .data = reinterpret_cast<uint8_t*>(exponents.data())}};

        compute_multiexponentiation(inout, drv, sequences);

        rstt::compressed_element g_0_3rt;
        rsto::add(g_0_3rt, gs[0], gs[0]);   // g_0_3 = g_0 + g_0
        rsto::add(g_0_3rt, g_0_3rt, gs[0]); // g_0_3 = g_0_3 + g_0

        REQUIRE(inout.size() == 1);
        REQUIRE(inout[0] == g_0_3rt);
      }

      SECTION("being a large multiplier case") {
        std::vector<uint64_t> exponents = {1'000'000};

        memmg::managed_array<rstt::compressed_element> inout = {gs[0]};

        std::vector<mtxb::exponent_sequence> sequences = {
            mtxb::exponent_sequence{.element_nbytes = sizeof(exponents[0]),
                                    .n = 1,
                                    .data = reinterpret_cast<uint8_t*>(exponents.data())}};

        compute_multiexponentiation(inout, drv, sequences);

        rstt::compressed_element g_0_3rt;
        rsto::scalar_multiply(g_0_3rt, exponents[0], gs[0]);

        REQUIRE(inout.size() == 1);
        REQUIRE(inout[0] == g_0_3rt);
      }

      SECTION("being the maximum allowed multiplier case") {
        std::vector<uint64_t> exponents = {std::numeric_limits<uint64_t>::max()};

        memmg::managed_array<rstt::compressed_element> inout = {gs[0]};

        std::vector<mtxb::exponent_sequence> sequences = {
            mtxb::exponent_sequence{.element_nbytes = sizeof(exponents[0]),
                                    .n = 1,
                                    .data = reinterpret_cast<uint8_t*>(exponents.data())}};

        compute_multiexponentiation(inout, drv, sequences);

        rstt::compressed_element g_0_3rt;
        rsto::scalar_multiply(g_0_3rt, exponents[0], gs[0]);

        REQUIRE(inout.size() == 1);
        REQUIRE(inout[0] == g_0_3rt);
      }
    }

    SECTION("with many exponents") {
      std::vector<uint16_t> exponents = {3, 2};

      memmg::managed_array<rstt::compressed_element> inout = {gs[2], gs[1]};

      std::vector<mtxb::exponent_sequence> sequences = {
          mtxb::exponent_sequence{.element_nbytes = sizeof(exponents[0]),
                                  .n = static_cast<uint8_t>(exponents.size()),
                                  .data = reinterpret_cast<uint8_t*>(exponents.data())}};

      compute_multiexponentiation(inout, drv, sequences);

      rstt::compressed_element g_rt;
      rsto::add(g_rt, gs[2], gs[2]); // g_rt = g_2 + g_2
      rsto::add(g_rt, g_rt, gs[2]);  // g_rt = g_rt + g_2
      rsto::add(g_rt, g_rt, gs[1]);  // g_rt = g_rt + g_1
      rsto::add(g_rt, g_rt, gs[1]);  // g_rt = g_rt + g_1

      REQUIRE(inout.size() == 1);
      REQUIRE(inout[0] == g_rt);
    }
  }

  SECTION("We handle many sequences with the same length,") {
    SECTION("having zero exponents") {
      memmg::managed_array<rstt::compressed_element> inout;

      std::vector<mtxb::exponent_sequence> sequences = {
          {.element_nbytes = 0, .n = 0, .data = nullptr},
          {.element_nbytes = 0, .n = 0, .data = nullptr}};

      mtxpi::compute_multiexponentiation(inout, drv, sequences);

      REQUIRE(inout.size() == 2);
      REQUIRE(inout[0] == rstt::compressed_element{});
      REQUIRE(inout[1] == rstt::compressed_element{});
    }

    SECTION("having one exponent") {
      memmg::managed_array<rstt::compressed_element> inout = {gs[2]};
      std::vector<uint8_t> exponents1 = {1};
      std::vector<uint16_t> exponents2 = {3};

      std::vector<mtxb::exponent_sequence> sequences = {
          {.element_nbytes = sizeof(exponents1[0]),
           .n = exponents1.size(),
           .data = reinterpret_cast<const uint8_t*>(exponents1.data())},
          {.element_nbytes = sizeof(exponents2[0]),
           .n = exponents2.size(),
           .data = reinterpret_cast<const uint8_t*>(exponents2.data())},
      };

      mtxpi::compute_multiexponentiation(inout, drv, sequences);

      rstt::compressed_element expected_result_rt;
      rsto::add(expected_result_rt, inout[0], inout[0]);
      rsto::add(expected_result_rt, expected_result_rt, inout[0]);

      REQUIRE(expected_result_rt == inout[1]);
      REQUIRE(expected_result_rt != rstt::compressed_element{});
    }

    SECTION("having many exponents") {
      memmg::managed_array<rstt::compressed_element> inout = {gs[0], gs[1], gs[2], gs[3]};

      std::vector<uint16_t> exponents1 = {2000, 7500, 5000, 1500};
      std::vector<uint32_t> exponents2 = {5000, 0, 400000, 10};
      std::vector<uint64_t> exponents3 = {2000 + 5000, 7500 + 0, 5000 + 400000, 1500 + 10};

      std::vector<mtxb::exponent_sequence> sequences = {
          {.element_nbytes = sizeof(exponents1[0]),
           .n = exponents1.size(),
           .data = reinterpret_cast<const uint8_t*>(exponents1.data())},
          {.element_nbytes = sizeof(exponents2[0]),
           .n = exponents2.size(),
           .data = reinterpret_cast<const uint8_t*>(exponents2.data())},
          {.element_nbytes = sizeof(exponents3[0]),
           .n = exponents3.size(),
           .data = reinterpret_cast<const uint8_t*>(exponents3.data())},
      };

      mtxpi::compute_multiexponentiation(inout, drv, sequences);

      rstt::compressed_element expected_result_rt;
      rsto::add(expected_result_rt, inout[0], inout[1]);

      REQUIRE(expected_result_rt == inout[2]);
      REQUIRE(inout[2] != rstt::compressed_element{});
    }
  }

  SECTION("We handle many sequences with different length and") {
    SECTION("having many exponents") {
      memmg::managed_array<rstt::compressed_element> inout = {gs[0], gs[0], gs[0]};
      std::vector<uint8_t> exponents1 = {1, 1, 1};
      std::vector<uint8_t> exponents2 = {3};

      std::vector<mtxb::exponent_sequence> sequences = {
          {.element_nbytes = sizeof(exponents1[0]),
           .n = exponents1.size(),
           .data = reinterpret_cast<const uint8_t*>(exponents1.data())},
          {.element_nbytes = sizeof(exponents2[0]),
           .n = exponents2.size(),
           .data = reinterpret_cast<const uint8_t*>(exponents2.data())},
      };

      mtxpi::compute_multiexponentiation(inout, drv, sequences);

      REQUIRE(inout[0] == inout[1]);
      REQUIRE(inout[0] != rstt::compressed_element{});
    }
  }

  SECTION("we handle multiple random sequences of varying length") {
    for (size_t i = 0; i < 10; ++i) {
      compute_random_test_case(rng, 1,
                               {.min_sequence_length = 0,
                                .max_sequence_length = 100,
                                .min_exponent_num_bytes = 1,
                                .max_exponent_num_bytes = 1},
                               drv);
    }
  }

  SECTION("we handle multiple outputs and random sequences of length 1") {
    for (size_t i = 0; i < 10; ++i) {
      compute_random_test_case(rng, i,
                               {.min_sequence_length = 1,
                                .max_sequence_length = 1,
                                .min_exponent_num_bytes = 1,
                                .max_exponent_num_bytes = 1},
                               drv);
    }
  }

  SECTION("we handle multiple outputs and"
          "multiple random sequences of varying length") {
    for (size_t i = 0; i < 10; ++i) {
      compute_random_test_case(rng, i,
                               {.min_sequence_length = 1,
                                .max_sequence_length = 100,
                                .min_exponent_num_bytes = 1,
                                .max_exponent_num_bytes = 1},
                               drv);
    }
  }

  SECTION("we handle random sequences of length 1 and varying num_bytes") {
    for (size_t i = 0; i < 10; ++i) {
      compute_random_test_case(rng, 1,
                               {.min_sequence_length = 1,
                                .max_sequence_length = 1,
                                .min_exponent_num_bytes = 1,
                                .max_exponent_num_bytes = 32},
                               drv);
    }
  }

  SECTION("we handle multiple random sequences of"
          "varying length and varying num_bytes") {
    for (size_t i = 0; i < 10; ++i) {
      compute_random_test_case(rng, 1,
                               {.min_sequence_length = 1,
                                .max_sequence_length = 100,
                                .min_exponent_num_bytes = 1,
                                .max_exponent_num_bytes = 32},
                               drv);
    }
  }

  SECTION("we handle multiple outputs and"
          "multiple random sequences of varying"
          "length and varying num_bytes") {
    for (size_t i = 0; i < 10; ++i) {
      compute_random_test_case(rng, i,
                               {.min_sequence_length = 1,
                                .max_sequence_length = 100,
                                .min_exponent_num_bytes = 1,
                                .max_exponent_num_bytes = 32},
                               drv);
    }
  }
}
