#include "sxt/multiexp/pippenger/radix_log2.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/multiexp/base/exponent.h"

using namespace sxt;
using namespace sxt::mtxpi;

TEST_CASE("we can compute the radix of arbitrary exponents") {

    SECTION("we correctly handle zero inputs and zero outputs") {
        mtxb::exponent max_exponent;

        // exponent = 140 and input / output is zero
        REQUIRE(compute_radix_log2(mtxb::exponent(140ull, 0ull, 0ull, 0ull), 0, 1) == 1);

        REQUIRE(compute_radix_log2(mtxb::exponent(140ull, 0ull, 0ull, 0ull), 1, 0) == 1);

        REQUIRE(compute_radix_log2(mtxb::exponent(140ull, 0ull, 0ull, 0ull), 0, 0) == 1);

        // exponent = 0
        REQUIRE(compute_radix_log2(mtxb::exponent(), 1, 1) == 1);
    }

    SECTION("we correctly handle different input and output values") {
        REQUIRE(
            compute_radix_log2(mtxb::exponent(140ull, 0ull, 0ull, 0ull), 2, 1) == 2
        );

        REQUIRE(
            compute_radix_log2(mtxb::exponent(140ull, 0ull, 0ull, 0ull), 1, 2) == 2
        );
    }

    SECTION("we correctly handle big input and output values") {
        REQUIRE(
            compute_radix_log2(mtxb::exponent(140ull, 0ull, 0ull, 0ull), 1, 10000000000) == 1
        );

        REQUIRE(
            compute_radix_log2(mtxb::exponent(140ull, 0ull, 0ull, 0ull), 10000000000, 1) == 1
        );
    }

    SECTION("we correctly handle big exponents") {
        // exponent = p = 2^252 + 27742317777372353535851937790883648493
        REQUIRE(
            compute_radix_log2(
                mtxb::exponent(
                    6346243789798364141ull,
                    1503914060200516822ull,
                    0ull,
                    1152921504606846976ull
                ),
                2, 10
            ) == 8
        );

        // exponent = 2^256 - 1
        REQUIRE(
            compute_radix_log2(
                mtxb::exponent(
                    std::numeric_limits<uint64_t>::max(),
                    std::numeric_limits<uint64_t>::max(),
                    std::numeric_limits<uint64_t>::max(),
                    std::numeric_limits<uint64_t>::max()
                ),
                1, 1
            ) == 16
        );
    }
}
