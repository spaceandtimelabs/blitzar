#include "sxt/base/num/log2p1.h"

#include <array>
#include <cmath>

#include "sxt/base/test/unit_test.h"

using namespace Catch;
using namespace sxt::basn;
using namespace sxt::basct;

TEST_CASE("we can take the base 2 logarithmic of arbitrary numbers") {
    // log2(0 + 1)
    REQUIRE(log2p1(std::array<uint8_t, 1>{0}) == Approx(0.));

    // log2(1 + 1)
    REQUIRE(log2p1(std::array<uint8_t, 1>{1}) == Approx(1.));

    // log2(2 + 1)
    REQUIRE(log2p1(std::array<uint8_t, 1>{2}) == Approx(1.584962500721156));

    // log2(3 + 1)
    REQUIRE(log2p1(std::array<uint8_t, 1>{3}) == Approx(2.));

    // log2(140 + 1)
    REQUIRE(log2p1(std::array<uint8_t, 1>{140}) == Approx(7.139551352398794));
    
    // log2(p + 1), p = 2^252 + 27742317777372353535851937790883648493 (256 bits)
    REQUIRE(
        log2p1(cspan<uint8_t>(reinterpret_cast<uint8_t *>(
            std::array<unsigned long long, 4>{
                6346243789798364141ull, 1503914060200516822ull, 0ull, 1152921504606846976ull
            }.data()), 32)
        ) == Approx(252.)
    );

    // log2(x + 1),  x = 7237005577332262213973186563043003164696160195119475...
    // ...807682098074301841107485259085803440698822625692763935725 (384 bits)
    REQUIRE(
        log2p1(cspan<uint8_t>(reinterpret_cast<uint8_t *>(
            std::array<unsigned long long, 6>{
                17991574144297587693ull, 410974272831746224ull,
                14625339942921143157ull, 1421653999304472111ull, 
                3714367641016598528ull, 3388131789017ull
            }.data()), 48)
        ) == Approx(361.62362713128294)
    );

    // maximum allowed number
    // log2(x + 1),  x = 2^1016 - 1 (1016 bits)
    REQUIRE(
        log2p1(cspan<uint8_t>(reinterpret_cast<uint8_t *>(
            std::array<unsigned long long, 16>{
                std::numeric_limits<uint64_t>::max(),
                std::numeric_limits<uint64_t>::max(),
                std::numeric_limits<uint64_t>::max(),
                std::numeric_limits<uint64_t>::max(),
                std::numeric_limits<uint64_t>::max(),
                std::numeric_limits<uint64_t>::max(),
                std::numeric_limits<uint64_t>::max(),
                std::numeric_limits<uint64_t>::max(),
                std::numeric_limits<uint64_t>::max(),
                std::numeric_limits<uint64_t>::max(),
                std::numeric_limits<uint64_t>::max(),
                std::numeric_limits<uint64_t>::max(),
                std::numeric_limits<uint64_t>::max(),
                std::numeric_limits<uint64_t>::max(),
                std::numeric_limits<uint64_t>::max(),
                72057594037927935ull,
            }.data()), 127)
        ) == Approx(1016.)
    );
}
