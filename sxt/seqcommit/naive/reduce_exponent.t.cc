#include "sxt/seqcommit/naive/reduce_exponent.h"

#include "sxt/base/test/unit_test.h"

#include <array>
#include <bitset>
#include <string>

using namespace sxt;
using namespace sxt::sqcnv;

static void to_array(std::array<unsigned long long, 4> &s, std::string &s_bin) {
    std::bitset<64> temp(0);
    for (size_t i = 0; i < s_bin.size(); ++i) {
        temp.set(i % 64, s_bin[i] - '0');

        if ((i + 1) % 64 == 0 || i + 1 >= s_bin.size()) {
            s[i / 64] = temp.to_ullong();

            temp = std::bitset<64>(0);
        }
    }
}

TEST_CASE("Test 1 - We can verify that reduce does not change input array A when A < p") {
    // p = 2^252 + 27742317777372353535851937790883648493 (decimal)
    // s = p - 1 (decimal)
    // s % p = p - 1 (decimal)
    std::array<unsigned long long, 4> s;
    std::string s_bin = std::string("0011011111001011101011110011101001011000110001100100100000011010") +
                        std::string("0110101100111001111011110100010101111011100111110111101100101000") +
                        std::string("0000000000000000000000000000000000000000000000000000000000000000") +
                        std::string("0000000000000000000000000000000000000000000000000000000000001");

    to_array(s, s_bin);

    SECTION("Verifying for A mod p == A holds (p = 2^252 + 27742317777372353535851937790883648493)") {
        std::array<unsigned long long, 4> expected_s = s;

        reduce_exponent((unsigned char *) s.data());

        REQUIRE(s == expected_s);
    }
}

TEST_CASE("Test 2 - We can verify that reduce change input array A when A >= p") {
    // p = 2^252 + 27742317777372353535851937790883648493 (decimal)
    // s = p (decimal)
    // s % p = 0 (decimal)
    std::array<unsigned long long, 4> s;
    std::string s_bin = std::string("1011011111001011101011110011101001011000110001100100100000011010") +
                        std::string("0110101100111001111011110100010101111011100111110111101100101000") +
                        std::string("0000000000000000000000000000000000000000000000000000000000000000") +
                        std::string("0000000000000000000000000000000000000000000000000000000000001");

    to_array(s, s_bin);

    SECTION("Verifying for A mod p == 0 holds (p = 2^252 + 27742317777372353535851937790883648493)") {
        reduce_exponent((unsigned char *) s.data());

        REQUIRE(s == std::array<unsigned long long, 4>({0, 0, 0, 0}));
    }
}


TEST_CASE("Test 3 - We can verify that reduce change input array A when A >= p") {
    // p = 2^252 + 27742317777372353535851937790883648493 (decimal)
    // s = p + 103 (decimal)
    // s % p = 103 (decimal)
    std::array<unsigned long long, 4> s;
    std::string s_bin = std::string("0010101000101011101011110011101001011000110001100100100000011010") +
                        std::string("0110101100111001111011110100010101111011100111110111101100101000") +
                        std::string("0000000000000000000000000000000000000000000000000000000000000000") +
                        std::string("0000000000000000000000000000000000000000000000000000000000001");

    to_array(s, s_bin);

    SECTION("Verifying for A mod p == 103 holds (p = 2^252 + 27742317777372353535851937790883648493)") {
        reduce_exponent((unsigned char *) s.data());

        REQUIRE(s == std::array<unsigned long long, 4>({103, 0, 0, 0}));
    }
}

TEST_CASE("Test 4 - We can verify that reduce correctly change input array A even when A is the maximum possible value") {
    // p = 2^252 + 27742317777372353535851937790883648493 (decimal)
    // s =     115792089237316195423570985008687907853269984665640564039457584007913129639935 (decimal)
    // s % p = 7237005577332262213973186563042994240413239274941949949428319933631315875100 (decimal)
    std::array<unsigned long long, 4> s;
    std::array<unsigned long long, 4> expected_s;
    std::string s_bin = std::string("1111111111111111111111111111111111111111111111111111111111111111") +
                        std::string("1111111111111111111111111111111111111111111111111111111111111111") +
                        std::string("1111111111111111111111111111111111111111111111111111111111111111") +
                        std::string("1111111111111111111111111111111111111111111111111111111111111111");

    std::string expected_s_bin = std::string("0011100010101001000110011011000100101110100011000011011101101011") +
                                 std::string("0000111011110011101111101100111000101111110110101111011101100011") +
                                 std::string("0111111111111111111111111111111111111111111111111111111111111111") +
                                 std::string("111111111111111111111111111111111111111111111111111111111111");

    to_array(s, s_bin);

    to_array(expected_s, expected_s_bin);

    SECTION("Verifying for A mod p holds (p = 2^252 + 27742317777372353535851937790883648493)") {
        reduce_exponent((unsigned char *) s.data());

        REQUIRE(s == expected_s);
    }
}