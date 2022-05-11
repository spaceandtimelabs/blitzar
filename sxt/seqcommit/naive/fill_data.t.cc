#include "sxt/seqcommit/naive/fill_data.h"

#include "sxt/base/test/unit_test.h"

#include <array>

using namespace sxt;
using namespace sxt::sqcnv;

TEST_CASE("Test 1 - We can verify that fill_data does not change input array A when A < p") {
    // p = 2^252 + 27742317777372353535851937790883648493 (decimal)
    // s = p - 1 (decimal)
    // s % p = p - 1 (decimal)
    std::array<unsigned long long, 4> data_array = {6346243789798364140ull, 1503914060200516822ull, 0ull, 1152921504606846976ull};

    SECTION("Verifying for A mod p == A holds (p = 2^252 + 27742317777372353535851937790883648493)") {
        std::array<unsigned long long, 4> s;
        std::array<unsigned long long, 4> expected_s = data_array;

        fill_data((unsigned char *) s.data(), (unsigned char *) data_array.data(), sizeof(data_array));

        REQUIRE(s == expected_s);
    }
}

TEST_CASE("Test 2 - We can verify that fill_data change input array A when A >= p") {
    // p = 2^252 + 27742317777372353535851937790883648493 (decimal)
    // s = p (decimal)
    // s % p = 0 (decimal)
    std::array<unsigned long long, 4> data_array = {6346243789798364141ull, 1503914060200516822ull, 0ull, 1152921504606846976ull};

    SECTION("Verifying for A mod p == 0 holds (p = 2^252 + 27742317777372353535851937790883648493)") {
        std::array<unsigned long long, 4> s;

        fill_data((unsigned char *) s.data(), (unsigned char *) data_array.data(), sizeof(data_array));

        REQUIRE(s == data_array);
    }
}

TEST_CASE("Test 3 - We can verify that fill_data change input array A when A >= p") {
    // p = 2^252 + 27742317777372353535851937790883648493 (decimal)
    // s = p + 103 (decimal)
    // s % p = 103 (decimal)
    std::array<unsigned long long, 4> data_array = {6346243789798364244ull, 1503914060200516822ull, 0ull, 1152921504606846976ull};

    SECTION("Verifying for A mod p == 103 holds (p = 2^252 + 27742317777372353535851937790883648493)") {
        std::array<unsigned long long, 4> s;

        fill_data((unsigned char *) s.data(), (unsigned char *) data_array.data(), sizeof(data_array));

        REQUIRE(s == data_array);
    }
}

TEST_CASE("Test 4 - We can verify that fill_data correctly change input array A even when A is the maximum possible value") {
    // p = 2^252 + 27742317777372353535851937790883648493 (decimal)
    // s =     115792089237316195423570985008687907853269984665640564039457584007913129639935 (decimal)
    // s % p = 7237005577332262213973186563042994240413239274941949949428319933631315875100 (decimal)
    std::array<unsigned long long, 4> data_array = {18446744073709551615ull, 18446744073709551615ull, 18446744073709551615ull, 18446744073709551615ull};
    std::array<unsigned long long, 4> expected_s = {15486807595281847580ull, 14334777244411350896ull, 18446744073709551614ull, 1152921504606846975ull};

    SECTION("Verifying for A mod p holds (p = 2^252 + 27742317777372353535851937790883648493)") {
        std::array<unsigned long long, 4> s;

        fill_data((unsigned char *) s.data(), (unsigned char *) data_array.data(), sizeof(data_array));

        REQUIRE(s == expected_s);
    }
}

TEST_CASE("Test 5 - We can verify that fill_data correctly transforms 8 bytes input into a 32 bytes output") {
    std::array<unsigned long long, 1> data_array = {6346243789798364244ull};

    SECTION("Verifying if 8 bytes is correctly transformed into 32 bytes") {
        std::array<unsigned long long, 4> s;

        fill_data((unsigned char *) s.data(), (unsigned char *) data_array.data(), sizeof(data_array));

        REQUIRE(s == std::array<unsigned long long, 4>({data_array[0], 0, 0, 0}));
    }
}

TEST_CASE("Test 6 - We can verify that fill_data correctly transforms 16 bytes input into a 32 bytes output") {
    std::array<unsigned long long, 2> data_array = {6346243789798364244ull, 1503914060200516822ull};

    SECTION("Verifying if 16 bytes is correctly transformed into 32 bytes") {
        std::array<unsigned long long, 4> s;

        fill_data((unsigned char *) s.data(), (unsigned char *) data_array.data(), sizeof(data_array));

        REQUIRE(s == std::array<unsigned long long, 4>({data_array[0], data_array[1], 0, 0}));
    }
}

TEST_CASE("Test 7 - We can verify that fill_data correctly transforms 24 bytes input into a 32 bytes output") {
    std::array<unsigned long long, 3> data_array = {6346243789798364244ull, 1503914060200516822ull, 18446744073709551615ull};

    SECTION("Verifying if 24 bytes is correctly transformed into 32 bytes") {
        std::array<unsigned long long, 4> s;

        fill_data((unsigned char *) s.data(), (unsigned char *) data_array.data(), sizeof(data_array));

        REQUIRE(s == std::array<unsigned long long, 4>({data_array[0], data_array[1], data_array[2], 0}));
    }
}

TEST_CASE("Test 8 - We can verify that fill_data does not reduce the input array when data_aray[31] = 127 (the maximum value") {
    std::array<unsigned char, 32> data_array = {
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,127,
    };

    SECTION("Verifying if 24 bytes is correctly transformed into 32 bytes") {
        std::array<unsigned char, 32> s;

        fill_data(s.data(), data_array.data(), sizeof(data_array));

        REQUIRE(s == data_array);
    }
}

TEST_CASE("Test 9 - We can verify that fill_data correctly reduces the input array when data_aray[31] > 127") {
    std::array<unsigned char, 32> data_array = {
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,128,
    };

    SECTION("Verifying if 24 bytes is correctly transformed into 32 bytes") {
        std::array<unsigned char, 32> s;

        fill_data(s.data(), data_array.data(), sizeof(data_array));

        REQUIRE(s != data_array);
    }
}