#include "sxt/base/bit/bit_position.h"

#include <vector>

#include "sxt/base/test/unit_test.h"

using namespace sxt;
using namespace sxt::basbt;

TEST_CASE("we can compute the positions of bits") {
  const size_t max_one_bits = 1'000;
  std::vector<unsigned> positions_data(max_one_bits);
  basct::span<unsigned> positions{positions_data};

  SECTION("we handle the case of no bits") {
    std::vector<uint8_t> blob;
    compute_bit_positions(positions, blob);
    REQUIRE(positions.empty());
  }

  SECTION("we handle a single bit") {
    std::vector<uint8_t> blob = {
        1,
    };
    compute_bit_positions(positions, blob);
    std::vector<unsigned> expected = {
        0,
    };
    positions_data.resize(positions.size());
    REQUIRE(positions_data == expected);
  }

  SECTION("we handle multiple bits") {
    std::vector<uint8_t> blob = {
        0b101,
    };
    compute_bit_positions(positions, blob);
    std::vector<unsigned> expected = {0, 2};
    positions_data.resize(positions.size());
    REQUIRE(positions_data == expected);
  }

  SECTION("we handle an a 64-bit integers") {
    std::vector<uint64_t> blob = {
        0b101,
        0b11,
    };
    compute_bit_positions(positions, basct::cspan<uint8_t>{
                                         reinterpret_cast<uint8_t*>(blob.data()),
                                         blob.size() * sizeof(blob[0]),
                                     });
    std::vector<unsigned> expected = {0, 2, 64, 65};
    positions_data.resize(positions.size());
    REQUIRE(positions_data == expected);
  }

  SECTION("we handle unaligned integers") {
    std::vector<uint64_t> blob = {
        0b101,
        0b11,
    };
    compute_bit_positions(positions, basct::cspan<uint8_t>{
                                         reinterpret_cast<uint8_t*>(blob.data()) + 1,
                                         blob.size() * sizeof(blob[0]) - 1,
                                     });
    std::vector<unsigned> expected = {64 - 8, 65 - 8};
    positions_data.resize(positions.size());
    REQUIRE(positions_data == expected);
  }
}
