#include <print>
#include <string_view>
#include <charconv>
#include <vector>

#include "sxt/base/num/divide_up.h"
#include "sxt/base/num/fast_random_number_generator.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/ristretto/random/element.h"
using namespace sxt;

//--------------------------------------------------------------------------------------------------
// make_partition_table 
//--------------------------------------------------------------------------------------------------
static void make_partition_table(std::string_view filename, unsigned n) noexcept {
  n = basn::divide_up(n, 16u) * 16u;
  std::print("creating table {} {}\n", filename, n);

  // make generators
  basn::fast_random_number_generator rng{1u, 2u};
  std::vector<c21t::element_p3> generators(n);
  rstrn::generate_random_elements(generators, rng);

  // compute precomputed partition values
  // write table
  (void)filename;
  (void)n;
}

//--------------------------------------------------------------------------------------------------
// main
//--------------------------------------------------------------------------------------------------
int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::print(stderr, "Usage: blitzar <cmd> <arg1> ... <argN>\n");
    return -1;
  }
  std::string_view cmd{argv[1]};
  if (cmd == "make-partition-table") {
    if (argc != 4) {
      std::print(stderr, "Usage: blitzar make-partition-table <filename> <n>\n");
      return -1;
    }
    std::string_view filename{argv[2]};
    std::string_view n_str{argv[3]};
    unsigned n;
    if (auto err = std::from_chars(n_str.begin(), n_str.end(), n); err.ec != std::errc{}) {
      std::print(stderr, "invalid number: {}\n", n_str);
      return -1;
    }
    make_partition_table(filename, n);
  }
  return 0;
}
