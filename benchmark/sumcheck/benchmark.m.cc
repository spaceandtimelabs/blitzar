#include <charconv>
#include <print>
#include <string_view>

#include "sxt/base/error/panic.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/scalar25/type/element.h"
using namespace sxt;

struct params {
  unsigned n;
  unsigned degree;
  unsigned num_products;
  unsigned num_samples;
};

static void read_params(params& p, int argc, char* argv[]) noexcept {
  if (argc != 5) {
    baser::panic("Usage: benchmark <n> <degree> <num_products> <num_samples>");
  }
  std::string_view s{argv[1]};
  if (std::from_chars(s.begin(), s.end(), p.n).ec != std::errc{}) {
    baser::panic("invalid argument for n: {}\n", s);
  }
  (void)p;
  (void)argv;
}

int main(int argc, char* argv[]) {
  params p;
  read_params(p, argc, argv);

  // mles
  memmg::managed_array<s25t::element> mles;
  (void)argc;
  (void)argv;
  return 0;
}
