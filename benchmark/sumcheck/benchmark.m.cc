#include <charconv>
#include <print>
#include <string_view>

#include "sxt/base/container/span.h"
#include "sxt/base/error/panic.h"
#include "sxt/base/num/fast_random_number_generator.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/scalar25/random/element.h"
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

  std::string_view s;

  // n
  s = {argv[1]};
  if (std::from_chars(s.begin(), s.end(), p.n).ec != std::errc{}) {
    baser::panic("invalid argument for n: {}\n", s);
  }

  // degree
  s = {argv[2]};
  if (std::from_chars(s.begin(), s.end(), p.degree).ec != std::errc{}) {
    baser::panic("invalid argument for degree: {}\n", s);
  }

  // num_products
  s = {argv[3]};
  if (std::from_chars(s.begin(), s.end(), p.num_products).ec != std::errc{}) {
    baser::panic("invalid argument for num_products: {}\n", s);
  }

  // num_samples
  s = {argv[4]};
  if (std::from_chars(s.begin(), s.end(), p.num_samples).ec != std::errc{}) {
    baser::panic("invalid argument for num_samples: {}\n", s);
  }
}

int main(int argc, char* argv[]) {
  params p;
  read_params(p, argc, argv);

  basn::fast_random_number_generator rng{1, 2};

  // mles
  memmg::managed_array<s25t::element> mles(p.n * p.degree * p.num_products);
  s25rn::generate_random_elements(mles, rng);

  return 0;
}
