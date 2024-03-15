#include <iostream>

#include "sxt/curve21/type/element_p3.h"
#include "sxt/curve21/type/literal.h"
#include "sxt/multiexp/pippenger2/multiexponentiation.h"
#include "sxt/memory/management/managed_array.h"
using namespace sxt;
using c21t::operator""_c21;

using E = c21t::element_p3;

static memmg::managed_array<E> make_lookup_array(unsigned n) {
  memmg::managed_array<E> res((1u << 16u) * n / 16u);
  auto e1 = 0x123_c21;
  auto e2 = 0x456_c21;
  for (unsigned i = 0; i < res.size(); ++i) {
    if (i % 2 == 0) {
      res[i] = e1;
    } else {
      res[i] = e2;
    }
  }
  return res;
}

int main() {
  std::cout << "arf\n";
  auto table = make_lookup_array(1024);
  std::cout << "nuf\n";
  return 0;
}
