#include <print>
#include <random>

#include "sxt/memory/management/managed_array.h"
using namespace sxt;

int main() {
  const unsigned n = 1'000'00;
  const unsigned m = 32;
  memmg::managed_array<double> data(n * m);
  (void)data;

  // 1 call repeat copies from ordinary memory to device
  // 2 copy to contiguous paged memory, copy from paged memory to device
  // 3 like 2, but use chunks
  std::println("arf");
  return 0;
}
