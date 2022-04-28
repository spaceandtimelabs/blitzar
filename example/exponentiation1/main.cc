#include <iostream>

#include "example/exponentiation1/exponentiate_cpu.h"
#include "example/exponentiation1/exponentiate_gpu.h"
#include "sxt/curve21/type/element_p3.h"
using namespace sxt;

int main() {
  const int n = 10;
  c21t::element_p3 elements_gpu[n];
  exponentiate_gpu(elements_gpu, n);

  c21t::element_p3 elements_cpu[n];
  exponentiate_cpu(elements_cpu, n);

  for (int i=0; i<n; ++i) {
    std::cout << elements_gpu[i] << " \t" << elements_cpu[i] << "\n";
  }
  return 0;
}
