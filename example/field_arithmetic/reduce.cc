#include <iostream>
#include <memory>

#include "sxt/field51/type/element.h"

#include "example/field_arithmetic/reduce1.h"

using namespace sxt;

int main() {
  int m = 1024;
  int n = 1000000;
  auto elements = new f51t::element[n];
  reduce1(elements, m, n);
  for (int i = 0; i < m; ++i) {
    std::cout << elements[i] << "\n";
  }
  delete[] elements;
  return 0;
}
