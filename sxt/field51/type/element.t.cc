#include "sxt/field51/type/element.h"

#include <iostream>
using namespace sxt::f51t;

int main() {
  element e1{0, 0, 0, 0, 0};
  std::cout << "e1 = " << e1 << "\n";

  element e2{1, 0, 0, 0, 0};
  std::cout << "e2 = " << e2 << "\n";

  element e3{10, 0, 0, 0, 0};
  std::cout << "e3 = " << e3 << "\n";

  element e4{16, 0, 0, 0, 0};
  std::cout << "e4 = " << e4 << "\n";

  element e5{0x100, 0, 0, 0, 0};
  std::cout << "e5 = " << e5 << "\n";
  return 0;
}
