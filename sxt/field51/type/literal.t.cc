#include "sxt/field51/type/literal.h"

#include <iostream>

using namespace sxt::f51t;

int main() {
  std::cout << "e1 = " << 0x0_f51 << std::endl;
  std::cout << "e2 = " << 0x1_f51 << std::endl;
  std::cout << "e3 = " << 0xa_f51 << std::endl;
  std::cout << "e4 = " << 0x10_f51 << std::endl;
  std::cout << "e6 = " << 0x3b86191f4f2865cc462f08daa6d911c0df283b53cb3b8f7d6027666f4c94e38_f51
            << std::endl;
  return 0;
}
