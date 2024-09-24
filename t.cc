#include <iostream>

static int* f() {
  int res = 123;
  return &res;
}

int main() {
  std::cout << *f() << std::endl;
  return 0;
}
