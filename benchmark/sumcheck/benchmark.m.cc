#include <print>

int main(int argc, char* argv[]) {
  if (argc != 5) {
    std::println("Usage: benchmark <n> <degree> <num_products> <num_samples>");
  }
  (void)argc;
  (void)argv;
  return 0;
}
