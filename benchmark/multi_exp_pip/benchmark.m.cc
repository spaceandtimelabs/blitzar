#include <print>
#include <string_view>
#include <charconv>

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::println("Usage: benchmark <num_outputs> <n>");
    return -1;
  }
  std::string_view num_outputs_str{argv[1]};
  std::string_view n_str{argv[2]};
  unsigned num_outputs, n;
  if (std::from_chars(num_outputs_str.begin(), num_outputs_str.end(), num_outputs).ec !=
      std::errc{}) {
    std::println("invalid argument: {}\n", num_outputs_str);
    return -1;
  }
  if (std::from_chars(n_str.begin(), n_str.end(), n).ec != std::errc{}) {
    std::println("invalid argument: {}\n", n_str);
    return -1;
  }
  (void)argc;
  (void)argv;
  std::println("arf");
  return 0;
}
