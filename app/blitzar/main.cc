#include <print>
#include <string_view>
#include <charconv>

//--------------------------------------------------------------------------------------------------
// make_partition_table 
//--------------------------------------------------------------------------------------------------
static void make_partition_table(std::string_view filename, unsigned n) noexcept {
  std::print("creating table {} {}\n", filename, n);
  (void)filename;
  (void)n;
}

//--------------------------------------------------------------------------------------------------
// main
//--------------------------------------------------------------------------------------------------
int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::print(stderr, "Usage: blitzar <cmd> <arg1> ... <argN>\n");
    return -1;
  }
  std::string_view cmd{argv[1]};
  if (cmd == "make-partition-table") {
    if (argc != 4) {
      std::print(stderr, "Usage: blitzar make-partition-table <filename> <n>\n");
      return -1;
    }
    std::string_view filename{argv[2]};
    std::string_view n_str{argv[3]};
    unsigned n;
    if (auto err = std::from_chars(n_str.begin(), n_str.end(), n); err.ec != std::errc{}) {
      std::print(stderr, "invalid number: {}\n", n_str);
      return -1;
    }
    make_partition_table(filename, n);
  }
  return 0;
}
