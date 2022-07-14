#include "sxt/base/iterator/iterator_facade.h"

#include <type_traits>

#include "sxt/base/test/unit_test.h"
using namespace sxt::basit;

namespace {
struct my_int {
  operator int() const noexcept { return this->val; }

  int val;
};

class test_iterator final : public iterator_facade<test_iterator> {
 public:
  using pointer = const int*;
  using reference = const int&;

  explicit test_iterator(int x = 0) noexcept : x_{x} {}

  bool equal_to(const test_iterator& other) const noexcept {
    return x_ == other.x_;
  }

  my_int dereference() const noexcept {
    return my_int{x_};
  }

  void advance(ptrdiff_t delta) noexcept { x_ += delta; }

  int distance_to(test_iterator rhs) const noexcept {
    return rhs.x_ - x_;
  }

 private:
   int x_;
};
} // namespace

namespace std {
template <>
struct iterator_traits<test_iterator> : iterator_traits_impl<test_iterator> {};
}  // namespace std

TEST_CASE("iterator_facade can be used to easily create custom iterators") {
  test_iterator iter;

  SECTION("we can query with the standard std::iterator_traits") {
    REQUIRE(std::is_same_v<std::iterator_traits<test_iterator>::difference_type,
                           int>);
    REQUIRE(std::is_same_v<std::iterator_traits<test_iterator>::value_type,
                           my_int>);
    REQUIRE(std::is_same_v<std::iterator_traits<test_iterator>::reference,
                           const int&>);
    REQUIRE(std::is_same_v<std::iterator_traits<test_iterator>::pointer,
                           const int*>);
    REQUIRE(
        std::is_same_v<std::iterator_traits<test_iterator>::iterator_category,
                       std::random_access_iterator_tag>);
  }

  SECTION("we can dereference iterators") {
    REQUIRE(*iter == 0);
    iter = test_iterator{10};
    REQUIRE(*iter == 10);
  }

  SECTION("we can use the arrow operator with iterators") {
    REQUIRE(iter->val == 0);
  }

  SECTION("we can increment iterators") {
    iter++;
    REQUIRE(*iter == 1);
    REQUIRE(*++iter == 2);
  }

  SECTION("we can decrement iterators") {
    iter--;
    REQUIRE(*iter == -1);
    REQUIRE(*--iter == -2);
  }

  SECTION("we can use integral values to change iterators") {
    auto iter2 = iter + 5;
    REQUIRE(*iter2 == 5);

    iter2 = iter - 5;
    REQUIRE(*iter2 == -5);

    iter += 5;
    REQUIRE(*iter == 5);

    iter -= 10;
    REQUIRE(*iter == -5);
  }

  SECTION("we can compare iterators") {
    test_iterator iter2{2};

    REQUIRE(iter == iter);
    REQUIRE(!(iter == iter2));
    REQUIRE(iter != iter2);
    REQUIRE(!(iter != iter));

    REQUIRE(iter <= iter);
    REQUIRE(iter < iter2);
    REQUIRE(!(iter < iter));

    REQUIRE(iter >= iter);
    REQUIRE(iter2 > iter);
    REQUIRE(!(iter > iter));
  }

  SECTION("we can compute the distance between iterators") {
    test_iterator iter2;
    REQUIRE(iter2 - iter == 0);

    iter2 = test_iterator{3};
    REQUIRE(iter2 - iter == 3);
  }

  SECTION("we can use the bracket operator with oterators") {
    REQUIRE(iter[3] == 3);
  }
}
