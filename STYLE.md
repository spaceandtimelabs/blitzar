## Organization
The project's structure largely follows the conventions from John Lakos' book 
"Large Scale C++ Software Design" ([1], [2]).

All library code belongs to a component; all components belong to a package;
all packages belong to a package group; and there are no dependency cycles
between components, packages, and package groups (§2.1 of [1]). For example,

```
sxt/base/iterator/counting_iterator.{h,cc,t.cc}
```
are the header, source, and unit tests for the component "counting_iterator". "counting_iterator"
belongs to the package "iterator", and "iterator" belongs to the package group "base".

If another component `sxt/multiexp/pippenger_multiprod/multiproduct` depends on
`base/iterator/counting_iterator`, then no component in the package group "base"
can depend on a component in the package group "multiexp" as that would introduce
a cycle (§3.5 of [1], [3])

All packages are given a unique namespace. For example, all components within
the package `base/iterator` use the namespace "basit". The first three letters
of a package namespace uniquely identify the package group (e.g. "bas"
identifies the package group "base"); following letters uniquely identify the
package within the package group (§2.10 of [1]). Package namespace names don't have to
be descriptive, but should be short and satisfy the uniqueness requirements.

## Naming
We follow the C++ standard library naming convention. Class names, variables, 
functions, and concepts use [snake case](https://en.wikipedia.org/wiki/Snake_case).

Class member names are suffixed with an underscore:
```
class abc {
 private:
   int x_;
};
```

Template parameters use [camel case](https://en.wikipedia.org/wiki/Camel_case) with the
first letter capitalized:
```
template <class MyType>
void f(MyType t) {
    // ...
}
```

## Error handling
Our error handling guidelines are inspired by Envoy's style guide ([4]).

Error codes and exceptions should be used to handle normal control flow. Crashing is a valid
error handling strategy and should be used for errors not considered part of normal control flow
([5]).

To make errors more explicit, we use noexcept for functions that either don't throw an exception or
would only throw exceptions outside of normal control flow (Item 14 of [6]). For example,

```
std::vector<int> copy_and_sort(const std::vector<int>& xv) noexcept {
  std::vector<int> res{xv.begin(), xv.end()};
  std::sort(res.begin(), res.end());
  return res;
}
```
copy_and_sort might throw std::bad_alloc and noexcept will cause the function to terminate; 
but such an error would be outside of normal control flow, so termination is acceptable.

## Memory Management
We make extensive use of RAII and allocator-aware containers to manage device memory and 
achieve certain host-side optimizations.

See [9], [10], and [11].

## Futures and Promises
In order to get the most out of GPUs and eventually scale to using multiple GPUs, we use
the asynchronous versions of CUDA API functions.

To make writing async code easier, we adopt the future-promise design from
[seastar.io](https://seastar.io/) ([12]) and use c++20 coroutines ([13]).

## Testing
We try to follow the guidelines from Kevlin Henney's talk "Structure and Interpretation of Test Cases"
([7], [8]).

In addition to checking correctness, tests also serve as documentation and
should be readable and describe code's behavior.

## References
1: John Lakos. 2019. [Large-scale C++ software design, Volume 1](https://www.amazon.com/Large-Scale-Architecture-Addison-Wesley-Professional-Computing/dp/0201717069/ref=sr_1_fkmr0_1?crid=1K4S108K8A8DU&keywords=large+scale+c%2B%2B+design+2nd&qid=1684861966&sprefix=large+scale+c%2B%2B+design+2nd%2Caps%2C162&sr=8-1-fkmr0&ufe=app_do%3Aamzn1.fos.006c50ae-5d4c-4777-9bc0-4513d670b6bc).

2: John Lakos. [Overview of [1]](https://youtu.be/d3zMfMC8l5U).

3: John Lakos. [Advanced Levelization Techniques](https://youtu.be/QjFpKJ8Xx78).

4: https://github.com/envoyproxy/envoy/blob/main/STYLE.md#error-handling

5: Matt Klein. [Crash early and crash often for more reliable software](https://medium.com/@mattklein123/crash-early-and-crash-often-for-more-reliable-software-597738dd21c5)

6: Scott Meyers. [Effective Modern C++](https://www.amazon.com/Effective-Modern-Specific-Ways-Improve/dp/1491903996?asin=1491903996&revisionId=&format=4&depth=1).

7: Kevlin Henney. [Structure and Interpretation of Test Cases](https://youtu.be/tWn8RA_DEic).

8: Kevlin Henney. [Programming with GUTs](https://youtu.be/azoucC_fwzw).

9: Pablo Halpern, John Lakos. [Value Proposition: Allocator-Aware (AA) Software](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2020/p2035r0.pdf).

10: John Lakos. [Local ('Arena') Memory Allocators](https://youtu.be/nZNd5FjSquk).

11: Pablo Halpern. [Allocators: The Good Parts](https://youtu.be/v3dz-AKOVL8).

12: Avi Kivity. [Building efficient I/O intensive applications with Seastar](https://youtu.be/p8d28t4qCTY).

13: Gor Nishanov. [C++ Coroutines: Under the covers](https://youtu.be/8C8NnE1Dg4A).
