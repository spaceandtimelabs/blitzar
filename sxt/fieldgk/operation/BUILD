load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "add",
    test_deps = [
        "//sxt/base/num:fast_random_number_generator",
        "//sxt/base/test:unit_test",
        "//sxt/fieldgk/constant:one",
        "//sxt/fieldgk/constant:zero",
        "//sxt/fieldgk/random:element",
        "//sxt/fieldgk/type:literal",
    ],
    deps = [
        "//sxt/base/field:arithmetic_utility",
        "//sxt/base/macro:cuda_callable",
        "//sxt/fieldgk/base:constants",
        "//sxt/fieldgk/base:subtract_p",
        "//sxt/fieldgk/type:element",
    ],
)

sxt_cc_component(
    name = "cmov",
    impl_deps = [
        "//sxt/fieldgk/type:element",
    ],
    is_cuda = True,
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/fieldgk/constant:one",
        "//sxt/fieldgk/constant:zero",
        "//sxt/fieldgk/type:element",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
)

sxt_cc_component(
    name = "invert",
    impl_deps = [
        ":pow_vartime",
        "//sxt/fieldgk/property:zero",
        "//sxt/fieldgk/type:element",
    ],
    is_cuda = True,
    test_deps = [
        ":mul",
        "//sxt/base/num:fast_random_number_generator",
        "//sxt/base/test:unit_test",
        "//sxt/fieldgk/constant:one",
        "//sxt/fieldgk/random:element",
        "//sxt/fieldgk/type:element",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
)

sxt_cc_component(
    name = "mul",
    impl_deps = [
        "//sxt/base/field:arithmetic_utility",
        "//sxt/fieldgk/base:reduce",
        "//sxt/fieldgk/type:element",
    ],
    is_cuda = True,
    test_deps = [
        "//sxt/base/num:fast_random_number_generator",
        "//sxt/base/test:unit_test",
        "//sxt/fieldgk/base:constants",
        "//sxt/fieldgk/base:montgomery",
        "//sxt/fieldgk/constant:zero",
        "//sxt/fieldgk/random:element",
        "//sxt/fieldgk/type:element",
        "//sxt/fieldgk/type:literal",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
)

sxt_cc_component(
  name = "muladd",
  with_test = False,
  deps = [
    ":add",
    ":mul",
    "//sxt/base/macro:cuda_callable",
  ],
)

sxt_cc_component(
    name = "neg",
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/fieldgk/base:constants",
        "//sxt/fieldgk/constant:one",
        "//sxt/fieldgk/constant:zero",
        "//sxt/fieldgk/type:element",
        "//sxt/fieldgk/type:literal",
    ],
    deps = [
        "//sxt/base/field:arithmetic_utility",
        "//sxt/base/macro:cuda_callable",
        "//sxt/fieldgk/base:constants",
        "//sxt/fieldgk/type:element",
    ],
)

sxt_cc_component(
    name = "pow_vartime",
    impl_deps = [
        ":mul",
        ":square",
        "//sxt/fieldgk/constant:one",
        "//sxt/fieldgk/type:element",
    ],
    is_cuda = True,
    with_test = False,
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
)

sxt_cc_component(
    name = "square",
    impl_deps = [
        "//sxt/base/field:arithmetic_utility",
        "//sxt/fieldgk/base:reduce",
        "//sxt/fieldgk/type:element",
    ],
    is_cuda = True,
    test_deps = [
        ":mul",
        "//sxt/base/num:fast_random_number_generator",
        "//sxt/base/test:unit_test",
        "//sxt/fieldgk/random:element",
        "//sxt/fieldgk/type:element",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
)

sxt_cc_component(
    name = "sub",
    test_deps = [
        "//sxt/base/num:fast_random_number_generator",
        "//sxt/base/test:unit_test",
        "//sxt/fieldgk/constant:one",
        "//sxt/fieldgk/constant:zero",
        "//sxt/fieldgk/random:element",
        "//sxt/fieldgk/type:literal",
    ],
    deps = [
        ":add",
        ":neg",
        "//sxt/base/macro:cuda_callable",
        "//sxt/fieldgk/type:element",
    ],
)
