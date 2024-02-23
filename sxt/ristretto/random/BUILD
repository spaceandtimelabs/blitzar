load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "element",
    impl_deps = [
        "//sxt/base/num:fast_random_number_generator",
        "//sxt/base/macro:cuda_warning",
        "//sxt/curve32/type:element_p3",
        "//sxt/field32/random:element",
        "//sxt/field32/type:element",
        "//sxt/ristretto/base:byte_conversion",
        "//sxt/ristretto/base:point_formation",
        "//sxt/ristretto/type:compressed_element",
    ],
    is_cuda = True,
    test_deps = [
        "//sxt/base/num:fast_random_number_generator",
        "//sxt/base/test:unit_test",
        "//sxt/curve32/property:curve",
    ],
    deps = [
        "//sxt/base/container:span",
        "//sxt/base/macro:cuda_callable",
    ],
)
