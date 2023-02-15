load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "element",
    impl_deps = [
        "//sxt/base/num:fast_random_number_generator",
        "//sxt/base/macro:cuda_warning",
        "//sxt/curve21/type:element_p3",
        "//sxt/field51/random:element",
        "//sxt/field51/type:element",
        "//sxt/ristretto/base:byte_conversion",
        "//sxt/ristretto/base:point_formation",
        "//sxt/ristretto/type:compressed_element",
    ],
    is_cuda = True,
    test_deps = [
        "//sxt/base/num:fast_random_number_generator",
        "//sxt/curve21/property:curve",
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "//sxt/base/container:span",
        "//sxt/base/macro:cuda_callable",
    ],
)
