load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "compressed_element",
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
)

sxt_cc_component(
    name = "literal",
    test_deps = [
        "//sxt/base/test:unit_test",
        "//sxt/curve21/property:curve",
    ],
    deps = [
        ":compressed_element",
        "//sxt/base/type:literal",
        "//sxt/curve21/type:element_p3",
        "//sxt/field51/base:byte_conversion",
        "//sxt/field51/type:element",
        "//sxt/ristretto/base:byte_conversion",
        "//sxt/ristretto/base:point_formation",
    ],
)
