load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "elligator",
    impl_deps = [
        "//sxt/ristretto/base:sqrt_ratio_m1",
        "//sxt/curve21/type:element_p3",
        "//sxt/field51/constant:d",
        "//sxt/field51/constant:one",
        "//sxt/field51/constant:sqrtm1",
        "//sxt/field51/operation:abs",
        "//sxt/field51/operation:add",
        "//sxt/field51/operation:cmov",
        "//sxt/field51/operation:mul",
        "//sxt/field51/operation:neg",
        "//sxt/field51/operation:sub",
        "//sxt/field51/operation:sq",
        "//sxt/field51/type:element",
    ],
    is_cuda = True,
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
)

sxt_cc_component(
    name = "point_formation",
    impl_deps = [
        ":elligator",
        "//sxt/field51/type:element",
        "//sxt/curve21/operation:add",
        "//sxt/curve21/type:element_p3",
    ],
    is_cuda = True,
    test_deps = [
        "//sxt/field51/type:element",
        "//sxt/curve21/property:curve",
        "//sxt/curve21/type:element_p3",
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
)

sxt_cc_component(
    name = "add",
    impl_deps = [
        "//sxt/ristretto/type:compressed_element",
        "//sxt/curve21/operation:add",
        "//sxt/curve21/type:element_p3",
        "//sxt/ristretto/base:byte_conversion",
    ],
    is_cuda = True,
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
)

sxt_cc_component(
    name = "scalar_multiply",
    impl_deps = [
        ":add",
        "//sxt/ristretto/type:compressed_element",
        "//sxt/curve21/operation:scalar_multiply",
        "//sxt/curve21/type:element_p3",
        "//sxt/ristretto/base:byte_conversion",
    ],
    is_cuda = True,
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "//sxt/base/container:span",
        "//sxt/base/macro:cuda_callable",
    ],
)
