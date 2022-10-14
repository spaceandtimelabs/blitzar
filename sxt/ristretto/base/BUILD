load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "byte_conversion",
    impl_deps = [
        ":sqrt_ratio_m1",
        "//sxt/curve21/type:element_p3",
        "//sxt/field51/type:element",
        "//sxt/field51/operation:add",
        "//sxt/field51/operation:sub",
        "//sxt/field51/operation:mul",
        "//sxt/field51/operation:sq",
        "//sxt/field51/operation:cmov",
        "//sxt/field51/operation:cneg",
        "//sxt/field51/operation:neg",
        "//sxt/field51/operation:abs",
        "//sxt/field51/operation:pow22523",
        "//sxt/field51/constant:d",
        "//sxt/field51/constant:one",
        "//sxt/field51/constant:sqrtm1",
        "//sxt/field51/constant:invsqrtamd",
        "//sxt/field51/property:sign",
        "//sxt/field51/property:zero",
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
    name = "sqrt_ratio_m1",
    impl_deps = [
        "//sxt/curve21/type:element_p3",
        "//sxt/field51/constant:sqrtm1",
        "//sxt/field51/operation:abs",
        "//sxt/field51/operation:add",
        "//sxt/field51/operation:cmov",
        "//sxt/field51/operation:mul",
        "//sxt/field51/operation:pow22523",
        "//sxt/field51/operation:sq",
        "//sxt/field51/operation:sub",
        "//sxt/field51/property:zero",
    ],
    is_cuda = True,
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "//sxt/base/macro:cuda_callable",
    ],
)
