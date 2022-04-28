load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "managed_array_fwd",
    with_test = False,
)

sxt_cc_component(
    name = "managed_array",
    deps = [
        ":managed_array_fwd",
        "//sxt/base/type:polymorphic_allocator",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
)
