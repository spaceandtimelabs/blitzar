load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "managed_array_fwd",
    with_test = False,
)

sxt_cc_component(
    name = "managed_array",
    test_deps = [
        "//sxt/base/test:allocator_aware",
        "//sxt/base/test:unit_test",
    ],
    deps = [
        ":managed_array_fwd",
        "//sxt/base/container:span_void",
        "//sxt/base/memory:alloc",
    ],
)
