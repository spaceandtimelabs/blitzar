load(
    "//bazel:sxt_build_system.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "async_device_resource",
    impl_deps = [
        "//sxt/base/error:panic",
    ],
    with_test = False,
    deps = [
        "//sxt/base/type:raw_stream",
    ],
)

sxt_cc_component(
    name = "device_resource",
    impl_deps = [
        "//sxt/base/error:panic",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    with_test = False,
)

sxt_cc_component(
    name = "pinned_resource",
    impl_deps = [
        "//sxt/base/error:panic",
    ],
    with_test = False,
)

sxt_cc_component(
    name = "managed_device_resource",
    impl_deps = [
        "//sxt/base/error:panic",
    ],
    with_test = False,
)

sxt_cc_component(
    name = "counting_resource",
    with_test = False,
)

sxt_cc_component(
    name = "chained_resource",
    impl_deps = [
        "//sxt/base/error:panic",
    ],
    test_deps = [
        ":counting_resource",
        "//sxt/base/test:unit_test",
    ],
)
