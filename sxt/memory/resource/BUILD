load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "async_device_resource",
    impl_deps = [
        "//sxt/base/error:panic",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    with_test = False,
    deps = [
        "//sxt/base/type:raw_stream",
        "@rules_cuda//cuda:runtime",
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
    deps = [
        "@rules_cuda//cuda:runtime",
    ],
)

sxt_cc_component(
    name = "pinned_resource",
    impl_deps = [
        "//sxt/base/error:panic",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    with_test = False,
    deps = [
        "@rules_cuda//cuda:runtime",
    ],
)

sxt_cc_component(
    name = "managed_device_resource",
    impl_deps = [
        "//sxt/base/error:panic",
    ],
    test_deps = [
        "//sxt/base/test:unit_test",
    ],
    deps = [
        "@rules_cuda//cuda:runtime",
    ],
)
