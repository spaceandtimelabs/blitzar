load(
    "//bazel:sxt_component.bzl",
    "sxt_cc_component",
)

sxt_cc_component(
    name = "memory_utility",
    deps = [
        ":cuda_utility",        
    ],
    with_test = False,
)

sxt_cc_component(
    name = "stream",
    deps = [
        ":cuda_utility",
    ],
    with_test = False,
)

sxt_cc_component(
    name = "cuda_utility",
    deps = [
        "@local_config_cuda//cuda:cuda",
    ],
    with_test = False,
)
