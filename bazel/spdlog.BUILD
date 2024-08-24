licenses(["notice"])  # Apache 2

cc_library(
    name = "libspdlog",
    hdrs = glob([
        "include/**/*.h",
    ]),
    copts = [
        "-stdlib=libc++",
        "-std=c++20",
    ],
    linkstatic = 1,
    includes = ["include"],
    visibility = ["//visibility:public"],
)
