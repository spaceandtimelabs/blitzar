licenses(["notice"])  # Apache 2

cc_library(
    name = "libspdlog",
    hdrs = glob([
        "include/**/*.h",
    ]),
    srcs = glob([
      "src/**/*.cpp",
    ],
      exclude = [
        "src/details/os_windows.cpp",
        "src/sinks/wincolor_sink.cpp",
      ],
    ),
    copts = [
      "-std=c++2b",
    ],
    includes = ["include"],
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "@com_github_fmtlib_fmt//:fmt",
    ],
)
