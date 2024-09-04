licenses(["notice"])  # Apache 2

cc_library(
    name = "fmtlib",
    hdrs = glob([
        "include/fmt/*.h",
    ]),
    defines = ["FMT_HEADER_ONLY"],
    includes = ["include"],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)
