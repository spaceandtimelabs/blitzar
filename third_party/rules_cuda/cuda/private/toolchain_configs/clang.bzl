load("@bazel_skylib//lib:paths.bzl", "paths")
load("@bazel_tools//tools/cpp:toolchain_utils.bzl", "find_cpp_toolchain")
load("//cuda/private:action_names.bzl", "ACTION_NAMES")
load("//cuda/private:artifact_categories.bzl", "ARTIFACT_CATEGORIES")
load("//cuda/private:providers.bzl", "CudaToolchainConfigInfo", "CudaToolkitInfo")
load("//cuda/private:toolchain.bzl", "use_cpp_toolchain")
load(
    "//cuda/private:toolchain_config_lib.bzl",
    "action_config",
    "artifact_name_pattern",
    "env_entry",
    "env_set",
    "feature",
    "flag_group",
    "flag_set",
    "with_feature_set",
)

all_compile_actions = [
    ACTION_NAMES.cuda_compile,
]

all_link_actions = [
]

def _impl(ctx):
    is_windows = "windows" in ctx.var["TARGET_CPU"]

    obj_ext = ".obj" if is_windows else ".o"
    artifact_name_patterns = [
        # artifact_name_pattern for object files
        artifact_name_pattern(
            category_name = ARTIFACT_CATEGORIES.object_file,
            prefix = "",
            extension = obj_ext,
        ),
        artifact_name_pattern(
            category_name = ARTIFACT_CATEGORIES.pic_object_file,
            prefix = "",
            extension = ".pic" + obj_ext,
        ),
        artifact_name_pattern(
            category_name = ARTIFACT_CATEGORIES.rdc_object_file,
            prefix = "",
            extension = ".rdc" + obj_ext,
        ),
        artifact_name_pattern(
            category_name = ARTIFACT_CATEGORIES.rdc_pic_object_file,
            prefix = "",
            extension = ".rdc.pic" + obj_ext,
        ),
    ]

    cc_toolchain = find_cpp_toolchain(ctx)

    clang_compile_env_feature = feature(
        name = "clang_compile_env",
        enabled = True,
        env_sets = [
            env_set(
                actions = [
                    ACTION_NAMES.cuda_compile,
                    ACTION_NAMES.device_link,
                ],
                env_entries = [
                    env_entry("INCLUDE", ";".join(cc_toolchain.built_in_include_directories)),
                    env_entry("PATH", paths.dirname(cc_toolchain.compiler_executable) + ";C:/Windows/system32"),
                ],
            ),
        ],
    )

    cuda_compile_action = action_config(
        action_name = ACTION_NAMES.cuda_compile,
        flag_sets = [
            flag_set(flag_groups = [
                flag_group(flags = ["-x", "cu"]),
                flag_group(
                    iterate_over = "arch_specs",
                    flag_groups = [
                        flag_group(
                            iterate_over = "arch_specs.stage2_archs",
                            flag_groups = [
                                flag_group(
                                    expand_if_true = "arch_specs.stage2_archs.virtual",
                                    flags = ["--cuda-gpu-arch=sm_%{arch_specs.stage2_archs.arch}"],
                                ),
                                flag_group(
                                    expand_if_true = "arch_specs.stage2_archs.gpu",
                                    flags = ["--cuda-gpu-arch=sm_%{arch_specs.stage2_archs.arch}"],
                                ),
                                flag_group(
                                    expand_if_true = "arch_specs.stage2_archs.lto",
                                    flags = ["--cuda-gpu-arch=sm_%{arch_specs.stage2_archs.arch}"],
                                ),
                            ],
                        ),
                    ],
                ),
                flag_group(flags = ["-fcuda-rdc"], expand_if_true = "use_rdc"),
            ]),
        ],
        implies = [
            # TODO:
            "compiler_input_flags",
            "compiler_output_flags",
            "ptxas_flags",
        ],
    )

    cuda_path_feature = feature(
        name = "cuda_path",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.cuda_compile,
                ],
                flag_groups = [flag_group(flags = ["--cuda-path=" + ctx.attr.cuda_toolkit[CudaToolkitInfo].path])],
            ),
        ],
    )

    supports_compiler_device_link_feature = feature(
        name = "supports_compiler_device_link",
        enabled = False,
    )

    supports_wrapper_device_link_feature = feature(
        name = "supports_wrapper_device_link",
        enabled = True,
    )

    supports_pic_feature = feature(
        name = "supports_pic",
        enabled = True,
    )

    default_compile_flags_feature = feature(
        name = "default_compile_flags",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = all_compile_actions,
                flag_groups = ([
                    flag_group(
                        flags = [
                            "-U_FORTIFY_SOURCE",
                            "-fstack-protector",
                            "-Wall",
                            "-Wthread-safety",
                            "-Wself-assign",
                            "-Wunused-but-set-parameter",
                            "-Wno-free-nonheap-object",
                            "-fcolor-diagnostics",
                            "-fno-omit-frame-pointer",
                        ],
                    ),
                ]),
            ),
            flag_set(
                actions = all_compile_actions,
                flag_groups = ([
                    flag_group(
                        flags = ["-g"],
                    ),
                ]),
                with_features = [with_feature_set(features = ["dbg"])],
            ),
            flag_set(
                actions = all_compile_actions,
                flag_groups = ([
                    flag_group(
                        flags = [
                            "-g0",
                            "-O2",
                            "-D_FORTIFY_SOURCE=1",
                            "-DNDEBUG",
                            "-ffunction-sections",
                            "-fdata-sections",
                        ],
                    ),
                ]),
                with_features = [with_feature_set(features = ["opt"])],
            ),
        ],
    )

    dbg_feature = feature(name = "dbg")

    opt_feature = feature(name = "opt")

    sysroot_feature = feature(
        name = "sysroot",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.cuda_compile,
                ],
                flag_groups = [
                    flag_group(
                        flags = ["--sysroot=%{sysroot}"],
                        expand_if_available = "sysroot",
                    ),
                ],
            ),
        ],
    )

    compile_flags_feature = feature(
        name = "compile_flags",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.cuda_compile,
                    ACTION_NAMES.device_link,
                ],
                flag_groups = [
                    flag_group(
                        flags = ["%{compile_flags}"],
                        iterate_over = "compile_flags",
                        expand_if_available = "compile_flags",
                    ),
                ],
            ),
        ],
    )

    host_compile_flags_feature = feature(
        name = "host_compile_flags",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.cuda_compile,
                    ACTION_NAMES.device_link,
                ],
                flag_groups = [
                    flag_group(
                        flags = ["%{host_compile_flags}"],
                        iterate_over = "host_compile_flags",
                    ),
                ],
            ),
        ],
    )

    pic_feature = feature(
        name = "pic",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.cuda_compile,
                ],
                flag_groups = [
                    flag_group(flags = ["-fPIC"], expand_if_available = "use_pic"),
                ],
            ),
        ],
    )

    per_object_debug_info_feature = feature(
        name = "per_object_debug_info",
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.cuda_compile,
                ],
                flag_groups = [
                    flag_group(
                        flags = ["-gsplit-dwarf", "-g"],
                        expand_if_available = "per_object_debug_info_file",
                    ),
                ],
            ),
        ],
    )

    defines_feature = feature(
        name = "defines",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.cuda_compile,
                ],
                flag_groups = [
                    flag_group(
                        flags = ["-D%{defines}"],
                        iterate_over = "defines",
                    ),
                ],
            ),
        ],
    )

    random_seed_feature = feature(
        name = "random_seed",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.cuda_compile,
                ],
                flag_groups = [
                    flag_group(
                        flags = ["-frandom-seed=%{output_file}"],
                        expand_if_available = "output_file",
                    ),
                ],
            ),
        ],
    )

    includes_feature = feature(
        name = "includes",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.cuda_compile,
                ],
                flag_groups = [
                    flag_group(
                        flags = ["-include", "%{includes}"],
                        iterate_over = "includes",
                        expand_if_available = "includes",
                    ),
                ],
            ),
        ],
    )

    include_paths_feature = feature(
        name = "include_paths",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.cuda_compile,
                ],
                flag_groups = [
                    flag_group(
                        flags = ["-iquote", "%{quote_include_paths}"],
                        iterate_over = "quote_include_paths",
                    ),
                    flag_group(
                        flags = ["-I%{include_paths}"],
                        iterate_over = "include_paths",
                    ),
                    flag_group(
                        flags = ["-isystem", "%{system_include_paths}"],
                        iterate_over = "system_include_paths",
                    ),
                ],
            ),
        ],
    )

    external_include_paths_feature = feature(
        name = "external_include_paths",
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.cuda_compile,
                ],
                flag_groups = [
                    flag_group(
                        flags = ["-isystem", "%{external_include_paths}"],
                        iterate_over = "external_include_paths",
                        expand_if_available = "external_include_paths",
                    ),
                ],
            ),
        ],
    )

    strip_debug_symbols_feature = feature(
        name = "strip_debug_symbols",
        flag_sets = [
            flag_set(
                actions = all_link_actions,
                flag_groups = [
                    flag_group(
                        flags = ["-Wl,-S"],
                        expand_if_available = "strip_debug_symbols",
                    ),
                ],
            ),
        ],
    )

    dependency_file_feature = feature(
        name = "dependency_file",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.cuda_compile,
                ],
                flag_groups = [
                    flag_group(
                        flags = ["-MD", "-MF", "%{dependency_file}"],
                        expand_if_available = "dependency_file",
                    ),
                ],
            ),
        ],
    )

    ptxas_flags_feature = feature(
        name = "ptxas_flags",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.cuda_compile,
                ],
                flag_groups = [
                    flag_group(
                        flags = ["-Xcuda-ptxas", "%{ptxas_flags}"],
                        iterate_over = "ptxas_flags",
                    ),
                ],
            ),
        ],
    )

    compiler_input_flags_feature = feature(
        name = "compiler_input_flags",
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.cuda_compile,
                ],
                flag_groups = [flag_group(flags = ["-c", "%{source_file}"])],
            ),
        ],
    )

    compiler_output_flags_feature = feature(
        name = "compiler_output_flags",
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.cuda_compile,
                    ACTION_NAMES.device_link,
                ],
                flag_groups = [flag_group(flags = ["-o", "%{output_file}"])],
            ),
        ],
    )

    action_configs = [
        cuda_compile_action,
    ]

    features = [
        cuda_path_feature,
        supports_compiler_device_link_feature,
        supports_wrapper_device_link_feature,
        dependency_file_feature,
        random_seed_feature,
        per_object_debug_info_feature,
        defines_feature,
        includes_feature,
        include_paths_feature,
        external_include_paths_feature,
        strip_debug_symbols_feature,
        default_compile_flags_feature,
        dbg_feature,
        opt_feature,
        compile_flags_feature,
        host_compile_flags_feature,
        sysroot_feature,
        ptxas_flags_feature,
        compiler_input_flags_feature,
        compiler_output_flags_feature,
    ]

    if is_windows:
        features.extend([
            clang_compile_env_feature,
        ])
    else:
        features.extend([
            pic_feature,
            supports_pic_feature,
        ])

    return [CudaToolchainConfigInfo(
        action_configs = action_configs,
        features = features,
        artifact_name_patterns = artifact_name_patterns,
        toolchain_identifier = ctx.attr.toolchain_identifier,
        cuda_toolkit = ctx.attr.cuda_toolkit,
    )]

cuda_toolchain_config = rule(
    doc = """This rule provides the predefined cuda toolchain configuration for Clang.""",
    implementation = _impl,
    attrs = {
        "cuda_toolkit": attr.label(mandatory = True, providers = [CudaToolkitInfo], doc = "A target that provides a `CudaToolkitInfo`."),
        "toolchain_identifier": attr.string(values = ["clang"], mandatory = True),
        "_cc_toolchain": attr.label(default = "@bazel_tools//tools/cpp:current_cc_toolchain"),  # legacy behaviour
    },
    provides = [CudaToolchainConfigInfo],
    toolchains = use_cpp_toolchain(),
)
