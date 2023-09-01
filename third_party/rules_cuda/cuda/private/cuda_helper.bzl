"""private helpers"""

load("@bazel_skylib//lib:paths.bzl", "paths")
load("@bazel_skylib//lib:types.bzl", "types")
load("@bazel_skylib//rules:common_settings.bzl", "BuildSettingInfo")
load("//cuda/private:action_names.bzl", "ACTION_NAMES")
load("//cuda/private:artifact_categories.bzl", "ARTIFACT_CATEGORIES")
load("//cuda/private:providers.bzl", "ArchSpecInfo", "CudaArchsInfo", "CudaInfo", "Stage2ArchInfo", "cuda_archs")
load("//cuda/private:rules/common.bzl", "ALLOW_CUDA_HDRS")
load("//cuda/private:toolchain_config_lib.bzl", "config_helper", "unique")

def _create_arch_number(arch_num_str):
    """Create a structured architecture number.

    It is encoded as a tuple (int, suffix_str) to ease the comparison."""
    if types.is_int(arch_num_str):
        return (int(arch_num_str), "")
    else:
        for i, c in enumerate(arch_num_str.elems()):
            if not c.isdigit():
                break
        return (int(arch_num_str[:i]), arch_num_str[i:])

def _format_arch_number(arch_num):
    return str(arch_num[0]) + arch_num[1]

def _get_arch_number(arch_str):
    arch_str = arch_str.strip()
    arch_num_str = None
    if arch_str.startswith("compute_"):
        arch_num_str = arch_str[len("compute_"):]
    elif arch_str.startswith("lto_"):
        arch_num_str = arch_str[len("lto_"):]
    elif arch_str.startswith("sm_"):
        arch_num_str = arch_str[len("sm_"):]
    if arch_num_str not in cuda_archs:
        fail("{} is not a supported cuda arch".format(arch_str))
    return _create_arch_number(arch_num_str)

def _get_stage2_arch_info(code_str):
    return Stage2ArchInfo(
        arch = _format_arch_number(_get_arch_number(code_str)),
        virtual = code_str.startswith("compute_"),
        gpu = code_str.startswith("sm_"),
        lto = code_str.startswith("lto_"),
    )

def _get_arch_spec(spec_str):
    '''Convert string into an ArchSpecInfo.

    aka, parse "compute_80:sm_80,sm_86"'''
    spec_str = spec_str.strip()
    if spec_str == "":
        return None

    stage1_arch = None
    stage2_archs = []

    virt = None  # stage1 str
    codes = None  # stage2 str
    virtual_codes = spec_str.split(":")
    if len(virtual_codes) == 2:
        virt, codes = virtual_codes
        codes = codes.split(",")
        if not virt.startswith("compute_"):
            fail("expect a virtual architecture, got", virt)
        stage1_arch = _format_arch_number(_get_arch_number(virt))
        stage2_archs = [_get_stage2_arch_info(code) for code in codes]
    else:
        (codes,) = virtual_codes
        codes = codes.split(",")
        stage1_arch = _format_arch_number(min([_get_arch_number(c) for c in codes]))
        stage2_archs = [_get_stage2_arch_info(code) for code in codes]
    arch_spec = ArchSpecInfo(stage1_arch = stage1_arch, stage2_archs = stage2_archs)
    return arch_spec

def _get_arch_specs(specs_str):
    """Convert string into a list of ArchSpecInfo.

    Args:
        specs_str: a string to be parsed, e.g., "compute_70:sm_70;compute_80:sm_80,sm_86".

    Returns:
        A list of `ArchSpecInfo`s
    """
    archs = []
    for sepc_str in specs_str.split(";"):
        spec = _get_arch_spec(sepc_str)
        if spec != None:
            archs.append(spec)
    return archs

def _check_src_extension(file, allowed_src_files):
    for pattern in allowed_src_files:
        if file.basename.endswith(pattern):
            return True
    return False

def _check_srcs_extensions(ctx, allowed_src_files, rule_name):
    """Ensure ctx.attr.srcs is valid."""
    for src in ctx.attr.srcs:
        files = src[DefaultInfo].files.to_list()
        if len(files) == 1 and files[0].is_source:
            if not _check_src_extension(files[0], allowed_src_files) and not files[0].is_directory:
                fail("in srcs attribute of {} rule {}: source file '{}' is misplaced here".format(rule_name, ctx.label, str(src.label)))
        else:
            at_least_one_good = False
            for file in files:
                if _check_src_extension(file, allowed_src_files) or file.is_directory:
                    at_least_one_good = True
                    break
            if not at_least_one_good:
                fail("'{}' does not produce any {} srcs files".format(str(src.label), rule_name), attr = "srcs")

def _get_basename_without_ext(basename, allow_exts, fail_if_not_match = True):
    for ext in sorted(allow_exts, key = len, reverse = True):
        if basename.endswith(ext):
            return basename[:-len(ext)]
    if fail_if_not_match:
        fail("'{}' does not have valid extension, allowed extension(s): {}".format(basename, allow_exts))
    else:
        return None

def _resolve_workspace_root_includes(ctx):
    src_path = paths.normalize(ctx.label.workspace_root)
    bin_path = paths.normalize(paths.join(ctx.bin_dir.path, src_path))
    return src_path, bin_path

def _resolve_includes(ctx, path):
    if paths.is_absolute(path):
        fail("invalid absolute path", path)

    src_path = paths.normalize(paths.join(ctx.label.workspace_root, ctx.label.package, path))
    bin_path = paths.join(ctx.bin_dir.path, src_path)
    return src_path, bin_path

def _check_opts(opt):
    opt = opt.strip()
    disallow_list_exact = [
        "--cuda",
        "-cuda",
        "--preprocess",
        "-E",
        "--compile",
        "-c",
        "--cubin",
        "-cubin",
        "--ptx",
        "-ptx",
        "--fatbin",
        "-fatbin",
        "--device-link",
        "-dlink",
        "--lib",
        "-lib",
        "--generate-dependencies",
        "-M",
        "--generate-nonsystem-dependencies",
        "-MM",
        "--run",
        "-run",
    ]
    if (opt.startswith("--generate-code") or opt.startswith("-gencode") or
        opt.startswith("--gpu-architecture") or opt.startswith("-arch") or
        opt.startswith("--gpu-code") or opt.startswith("-code") or
        opt.startswith("--relocatable-device-code") or opt.startswith("-rdc") or
        opt in disallow_list_exact):
        fail(opt, "is not allowed to be specified directly via copts of rules_cuda related rules")
    return True

def _get_cuda_archs_info(ctx):
    return ctx.attr._default_cuda_archs[CudaArchsInfo]

def _create_common_info(
        cuda_archs_info = None,
        includes = [],
        quote_includes = [],
        system_includes = [],
        headers = [],
        transitive_headers = [],
        defines = [],
        local_defines = [],
        compile_flags = [],
        link_flags = [],
        host_defines = [],
        host_local_defines = [],
        host_compile_flags = [],
        host_link_flags = [],
        ptxas_flags = [],
        transitive_cc_info = None,
        transitive_linking_contexts = []):
    """Constructor of the common object.

    Args:
        cuda_archs_info: `CudaArchsInfo`.
        includes: include paths. Can be used with `#include <...>` and `#include "..."`.
        quote_includes: include paths. Can be used with `#include "..."`.
        system_includes: include paths. Can be used with `#include <...>`.
        headers: headers directly relate with this target.
        transitive_headers: headers transitively gather from `deps`.
        defines: public `#define`s. Pass to compiler driver directly. Will be seen by downstream targets.
        local_defines: private `#define`s. Pass to compiler driver directly. Will not be seen by downstream targets.
        compile_flags: flags pass to compiler driver directly.
        link_flags: flags pass to device linker.
        host_defines: public `#define`s. Pass to host compiler. Will be seen by downstream targets.
        host_local_defines: private `#define`s. Pass to host compiler. Will not be seen by downstream targets.
        host_compile_flags: flags pass to host compiler.
        host_link_flags: flags pass to host linker.
        ptxas_flags: flags pass to `ptxas`.
        transitive_linking_contexts: `CcInfo.linking_context` extracted from `deps`
    """
    return struct(
        cuda_archs_info = cuda_archs_info,
        includes = includes,
        quote_includes = quote_includes,
        system_includes = system_includes,
        headers = depset(headers, transitive = transitive_headers),
        defines = defines,
        local_defines = local_defines,
        compile_flags = compile_flags,
        link_flags = link_flags,
        host_defines = host_defines,
        host_local_defines = host_local_defines,
        host_compile_flags = host_compile_flags,
        host_link_flags = host_link_flags,
        ptxas_flags = ptxas_flags,
        transitive_cc_info = transitive_cc_info,
        transitive_linker_inputs = [ctx.linker_inputs for ctx in transitive_linking_contexts],
        transitive_linking_contexts = transitive_linking_contexts,
    )

def _create_common(ctx):
    """Helper to gather and process various information from `ctx` object to ease the parameter passing for internal macros.

    See `cuda_helper.create_common_info` what information a common object encapsulates.
    """
    attr = ctx.attr

    all_cc_deps = [dep for dep in attr.deps if CcInfo in dep]
    if hasattr(attr, "_builtin_deps"):
        all_cc_deps.extend([dep for dep in attr._builtin_deps if CcInfo in dep])

    merged_cc_info = cc_common.merge_cc_infos(cc_infos = [dep[CcInfo] for dep in all_cc_deps])

    # gather include info
    includes = merged_cc_info.compilation_context.includes.to_list()
    system_includes = []
    quote_includes = []
    quote_includes.extend(_resolve_workspace_root_includes(ctx))
    for inc in attr.includes:
        system_includes.extend(_resolve_includes(ctx, inc))
    system_includes.extend(merged_cc_info.compilation_context.system_includes.to_list())
    quote_includes.extend(merged_cc_info.compilation_context.quote_includes.to_list())

    # gather header info
    public_headers = []
    private_headers = []
    for fs in attr.hdrs:
        public_headers.extend(fs.files.to_list())
    for fs in attr.srcs:
        hdr = [f for f in fs.files.to_list() if _check_src_extension(f, ALLOW_CUDA_HDRS)]
        private_headers.extend(hdr)
    headers = public_headers + private_headers
    transitive_headers = [merged_cc_info.compilation_context.headers]

    # gather linker info
    transitive_linking_contexts = [merged_cc_info.linking_context]

    # gather compile info
    defines = []
    local_defines = [i for i in attr.local_defines]
    compile_flags = attr._default_cuda_copts[BuildSettingInfo].value + [o for o in attr.copts if _check_opts(o)]
    link_flags = []
    if hasattr(attr, "linkopts"):
        link_flags.extend([o for o in attr.linkopts if _check_opts(o)])
    host_defines = []
    host_local_defines = [i for i in attr.host_local_defines]
    host_compile_flags = attr._default_host_copts[BuildSettingInfo].value + [i for i in attr.host_copts]
    host_link_flags = []
    if hasattr(attr, "host_linkopts"):
        host_link_flags.extend([i for i in attr.host_linkopts])
    for dep in attr.deps:
        if CudaInfo in dep:
            defines.extend(dep[CudaInfo].defines.to_list())
    host_defines.extend(merged_cc_info.compilation_context.defines.to_list())
    defines.extend(attr.defines)
    host_defines.extend(attr.host_defines)

    ptxas_flags = [o for o in attr.ptxasopts if _check_opts(o)]

    return _create_common_info(
        cuda_archs_info = _get_cuda_archs_info(ctx),
        includes = includes,
        quote_includes = quote_includes,
        system_includes = system_includes,
        headers = headers,
        transitive_headers = transitive_headers,
        defines = defines,
        local_defines = local_defines,
        compile_flags = compile_flags,
        link_flags = link_flags,
        host_defines = host_defines,
        host_local_defines = host_local_defines,
        host_compile_flags = host_compile_flags,
        host_link_flags = host_link_flags,
        ptxas_flags = ptxas_flags,
        transitive_cc_info = merged_cc_info,
        transitive_linking_contexts = transitive_linking_contexts,
    )

def _create_cuda_info(defines = None, objects = None, rdc_objects = None, pic_objects = None, rdc_pic_objects = None):
    """Constructor for `CudaInfo`. See the providers documentation for detail."""
    ret = CudaInfo(
        defines = defines if defines != None else depset([]),
        objects = objects if objects != None else depset([]),
        rdc_objects = rdc_objects if rdc_objects != None else depset([]),
        pic_objects = pic_objects if pic_objects != None else depset([]),
        rdc_pic_objects = rdc_pic_objects if rdc_pic_objects != None else depset([]),
    )
    return ret

def _get_artifact_category_from_action(action_name, use_pic = None, use_rdc = None):
    """Query the canonical artifact category name."""
    if action_name == ACTION_NAMES.cuda_compile:
        if use_pic:
            if use_rdc:
                return ARTIFACT_CATEGORIES.rdc_pic_object_file
            else:
                return ARTIFACT_CATEGORIES.pic_object_file
        elif use_rdc:
            return ARTIFACT_CATEGORIES.rdc_object_file
        else:
            return ARTIFACT_CATEGORIES.object_file
    elif action_name == ACTION_NAMES.device_link:
        if not use_rdc:
            fail("non relocatable device code cannot be device linked")
        if use_pic:
            return ARTIFACT_CATEGORIES.rdc_pic_object_file
        else:
            return ARTIFACT_CATEGORIES.rdc_object_file
    else:
        fail("NotImplemented")

def _get_artifact_name(cuda_toolchain, category_name, output_basename):
    """Create the artifact name that follow the toolchain configuration.

    Args:
        cuda_toolchain: CUDA toolchain returned by `find_cuda_toolchain`.
        category_name: The canonical artifact category name return by `cuda_helper.get_artifact_category_from_action`
        output_basename: The basename.
    """
    return config_helper.get_artifact_name(cuda_toolchain.artifact_name_patterns, category_name, output_basename)

def _check_must_enforce_rdc(*, arch_specs = None, cuda_archs_info = None):
    """Force enable rdc if dlto is enabled."""
    if arch_specs == None:
        arch_specs = cuda_archs_info.arch_specs
    for arch_spec in arch_specs:
        for stage2_arch in arch_spec.stage2_archs:
            if stage2_arch.lto:
                return True
    return False

# buildifier: disable=unused-variable
def _create_compile_variables(
        cuda_toolchain,
        feature_configuration,
        cuda_archs_info,
        source_file = None,
        output_file = None,
        host_compiler = None,
        compile_flags = [],
        host_compile_flags = [],
        include_paths = [],
        quote_include_paths = [],
        system_include_paths = [],
        defines = [],
        host_defines = [],
        ptxas_flags = [],
        use_pic = False,
        use_rdc = False):
    """Returns variables used for `compile` actions.

    Args:
        cuda_toolchain: cuda_toolchain for which we are creating build variables.
        feature_configuration: Feature configuration to be queried.
        cuda_archs_info: `CudaArchsInfo`
        source_file: source file for the compilation.
        output_file: output file of the compilation.
        host_compiler: host compiler path.
        compile_flags: flags pass to compiler driver directly.
        host_compile_flags: flags pass to host compiler.
        include_paths: include paths. Can be used with `#include <...>` and `#include "..."`.
        quote_include_paths: include paths. Can be used with `#include "..."`.
        system_include_paths: include paths. Can be used with `#include <...>`.
        defines: `#define`s. Pass to compiler driver directly.
        host_defines: `#define`s. Pass to host compiler.
        ptxas_flags: flags pass to `ptxas`.
        use_pic: whether to compile for position independent code.
        use_rdc: whether to compile for relocatable device code.
    """
    arch_specs = cuda_archs_info.arch_specs
    if not use_rdc:
        use_rdc = _check_must_enforce_rdc(arch_specs = arch_specs)

    return struct(
        arch_specs = arch_specs,
        use_arch_native = len(arch_specs) == 0,
        source_file = source_file,
        output_file = output_file,
        host_compiler = host_compiler,
        compile_flags = compile_flags,
        host_compile_flags = host_compile_flags,
        include_paths = include_paths,
        quote_include_paths = quote_include_paths,
        system_include_paths = system_include_paths,
        defines = defines,
        host_defines = host_defines,
        ptxas_flags = ptxas_flags,
        use_pic = use_pic,
        use_rdc = use_rdc,
    )

# buildifier: disable=unused-variable
def _create_device_link_variables(
        cuda_toolchain,
        feature_configuration,
        cuda_archs_info,
        output_file = None,
        host_compiler = None,
        host_compile_flags = [],
        user_link_flags = [],
        use_pic = False):
    """Returns variables used for `device_link` actions.

    Args:
        cuda_toolchain: cuda_toolchain for which we are creating build variables.
        feature_configuration: Feature configuration to be queried.
        cuda_archs_info: `CudaArchsInfo`
        output_file: output file of the device linking.
        host_compiler: host compiler path.
        host_compile_flags: flags pass to host compiler.
        user_link_flags: flags for device linking.
        use_pic: whether to compile for position independent code.
    """
    arch_specs = cuda_archs_info.arch_specs

    # For using -gencode with lto, see
    # https://forums.developer.nvidia.com/t/using-dlink-time-opt-together-with-gencode-in-cmake/165224/4
    # compile: -gencode=arch=compute_52,code=[compute_52,lto_52,lto_61]
    # dlink  : -gencode=arch=compute_52,code=[sm_52,sm_61] -dlto
    use_dlto = False
    for arch_spec in arch_specs:
        for stage2_arch in arch_spec.stage2_archs:
            if stage2_arch.lto:
                use_dlto = True
                break
    return struct(
        arch_specs = arch_specs,
        use_arch_native = len(arch_specs) == 0,
        output_file = output_file,
        host_compiler = host_compiler,
        host_compile_flags = host_compile_flags,
        user_link_flags = user_link_flags,
        use_dlto = use_dlto,
        use_pic = use_pic,
    )

def _get_all_unsupported_features(ctx, cuda_toolchain, unsupported_features):
    all_unsupported = list(ctx.disabled_features)
    all_unsupported.extend([f[1:] for f in ctx.attr.features if f.startswith("-")])
    if unsupported_features != None:
        all_unsupported.extend(unsupported_features)
    return unique(all_unsupported)

def _get_all_requested_features(ctx, cuda_toolchain, requested_features):
    all_features = []
    compilation_mode = ctx.var.get("COMPILATION_MODE", None)
    if compilation_mode == None:
        print("unknown COMPILATION_MODE, use opt")  # buildifier: disable=print
        compilation_mode = "opt"
    all_features.append(compilation_mode)

    all_features.extend(ctx.features)
    all_features.extend([f for f in ctx.attr.features if not f.startswith("-")])
    all_features.extend(requested_features)
    all_features = unique(all_features)

    # https://github.com/bazelbuild/bazel/blob/41feb616ae/src/main/java/com/google/devtools/build/lib/rules/cpp/CcCommon.java#L953-L967
    if "static_link_msvcrt" in all_features:
        all_features.append("static_link_msvcrt_debug" if compilation_mode == "dbg" else "static_link_msvcrt_no_debug")
    else:
        all_features.append("dynamic_link_msvcrt_debug" if compilation_mode == "dbg" else "dynamic_link_msvcrt_no_debug")

    return all_features

# buildifier: disable=function-docstring-args
def _configure_features(ctx, cuda_toolchain, requested_features = None, unsupported_features = None, _debug = False):
    """Creates a feature_configuration instance.

    Args:
        ctx: The rule context.
        cuda_toolchain: cuda_toolchain for which we configure features.
        requested_features: List of features to be enabled.
        unsupported_features: List of features that are unsupported by the current rule.
    """
    all_requested_features = _get_all_requested_features(ctx, cuda_toolchain, requested_features)
    all_unsupported_features = _get_all_unsupported_features(ctx, cuda_toolchain, unsupported_features)
    return config_helper.configure_features(
        selectables_info = cuda_toolchain.selectables_info,
        requested_features = all_requested_features,
        unsupported_features = all_unsupported_features,
        _debug = _debug,
    )

cuda_helper = struct(
    get_arch_specs = _get_arch_specs,
    check_srcs_extensions = _check_srcs_extensions,
    check_must_enforce_rdc = _check_must_enforce_rdc,
    get_basename_without_ext = _get_basename_without_ext,
    create_common_info = _create_common_info,
    create_common = _create_common,
    create_cuda_info = _create_cuda_info,
    get_artifact_category_from_action = _get_artifact_category_from_action,
    get_artifact_name = _get_artifact_name,
    create_compile_variables = _create_compile_variables,
    create_device_link_variables = _create_device_link_variables,
    configure_features = _configure_features,  # wrapped for collecting info from ctx and cuda_toolchain
    get_command_line = config_helper.get_command_line,
    get_tool_for_action = config_helper.get_tool_for_action,
    action_is_enabled = config_helper.is_enabled,
    is_enabled = config_helper.is_enabled,
    get_environment_variables = config_helper.get_environment_variables,
)
