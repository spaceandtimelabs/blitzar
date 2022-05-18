load("@rules_cc//cc:action_names.bzl", 
     "CPP_LINK_EXECUTABLE_ACTION_NAME",
     "CPP_COMPILE_ACTION_NAME",
     )
load("@rules_cc//cc:toolchain_utils.bzl", "find_cpp_toolchain")
load("//bazel:cc_archive_action.bzl", "sxt_cc_archive_action")
# load("//bazel:cuda_dlink_action.bzl", "cuda_dlink_action")

def get_cc_info_deps(deps):
  return cc_common.merge_cc_infos(
      cc_infos = [dep[CcInfo] for dep in deps])

def dlink_action(ctx, linking_context):
  outfile = ctx.actions.declare_file(ctx.label.name + ".o")
  cc_toolchain = find_cpp_toolchain(ctx)
  feature_configuration = cc_common.configure_features(
      ctx = ctx,
      cc_toolchain = cc_toolchain,
      requested_features = ctx.features,
      unsupported_features = ctx.disabled_features,
  )
  ld = cc_common.get_tool_for_action(
      feature_configuration = feature_configuration,
      action_name = CPP_LINK_EXECUTABLE_ACTION_NAME,
  )
  args = [
      '-x', 'cuda',
      '-dlink',
      '-fPIC',
      '-o', outfile.path
  ]
  inputs = []
  for linker_inputs in linking_context.linker_inputs.to_list():
    for lib in linker_inputs.libraries:
      if lib.static_library:
        args.append(lib.static_library.path)
        inputs.append(lib.static_library)
      if lib.objects:
        args += [ f.path for f in lib.objects ]
        inputs += lib.objects
      if lib.pic_objects:
        args += [ f.path for f in lib.pic_objects ]
        inputs += lib.pic_objects
  ctx.actions.run(
      executable = ld,
      arguments = args,
      use_default_shell_env = True,
      inputs = depset(
          inputs,
          transitive = [cc_toolchain.all_files],
      ),
      outputs = [outfile],
  )
  return outfile

def _cuda_dlink_impl(ctx):
  arcive_out_file = ctx.actions.declare_file(ctx.label.name + ".a")
  deps = ctx.attr.deps
  cc_info_deps = get_cc_info_deps(deps)
  linking_context = cc_info_deps.linking_context
  obj = dlink_action(ctx, linking_context)
  linking_context = sxt_cc_archive_action(ctx, [obj], arcive_out_file)
  outputs = [
      arcive_out_file,
  ]
  cc_info = CcInfo(
      compilation_context = cc_info_deps.compilation_context,
      linking_context = linking_context,
  )
  cc_info = cc_common.merge_cc_infos(cc_infos=[cc_info, cc_info_deps])
  return [
      DefaultInfo(files = depset(outputs)),
      cc_info,
  ]

cuda_dlink = rule(
    implementation = _cuda_dlink_impl,
    attrs = {
        "_cc_toolchain": attr.label(default = Label("@bazel_tools//tools/cpp:current_cc_toolchain")),
        "_driver": attr.label(
            default = Label("@local_config_cuda//crosstool:crosstool_wrapper_driver_is_not_gcc"),
            # executable = True,
            # cfg = "exec",
        ),
        # "_cuda_toolchain": attr.label(default = Label("@local_config_cuda//crosstool:toolchain")),
        "deps" : attr.label_list(),
    },
    toolchains = ["@bazel_tools//tools/cpp:toolchain_type"],
    fragments = ["cpp"],
)
