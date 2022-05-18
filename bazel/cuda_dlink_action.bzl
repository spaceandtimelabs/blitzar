load("@rules_cc//cc:action_names.bzl", 
     "CPP_COMPILE_ACTION_NAME",
     # "CPP_LINK_EXECUTABLE_ACTION_NAME",
     )
load("@rules_cc//cc:toolchain_utils.bzl", "find_cpp_toolchain")

def get_linker_and_args(ctx, cc_toolchain, feature_configuration, rpaths, output_file):
    user_link_flags = ctx.fragments.cpp.linkopts
    link_variables = cc_common.create_link_variables(
        feature_configuration = feature_configuration,
        cc_toolchain = cc_toolchain,
        is_linking_dynamic_library = False,
        runtime_library_search_directories = rpaths,
        user_link_flags = user_link_flags,
        output_file = output_file,
    )
    link_args = cc_common.get_memory_inefficient_command_line(
        feature_configuration = feature_configuration,
        action_name = CPP_LINK_EXECUTABLE_ACTION_NAME,
        variables = link_variables,
    )
    link_env = cc_common.get_environment_variables(
        feature_configuration = feature_configuration,
        action_name = CPP_LINK_EXECUTABLE_ACTION_NAME,
        variables = link_variables,
    )
    ld = cc_common.get_tool_for_action(
        feature_configuration = feature_configuration,
        action_name = CPP_LINK_EXECUTABLE_ACTION_NAME,
    )

    return ld, link_args, link_env


def cuda_dlink_action(ctx, linking_context, compilation_context):
  outfile = ctx.actions.declare_file(ctx.label.name + ".o")
  cc_toolchain = find_cpp_toolchain(ctx)
  ld, link_args, link_env = get_linker_and_args(
      ctx, cc_toolchain, feature_configuration, outfile.path)
  args = [
      '-x', 'cuda',
      '-dlink',
      '-o', outfile.path
  ]
  inputs = list(link_args)
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
      env = link_env,
      use_default_shell_env = True,
      inputs = depset(
          inputs,
          transitive = [cc_toolchain.all_files],
      ),
      outputs = [outfile],
  )
  return outfile
