"""Entry point for extensions used by bzlmod."""

load("//cuda/private:repositories.bzl", "local_cuda")

cuda_toolkit = tag_class(attrs = {
    "name": attr.string(doc = "Name for the toolchain repository", default = "local_cuda"),
    "toolkit_path": attr.string(doc = "Path to the CUDA SDK, if empty the environment variable CUDA_PATH will be used to deduce this path."),
})

def _init(module_ctx):
    registrations = {}
    for mod in module_ctx.modules:
        for toolchain in mod.tags.local_toolchain:
            if not mod.is_root:
                fail("Only the root module may override the path for the local cuda toolchain")
            if toolchain.name in registrations.keys():
                if toolchain.toolkit_path == registrations[toolchain.name]:
                    # No problem to register a matching toolchain twice
                    continue
                fail("Multiple conflicting toolchains declared for name {} ({} and {}".format(toolchain.name, toolchain.toolkit_path, registrations[toolchain.name]))
            else:
                registrations[toolchain.name] = toolchain.toolkit_path
    for name, toolkit_path in registrations.items():
        local_cuda(name = name, toolkit_path = toolkit_path)

toolchain = module_extension(
    implementation = _init,
    tag_classes = {"local_toolchain": cuda_toolkit},
)
