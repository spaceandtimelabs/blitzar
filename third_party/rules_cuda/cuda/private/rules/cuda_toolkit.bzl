load("//cuda/private:providers.bzl", "CudaToolkitInfo")

def _impl(ctx):
    version_major, version_minor = ctx.attr.version.split(".")[:2]
    return CudaToolkitInfo(
        path = ctx.attr.path,
        version_major = int(version_major),
        version_minor = int(version_minor),
        nvlink = ctx.file.nvlink,
        link_stub = ctx.file.link_stub,
        bin2c = ctx.file.bin2c,
        fatbinary = ctx.file.fatbinary,
    )

cuda_toolkit = rule(
    doc = """This rule provides CudaToolkitInfo.""",
    implementation = _impl,
    attrs = {
        "path": attr.string(mandatory = True, doc = "Root path to the CUDA Toolkit."),
        "version": attr.string(mandatory = True, doc = "Version of the CUDA Toolkit."),
        "nvlink": attr.label(allow_single_file = True, doc = "The nvlink executable."),
        "link_stub": attr.label(allow_single_file = True, doc = "The link.stub text file."),
        "bin2c": attr.label(allow_single_file = True, doc = "The bin2c executable."),
        "fatbinary": attr.label(allow_single_file = True, doc = "The fatbinary executable."),
    },
    provides = [CudaToolkitInfo],
)
