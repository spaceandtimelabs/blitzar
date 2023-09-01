load("@bazel_skylib//rules:common_settings.bzl", "BuildSettingInfo")
load("//cuda/private:cuda_helper.bzl", "cuda_helper")
load("//cuda/private:providers.bzl", "CudaArchsInfo")

def _cuda_archs_flag_impl(ctx):
    specs_str = ctx.build_setting_value
    return CudaArchsInfo(arch_specs = cuda_helper.get_arch_specs(specs_str))

cuda_archs_flag = rule(
    doc = """A build setting for specifying cuda archs to compile for.

To retain the flexiblity of NVCC, the [extended notation](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#extended-notation) is adopted.

When passing cuda_archs from commandline, its spec grammar is as follows:

    ARCH_SPECS   ::= ARCH_SPEC [ ';' ARCH_SPECS ]
    ARCH_SPEC    ::= [ VIRTUAL_ARCH ':' ] GPU_ARCHS
    GPU_ARCHS    ::= GPU_ARCH [ ',' GPU_ARCHS ]
    GPU_ARCH     ::= 'sm_' ARCH_NUMBER
                   | 'lto_' ARCH_NUMBER
                   | VIRTUAL_ARCH
    VIRTUAL_ARCH ::= 'compute_' ARCH_NUMBER
                   | 'lto_' ARCH_NUMBER
    ARCH_NUMBER  ::= (a string in predefined cuda_archs list)

E.g.:

- `compute_80:sm_80,sm_86`:
  Use `compute_80` PTX, generate cubin with `sm_80` and `sm_86`, no PTX embedded
- `compute_80:compute_80,sm_80,sm_86`:
  Use `compute_80` PTX, generate cubin with `sm_80` and `sm_86`, PTX embedded
- `compute_80:compute_80`:
  Embed `compute_80` PTX, fully relay on `ptxas`
- `sm_80,sm_86`:
  Same as `compute_80:sm_80,sm_86`, the arch with minimum integer value will be automatically populated.
- `sm_80;sm_86`:
  Two specs used.
- `compute_80`:
  Same as `compute_80:compute_80`

Best Practices:

- Library supports a full range of archs from xx to yy, you should embed the yy PTX
- Library supports a sparse range of archs from xx to yy, you should embed the xx PTX""",
    implementation = _cuda_archs_flag_impl,
    build_setting = config.string(flag = True),
    provides = [CudaArchsInfo],
)

def _repeatable_string_flag_impl(ctx):
    flags = ctx.build_setting_value
    if (flags == [""]):
        flags = []
    return BuildSettingInfo(value = flags)

repeatable_string_flag = rule(
    implementation = _repeatable_string_flag_impl,
    build_setting = config.string(flag = True, allow_multiple = True),
    provides = [BuildSettingInfo],
)
