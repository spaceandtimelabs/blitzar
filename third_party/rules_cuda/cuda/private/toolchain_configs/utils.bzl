def nvcc_version_ge(ctx, major, minor):
    if ctx.attr.toolchain_identifier != "nvcc":
        return False
    if ctx.attr.nvcc_version_major < major:
        return False
    if ctx.attr.nvcc_version_minor < minor:
        return False
    return True
