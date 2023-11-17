config_setting(
    name = "portable_glibc_build",
    values = {"define": "SXT_PORTABLE_GLIBC=1"},
)

# Uses method described in this stackoverflow post
# https://stackoverflow.com/a/5977518/4447365
# and https://github.com/lightstep/lightstep-tracer-cpp/blob/master/bazel/BUILD#L20-L36
# to force linking to older glibc symbols.
cc_library(
    name = "glibc_version_lib",
    srcs = select({
        "//bazel:portable_glibc_build": [
            "glibc_version.cpp",
        ],
        "//conditions:default": [],
    }),
    linkopts = [
        "-Wl,--wrap=shm_unlink",
        "-Wl,--wrap=pthread_join",
        "-Wl,--wrap=dlerror",
        "-Wl,--wrap=__pthread_key_create",
        "-Wl,--wrap=pthread_mutexattr_destroy",
        "-Wl,--wrap=pthread_rwlock_tryrdlock",
        "-Wl,--wrap=pthread_rwlock_destroy",
        "-Wl,--wrap=pthread_setspecific",
        "-Wl,--wrap=pthread_rwlock_timedwrlock",
        "-Wl,--wrap=sem_destroy",
        "-Wl,--wrap=sem_wait",
        "-Wl,--wrap=pthread_once",
        "-Wl,--wrap=pthread_create",
        "-Wl,--wrap=pthread_rwlock_rdlock",
        "-Wl,--wrap=pthread_rwlock_timedrdlock",
        "-Wl,--wrap=pthread_kill",
        "-Wl,--wrap=pthread_mutexattr_init",
        "-Wl,--wrap=pthread_key_create",
        "-Wl,--wrap=pthread_rwlock_init",
        "-Wl,--wrap=pthread_rwlockattr_init",
        "-Wl,--wrap=dlopen",
        "-Wl,--wrap=sem_init",
        "-Wl,--wrap=pthread_rwlock_unlock",
        "-Wl,--wrap=pthread_detach",
        "-Wl,--wrap=pthread_mutexattr_setpshared",
        "-Wl,--wrap=shm_open",
        "-Wl,--wrap=sem_timedwait",
        "-Wl,--wrap=dladdr",
        "-Wl,--wrap=pthread_rwlock_wrlock",
        "-Wl,--wrap=sem_post",
        "-Wl,--wrap=pthread_rwlock_trywrlock",
        "-Wl,--wrap=pthread_condattr_setpshared",
        "-Wl,--wrap=sem_trywait",
        "-Wl,--wrap=pthread_key_delete",
        "-Wl,--wrap=pthread_getspecific",
        "-Wl,--wrap=pthread_mutexattr_settype",
        "-Wl,--wrap=dlvsym",
        "-Wl,--wrap=dlsym",
        "-Wl,--wrap=pthread_mutex_trylock",
        "-Wl,--wrap=pthread_rwlockattr_setpshared",
        "-Wl,--wrap=pthread_rwlockattr_destroy",
        "-Wl,--wrap=dlclose",
        "-Wl,--wrap=__libc_single_threaded",

        # Using -lrt isn't properly linking the library.
        # Since it's a required library (provides shm_open and shm_unlink),
        # we hard-code to force its usage.
        "/lib/x86_64-linux-gnu/librt.so.1",
    ],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)
