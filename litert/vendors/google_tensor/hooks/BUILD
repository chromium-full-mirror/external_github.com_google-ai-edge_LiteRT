load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    default_applicable_licenses = ["//third_party/odml:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "hooks",
    srcs = [
        "hooks_utils.cc",
        "power_trace_hook.cc",
        "tpu_tile_hook.cc",
    ],
    hdrs = [
        "hooks_utils.h",
        "power_trace_hook.h",
        "tpu_tile_hook.h",
    ],
    deps = [
        "//litert/c:litert_any",
        "//litert/c:litert_common",
        "//litert/c:litert_metrics_header",
        "//litert/c/internal:litert_logging",
        "//litert/vendors/c:litert_dispatch_c_api",
        "//litert/vendors/google_tensor/dispatch:dispatch_api_headers",
        "//platforms/darwinn/devtools/power_stats",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test(
    name = "hooks_test",
    srcs = ["hooks_test.cc"],
    deps = [
        ":hooks",
        "//litert/c:litert_common",
        "//litert/c:litert_environment",  # buildcleaner: keep
        "//litert/c:litert_profiler_types",
        "//litert/vendors/c:litert_dispatch_c_api",
        "//litert/vendors/google_tensor/dispatch:dispatch_api_headers",
        "//litert/vendors/google_tensor/dispatch:dispatch_api_static",  # buildcleaner: keep
        "//litert/vendors/google_tensor/dispatch:google_tensor_hook",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)
