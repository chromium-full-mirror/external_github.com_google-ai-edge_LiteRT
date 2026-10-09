#include "litert/vendors/google_tensor/hooks/tpu_tile_hook.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_any.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_metrics.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/google_tensor/dispatch/litert_dispatch_invocation_context.h"
#include "litert/vendors/google_tensor/dispatch/litert_dispatch_metrics.h"
#include "litert/vendors/google_tensor/hooks/hooks_utils.h"

namespace litert::google_tensor {

struct TpuTileTimeContext {
  bool dump_tpu_tile_time_metrics = false;
  std::optional<std::string> tpu_metrics_dump_path = std::nullopt;
  bool initialized = false;

  uint64_t total_tpu_tile_time_us = 0;
  int inference_count = 0;
  std::vector<uint64_t> per_inference_tpu_tile_times_us;

  bool in_compiled_model_run = false;
  uint64_t current_inference_tile_time_us = 0;
  int current_inference_dispatch_count = 0;
  bool current_inference_failed = false;

  TpuTileMetricsCollector collector;
};

namespace {

constexpr absl::string_view kDumpTpuMetricsTrue =
    "dump_tpu_tile_time_metrics:true";
constexpr absl::string_view kTpuMetricsDumpPathKey =
    "tpu_tile_time_metrics_dump_path:\"";
constexpr absl::string_view kHardwareExecutionTimeMetricName =
    "hardware_execution_time_us";

TpuTileMetricsCollector CreateDefaultMetricsCollector() {
  return TpuTileMetricsCollector{
      .start_metrics =
          [](LiteRtDispatchInvocationContext icontext, int detail_level) {
            return icontext->StartMetricsCollection(detail_level);
          },
      .stop_metrics =
          [](LiteRtDispatchInvocationContext icontext,
             LiteRtDispatchMetrics& metrics) {
            return icontext->StopMetricsCollection(metrics);
          },
  };
}

void ResetCurrentInferenceState(TpuTileTimeContext* context) {
  context->current_inference_tile_time_us = 0;
  context->current_inference_dispatch_count = 0;
  context->current_inference_failed = false;
}

bool ExtractHardwareExecutionTimeUs(const LiteRtDispatchMetricsT& metrics,
                                    uint64_t* out_time_us) {
  const int num_metrics = metrics.GetNumMetrics();
  for (int i = 0; i < num_metrics; ++i) {
    LiteRtMetric metric;
    metric.name = nullptr;
    if (metrics.GetMetric(i, metric) != kLiteRtStatusOk ||
        metric.name == nullptr) {
      continue;
    }
    if (absl::string_view(metric.name) == kHardwareExecutionTimeMetricName) {
      if (metric.value.type == kLiteRtAnyTypeInt) {
        if (metric.value.int_value < 0) {
          return false;
        }
        *out_time_us = static_cast<uint64_t>(metric.value.int_value);
        return true;
      }
      if (metric.value.type == kLiteRtAnyTypeReal) {
        if (metric.value.real_value < 0.0) {
          return false;
        }
        *out_time_us = static_cast<uint64_t>(metric.value.real_value);
        return true;
      }
      return false;
    }
  }
  return false;
}

void CommitCurrentInferenceIfValid(TpuTileTimeContext* context) {
  if (context->current_inference_dispatch_count > 0 &&
      !context->current_inference_failed) {
    context->inference_count++;
    context->total_tpu_tile_time_us += context->current_inference_tile_time_us;
    context->per_inference_tpu_tile_times_us.push_back(
        context->current_inference_tile_time_us);
  }
  ResetCurrentInferenceState(context);
}

}  // namespace

TpuTileTimeContext* CreateTpuTileTimeContext() {
  auto* context = new TpuTileTimeContext();
  context->collector = CreateDefaultMetricsCollector();
  return context;
}

void DestroyTpuTileTimeContext(TpuTileTimeContext* context) { delete context; }

void ParseTpuTileTimeConfig(absl::string_view input,
                            TpuTileTimeContext* context) {
  if (!context) return;
  context->initialized = true;
  if (absl::StrContains(input, kDumpTpuMetricsTrue)) {
    context->dump_tpu_tile_time_metrics = true;
  }

  auto tpu_path_pos = input.find(kTpuMetricsDumpPathKey);
  if (tpu_path_pos != absl::string_view::npos) {
    auto start = tpu_path_pos + kTpuMetricsDumpPathKey.length();
    auto end = input.find('"', start);
    if (end != absl::string_view::npos) {
      context->tpu_metrics_dump_path =
          std::string(input.substr(start, end - start));
    }
  }
}

void SetTpuTileMetricsCollector(TpuTileTimeContext* context,
                                TpuTileMetricsCollector collector) {
  if (!context) return;
  context->collector = std::move(collector);
}

void HandleTpuTileTimeRuntimeStart(TpuTileTimeContext* context,
                                   LiteRtDispatchInvocationContext icontext) {
  if (!context) return;

  if (!context->initialized) {
    context->initialized = true;
    std::string input = GetVendorHookArgsConfig();
    if (!input.empty()) {
      ParseTpuTileTimeConfig(input, context);
    }
  }
  if (!context->dump_tpu_tile_time_metrics) return;

  if (icontext == nullptr) {
    // Model-level start (`CompiledModel::Run`).
    context->in_compiled_model_run = true;
    ResetCurrentInferenceState(context);
    return;
  }

  // Dispatch-level start (`dispatch_api::Invoke`).
  if (!context->in_compiled_model_run) {
    ResetCurrentInferenceState(context);
  }

  if (context->collector.start_metrics) {
    LiteRtStatus status = context->collector.start_metrics(icontext, 1);
    if (status != kLiteRtStatusOk) {
      LITERT_LOG(
          LITERT_WARNING,
          "Google TensorHook: StartMetricsCollection failed with status %d",
          status);
      context->current_inference_failed = true;
    }
  }
}

void HandleTpuTileTimeRuntimeStop(TpuTileTimeContext* context,
                                  LiteRtDispatchInvocationContext icontext) {
  if (!context || !context->dump_tpu_tile_time_metrics) return;

  if (icontext == nullptr) {
    // Model-level stop (`CompiledModel::Run`).
    CommitCurrentInferenceIfValid(context);
    context->in_compiled_model_run = false;
    return;
  }

  // Dispatch-level stop (`dispatch_api::Invoke`).
  context->current_inference_dispatch_count++;

  LiteRtDispatchMetrics metrics_raw = nullptr;
  LiteRtStatus status = kLiteRtStatusErrorRuntimeFailure;
  if (context->collector.stop_metrics) {
    status = context->collector.stop_metrics(icontext, metrics_raw);
  }
  std::unique_ptr<LiteRtDispatchMetricsT> metrics_deleter(metrics_raw);

  if (status != kLiteRtStatusOk || metrics_raw == nullptr) {
    LITERT_LOG(LITERT_WARNING,
               "Google TensorHook: StopMetricsCollection failed with status %d",
               status);
    context->current_inference_failed = true;
  } else {
    uint64_t current_time_us = 0;
    if (ExtractHardwareExecutionTimeUs(*metrics_raw, &current_time_us)) {
      context->current_inference_tile_time_us += current_time_us;
    } else {
      LITERT_LOG(LITERT_WARNING,
                 "Google TensorHook: Metric '%s' missing or invalid in "
                 "dispatch metrics",
                 kHardwareExecutionTimeMetricName.data());
      context->current_inference_failed = true;
    }
  }

  if (!context->in_compiled_model_run) {
    CommitCurrentInferenceIfValid(context);
  }
}

void HandleTpuTileTimeStopAndProcess(TpuTileTimeContext* context) {
  if (!context) return;

  if (context->dump_tpu_tile_time_metrics && context->inference_count > 0 &&
      !context->per_inference_tpu_tile_times_us.empty()) {
    std::vector<uint64_t> sorted_times =
        context->per_inference_tpu_tile_times_us;
    std::sort(sorted_times.begin(), sorted_times.end());
    const size_t n = sorted_times.size();

    const double avg_tpu_tile_time_us =
        static_cast<double>(context->total_tpu_tile_time_us) /
        static_cast<double>(context->inference_count);
    const uint64_t min_tpu_tile_time_us = sorted_times.front();
    const double median_tpu_tile_time_us =
        (n % 2 == 1) ? static_cast<double>(sorted_times[n / 2])
                     : (static_cast<double>(sorted_times[n / 2 - 1]) +
                        static_cast<double>(sorted_times[n / 2])) /
                           2.0;
    const size_t p95_index = std::min(
        n - 1,
        static_cast<size_t>(std::ceil(0.95 * static_cast<double>(n)) - 1.0));
    const uint64_t p95_tpu_tile_time_us = sorted_times[p95_index];

    std::string output = absl::StrFormat(
        "==== Google Tensor TPU Tile Time Metrics ====\n"
        "Total inferences: %d\n"
        "Total TPU tile time (us): %llu\n"
        "Average TPU tile time (us): %.2f\n"
        "Min TPU tile time (us): %llu\n"
        "Median TPU tile time (us): %.2f\n"
        "P95 TPU tile time (us): %llu\n"
        "==================================",
        context->inference_count, context->total_tpu_tile_time_us,
        avg_tpu_tile_time_us, min_tpu_tile_time_us, median_tpu_tile_time_us,
        p95_tpu_tile_time_us);
    LITERT_LOG(LITERT_INFO, "%s", output.c_str());

    if (context->tpu_metrics_dump_path.has_value() &&
        !context->tpu_metrics_dump_path->empty()) {
      std::ofstream outfile(*context->tpu_metrics_dump_path);
      if (outfile.is_open()) {
        outfile << output << "\n";
        outfile.close();
        LITERT_LOG(LITERT_INFO, "Google TensorHook: TPU metrics dumped to %s",
                   context->tpu_metrics_dump_path->c_str());
      }
    }
  }

  context->total_tpu_tile_time_us = 0;
  context->inference_count = 0;
  context->per_inference_tpu_tile_times_us.clear();
  context->in_compiled_model_run = false;
  ResetCurrentInferenceState(context);
  context->tpu_metrics_dump_path = std::nullopt;
  context->dump_tpu_tile_time_metrics = false;
  context->initialized = false;
}

}  // namespace litert::google_tensor
