#include "litert/vendors/google_tensor/hooks/power_trace_hook.h"

#include <cstdint>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "platforms/darwinn/devtools/power_stats/power_stats.h"
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/google_tensor/hooks/hooks_utils.h"

namespace litert::google_tensor {

struct PowerTraceContext {
  std::unique_ptr<platforms::darwinn::devtools::PowerStats> power_stats;
  PowerEnergyReader energy_reader;
  std::optional<uint64_t> start_energy = std::nullopt;
  std::optional<std::string> power_dump_path = std::nullopt;
  bool dump_power_metrics = false;

  int inference_count = 0;
  uint64_t total_inference_energy_uws = 0;
  bool initialized = false;
  bool energy_read_failed = false;
};

namespace {

constexpr absl::string_view kDumpPowerMetricsTrue = "dump_power_metrics:true";
constexpr absl::string_view kPowerDumpPathKey = "power_dump_path:\"";

absl::Status EnsurePowerReaderInitialized(PowerTraceContext* context) {
  if (!context->dump_power_metrics) {
    return absl::OkStatus();
  }
  if (context->energy_reader) {
    return absl::OkStatus();
  }
  if (!context->power_stats) {
    auto stats_or = platforms::darwinn::devtools::PowerStats::Create();
    if (!stats_or) {
      context->energy_read_failed = true;
      return absl::InternalError("Failed to create PowerStats");
    }
    context->power_stats = std::move(stats_or);
  }
  platforms::darwinn::devtools::PowerStats* raw_stats =
      context->power_stats.get();
  context->energy_reader = [raw_stats]() -> absl::StatusOr<uint64_t> {
    return raw_stats->GetEnergyConsumedUWs(
        platforms::darwinn::devtools::power_stats::SUBSYSTEM_TPU);
  };
  return absl::OkStatus();
}

absl::Status CaptureBaselineEnergyIfNeeded(PowerTraceContext* context) {
  if (!context->dump_power_metrics || context->energy_read_failed ||
      context->start_energy.has_value()) {
    return absl::OkStatus();
  }
  if (auto status = EnsurePowerReaderInitialized(context); !status.ok()) {
    return status;
  }
  auto energy_or = context->energy_reader();
  if (!energy_or.ok()) {
    context->energy_read_failed = true;
    return energy_or.status();
  }
  context->start_energy = *energy_or;
  return absl::OkStatus();
}

}  // namespace

PowerTraceContext* CreatePowerTraceContext() { return new PowerTraceContext(); }

void DestroyPowerTraceContext(PowerTraceContext* context) { delete context; }

void ParsePowerTraceConfig(absl::string_view input,
                           PowerTraceContext* context) {
  if (!context) return;
  context->initialized = true;
  if (absl::StrContains(input, kDumpPowerMetricsTrue)) {
    context->dump_power_metrics = true;
  }

  auto path_pos = input.find(kPowerDumpPathKey);
  if (path_pos != absl::string_view::npos) {
    auto start = path_pos + kPowerDumpPathKey.length();
    auto end = input.find('"', start);
    if (end != absl::string_view::npos) {
      context->power_dump_path = std::string(input.substr(start, end - start));
    }
  }
}

void SetPowerEnergyReader(PowerTraceContext* context,
                          PowerEnergyReader energy_reader) {
  if (!context) return;
  context->energy_reader = std::move(energy_reader);
}

void HandlePowerRuntimeStart(PowerTraceContext* context,
                             LiteRtDispatchInvocationContext icontext) {
  if (!context || icontext != nullptr) return;

  if (!context->initialized) {
    context->initialized = true;
    std::string input = GetVendorHookArgsConfig();
    if (!input.empty()) {
      ParsePowerTraceConfig(input, context);
    }
  }
  if (!context->dump_power_metrics) return;

  if (auto status = CaptureBaselineEnergyIfNeeded(context); !status.ok()) {
    LITERT_LOG(LITERT_ERROR,
               "Google TensorHook failed HandlePowerRuntimeStart: %s",
               std::string(status.message()).c_str());
  }
}

void HandlePowerRuntimeStop(PowerTraceContext* context,
                            LiteRtDispatchInvocationContext icontext) {
  if (!context || icontext != nullptr) return;
  if (!context->dump_power_metrics || context->energy_read_failed) return;

  // Only increment the inference count during the timed loop; the ending ODPM
  // energy counter is read once at StopAndProcess to avoid per-inference
  // sysfs file I/O overhead.
  if (context->start_energy.has_value()) {
    context->inference_count++;
  }
}

void HandlePowerStopAndProcess(PowerTraceContext* context) {
  if (!context) return;
  if (context->dump_power_metrics) {
    if (!context->energy_read_failed && context->inference_count > 0 &&
        context->start_energy.has_value() && context->energy_reader) {
      auto end_energy_or = context->energy_reader();
      if (!end_energy_or.ok()) {
        context->energy_read_failed = true;
        LITERT_LOG(LITERT_ERROR,
                   "Google TensorHook failed final ODPM energy read: %s",
                   std::string(end_energy_or.status().message()).c_str());
      } else if (*end_energy_or < *context->start_energy) {
        context->energy_read_failed = true;
        LITERT_LOG(LITERT_ERROR,
                   "Google TensorHook: Final energy (%lu) is less than start "
                   "energy (%lu)",
                   *end_energy_or, *context->start_energy);
      } else {
        context->total_inference_energy_uws =
            *end_energy_or - *context->start_energy;
      }
    }

    if (context->energy_read_failed || context->inference_count <= 0) {
      LITERT_LOG(LITERT_WARNING,
                 "Google TensorHook: Skipping power metrics dump due to failed "
                 "ODPM energy reads or zero recorded inferences (count=%d).",
                 context->inference_count);
    } else {
      const uint64_t diff_energy = context->total_inference_energy_uws;
      const int count = context->inference_count;
      const double avg_energy_uj = static_cast<double>(diff_energy) / count;

      std::string output = absl::StrFormat(
          "Average TPU Energy consumed per inference(uJ): %.3f\n"
          "Total Energy (uJ): %llu\n"
          "Inference count: %d\n",
          avg_energy_uj, diff_energy, count);

      LITERT_LOG(LITERT_INFO, "Power hook has generated following data:\n%s",
                 output.c_str());

      if (context->power_dump_path.has_value() &&
          !context->power_dump_path->empty()) {
        std::ofstream outfile(*context->power_dump_path);
        if (outfile.is_open()) {
          outfile << output;
          outfile.close();
          LITERT_LOG(LITERT_INFO,
                     "Google TensorHook: Power metrics dumped to %s",
                     context->power_dump_path->c_str());
        }
      }
    }
  }

  // State reset for this session context
  context->power_stats.reset();
  context->energy_reader = nullptr;
  context->start_energy = std::nullopt;
  context->total_inference_energy_uws = 0;
  context->inference_count = 0;
  context->power_dump_path = std::nullopt;
  context->dump_power_metrics = false;
  context->initialized = false;
  context->energy_read_failed = false;
}

}  // namespace litert::google_tensor
