#include <stdlib.h>  // IWYU pragma: keep

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_profiler_types.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/google_tensor/dispatch/google_tensor_hook.h"
#include "litert/vendors/google_tensor/dispatch/litert_dispatch_metrics.h"
#include "litert/vendors/google_tensor/dispatch/sb_api.h"
#include "litert/vendors/google_tensor/hooks/power_trace_hook.h"
#include "litert/vendors/google_tensor/hooks/tpu_tile_hook.h"

namespace litert::google_tensor {
namespace {

using ::testing::HasSubstr;

std::string ReadFileContents(const std::string& path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    return "";
  }
  std::stringstream buffer;
  buffer << file.rdbuf();
  return buffer.str();
}

bool FileExists(const std::string& path) {
  std::ifstream file(path);
  return file.good();
}

LiteRtDispatchMetrics CreateFakeDispatchMetrics(const char* metric_name,
                                                int64_t value) {
  const char* keys[1] = {metric_name};
  int64_t values[1] = {value};
  ThrInvocationMetrics thr_metrics{
      .num_metrics = 1,
      .metric_keys = keys,
      .metric_values = values,
  };
  return new LiteRtDispatchMetricsT(thr_metrics);
}

LiteRtDispatchInvocationContext FakeInvocationContext() {
  return reinterpret_cast<LiteRtDispatchInvocationContext>(0x1);
}

TEST(TpuTileHookTest, SinglePartitionDoesNotDoubleCountAndReportsDistribution) {
  std::string dump_path = absl::StrCat(::testing::TempDir(), "/tpu_single.txt");
  std::remove(dump_path.c_str());

  std::unique_ptr<TpuTileTimeContext, void (*)(TpuTileTimeContext*)> ctx(
      CreateTpuTileTimeContext(), DestroyTpuTileTimeContext);
  ParseTpuTileTimeConfig(absl::StrCat("dump_tpu_tile_time_metrics:true,"
                                      "tpu_tile_time_metrics_dump_path:\"",
                                      dump_path, "\""),
                         ctx.get());

  std::vector<int64_t> simulated_times = {100, 150, 125, 126};
  size_t call_idx = 0;
  SetTpuTileMetricsCollector(
      ctx.get(), TpuTileMetricsCollector{
                     .start_metrics = [](LiteRtDispatchInvocationContext,
                                         int) { return kLiteRtStatusOk; },
                     .stop_metrics =
                         [&](LiteRtDispatchInvocationContext,
                             LiteRtDispatchMetrics& metrics) {
                           metrics = CreateFakeDispatchMetrics(
                               "hardware_execution_time_us",
                               simulated_times[call_idx++]);
                           return kLiteRtStatusOk;
                         },
                 });

  for (size_t i = 0; i < simulated_times.size(); ++i) {
    // Simulate CompiledModel::Run wrapping a single dispatch_api::Invoke.
    HandleTpuTileTimeRuntimeStart(ctx.get(), /*icontext=*/nullptr);
    HandleTpuTileTimeRuntimeStart(ctx.get(), FakeInvocationContext());
    HandleTpuTileTimeRuntimeStop(ctx.get(), FakeInvocationContext());
    HandleTpuTileTimeRuntimeStop(ctx.get(), /*icontext=*/nullptr);
  }

  HandleTpuTileTimeStopAndProcess(ctx.get());

  std::string output = ReadFileContents(dump_path);
  EXPECT_THAT(output, HasSubstr("Total inferences: 4\n"));
  EXPECT_THAT(output, HasSubstr("Total TPU tile time (us): 501\n"));
  EXPECT_THAT(output, HasSubstr("Average TPU tile time (us): 125.25\n"));
  EXPECT_THAT(output, HasSubstr("Min TPU tile time (us): 100\n"));
  EXPECT_THAT(output, HasSubstr("Median TPU tile time (us): 125.50\n"));
  EXPECT_THAT(output, HasSubstr("P95 TPU tile time (us): 150\n"));
}

TEST(TpuTileHookTest, MultiPartitionSumsTileTimesPerInference) {
  std::string dump_path = absl::StrCat(::testing::TempDir(), "/tpu_multi.txt");
  std::remove(dump_path.c_str());

  std::unique_ptr<TpuTileTimeContext, void (*)(TpuTileTimeContext*)> ctx(
      CreateTpuTileTimeContext(), DestroyTpuTileTimeContext);
  ParseTpuTileTimeConfig(absl::StrCat("dump_tpu_tile_time_metrics:true,"
                                      "tpu_tile_time_metrics_dump_path:\"",
                                      dump_path, "\""),
                         ctx.get());

  // Two inferences, each with two NPU partitions: (60 + 40 = 100) and (70 + 50
  // = 120).
  std::vector<int64_t> dispatch_times = {60, 40, 70, 50};
  size_t call_idx = 0;
  SetTpuTileMetricsCollector(
      ctx.get(), TpuTileMetricsCollector{
                     .start_metrics = [](LiteRtDispatchInvocationContext,
                                         int) { return kLiteRtStatusOk; },
                     .stop_metrics =
                         [&](LiteRtDispatchInvocationContext,
                             LiteRtDispatchMetrics& metrics) {
                           metrics = CreateFakeDispatchMetrics(
                               "hardware_execution_time_us",
                               dispatch_times[call_idx++]);
                           return kLiteRtStatusOk;
                         },
                 });

  for (int inference = 0; inference < 2; ++inference) {
    HandleTpuTileTimeRuntimeStart(ctx.get(), /*icontext=*/nullptr);
    for (int partition = 0; partition < 2; ++partition) {
      HandleTpuTileTimeRuntimeStart(ctx.get(), FakeInvocationContext());
      HandleTpuTileTimeRuntimeStop(ctx.get(), FakeInvocationContext());
    }
    HandleTpuTileTimeRuntimeStop(ctx.get(), /*icontext=*/nullptr);
  }

  HandleTpuTileTimeStopAndProcess(ctx.get());

  std::string output = ReadFileContents(dump_path);
  EXPECT_THAT(output, HasSubstr("Total inferences: 2\n"));
  EXPECT_THAT(output, HasSubstr("Total TPU tile time (us): 220\n"));
  EXPECT_THAT(output, HasSubstr("Average TPU tile time (us): 110.00\n"));
}

TEST(TpuTileHookTest, FailedMetricsAreSkippedFromInferenceCount) {
  std::string dump_path =
      absl::StrCat(::testing::TempDir(), "/tpu_failed_skip.txt");
  std::remove(dump_path.c_str());

  std::unique_ptr<TpuTileTimeContext, void (*)(TpuTileTimeContext*)> ctx(
      CreateTpuTileTimeContext(), DestroyTpuTileTimeContext);
  ParseTpuTileTimeConfig(absl::StrCat("dump_tpu_tile_time_metrics:true,"
                                      "tpu_tile_time_metrics_dump_path:\"",
                                      dump_path, "\""),
                         ctx.get());

  // Call 0: failed collection. Call 1: valid run (80 us).
  int call_idx = 0;
  SetTpuTileMetricsCollector(
      ctx.get(), TpuTileMetricsCollector{
                     .start_metrics = [](LiteRtDispatchInvocationContext,
                                         int) { return kLiteRtStatusOk; },
                     .stop_metrics =
                         [&](LiteRtDispatchInvocationContext,
                             LiteRtDispatchMetrics& metrics) {
                           int current = call_idx++;
                           if (current == 0) {
                             metrics = nullptr;
                             return kLiteRtStatusErrorRuntimeFailure;
                           }
                           metrics = CreateFakeDispatchMetrics(
                               "hardware_execution_time_us", 80);
                           return kLiteRtStatusOk;
                         },
                 });

  // Run 1 (fails StopMetricsCollection).
  HandleTpuTileTimeRuntimeStart(ctx.get(), /*icontext=*/nullptr);
  HandleTpuTileTimeRuntimeStart(ctx.get(), FakeInvocationContext());
  HandleTpuTileTimeRuntimeStop(ctx.get(), FakeInvocationContext());
  HandleTpuTileTimeRuntimeStop(ctx.get(), /*icontext=*/nullptr);

  // Run 2 (succeeds with 80 us).
  HandleTpuTileTimeRuntimeStart(ctx.get(), /*icontext=*/nullptr);
  HandleTpuTileTimeRuntimeStart(ctx.get(), FakeInvocationContext());
  HandleTpuTileTimeRuntimeStop(ctx.get(), FakeInvocationContext());
  HandleTpuTileTimeRuntimeStop(ctx.get(), /*icontext=*/nullptr);

  HandleTpuTileTimeStopAndProcess(ctx.get());

  std::string output = ReadFileContents(dump_path);
  EXPECT_THAT(output, HasSubstr("Total inferences: 1\n"));
  EXPECT_THAT(output, HasSubstr("Total TPU tile time (us): 80\n"));
  EXPECT_THAT(output, HasSubstr("Average TPU tile time (us): 80.00\n"));
}

TEST(PowerTraceHookTest,
     ReadsEnergyOnlyAtStartAndEndOfBenchmarkAndIgnoresDispatchEvents) {
  std::string dump_path =
      absl::StrCat(::testing::TempDir(), "/power_bench.txt");
  std::remove(dump_path.c_str());

  std::unique_ptr<PowerTraceContext, void (*)(PowerTraceContext*)> ctx(
      CreatePowerTraceContext(), DestroyPowerTraceContext);
  ParsePowerTraceConfig(absl::StrCat("dump_power_metrics:true,"
                                     "power_dump_path:\"",
                                     dump_path, "\""),
                        ctx.get());

  // Energy readings:
  // 1st read (first inference start): 2000 uWs
  // 2nd read (stop and process): 2750 uWs
  std::vector<uint64_t> energy_readings = {2000, 2750};
  int energy_read_calls = 0;
  SetPowerEnergyReader(ctx.get(), [&]() -> absl::StatusOr<uint64_t> {
    return energy_readings[energy_read_calls++];
  });

  // 3 inferences, each with 2 NPU partitions.
  for (int i = 0; i < 3; ++i) {
    HandlePowerRuntimeStart(ctx.get(), /*icontext=*/nullptr);
    HandlePowerRuntimeStart(ctx.get(), FakeInvocationContext());
    HandlePowerRuntimeStop(ctx.get(), FakeInvocationContext());
    HandlePowerRuntimeStart(ctx.get(), FakeInvocationContext());
    HandlePowerRuntimeStop(ctx.get(), FakeInvocationContext());
    HandlePowerRuntimeStop(ctx.get(), /*icontext=*/nullptr);
  }
  // Only 1 sysfs energy read (at the very first model-level RuntimeStart).
  EXPECT_EQ(energy_read_calls, 1);

  // StopAndProcess performs the single ending energy read.
  HandlePowerStopAndProcess(ctx.get());
  EXPECT_EQ(energy_read_calls, 2);

  std::string output = ReadFileContents(dump_path);
  EXPECT_THAT(
      output,
      HasSubstr("Average TPU Energy consumed per inference(uJ): 250.000\n"));
  EXPECT_THAT(output, HasSubstr("Total Energy (uJ): 750\n"));
  EXPECT_THAT(output, HasSubstr("Inference count: 3\n"));
}

TEST(PowerTraceHookTest, SkipsFileDumpWhenEnergyReadFails) {
  std::string dump_path =
      absl::StrCat(::testing::TempDir(), "/power_failed.txt");
  std::remove(dump_path.c_str());

  std::unique_ptr<PowerTraceContext, void (*)(PowerTraceContext*)> ctx(
      CreatePowerTraceContext(), DestroyPowerTraceContext);
  ParsePowerTraceConfig(absl::StrCat("dump_power_metrics:true,"
                                     "power_dump_path:\"",
                                     dump_path, "\""),
                        ctx.get());

  SetPowerEnergyReader(ctx.get(), []() -> absl::StatusOr<uint64_t> {
    return absl::InternalError("ODPM sysfs unavailable");
  });

  HandlePowerRuntimeStart(ctx.get(), /*icontext=*/nullptr);
  HandlePowerRuntimeStop(ctx.get(), /*icontext=*/nullptr);
  HandlePowerStopAndProcess(ctx.get());

  EXPECT_FALSE(FileExists(dump_path));
}

TEST(GoogleTensorHookTest, GetHooksAndVendorHookDispatchStopAndProcess) {
  setenv("LITERT_VENDOR_HOOK_ARGS", "dump_tpu_tile_time_metrics: true", 1);

  LiteRtHook hook = nullptr;
  void* user_data = nullptr;
  ASSERT_EQ(GetHooks(/*device_context=*/nullptr, &hook, &user_data),
            kLiteRtStatusOk);
  ASSERT_NE(hook, nullptr);
  ASSERT_NE(user_data, nullptr);

  hook(kLiteRtHookTypeRuntimeStart, nullptr, 0, user_data);
  hook(kLiteRtHookTypeRuntimeStop, nullptr, 0, user_data);
  hook(kLiteRtHookTypeStopAndProcess, nullptr, 0, user_data);

  unsetenv("LITERT_VENDOR_HOOK_ARGS");
}

}  // namespace
}  // namespace litert::google_tensor
