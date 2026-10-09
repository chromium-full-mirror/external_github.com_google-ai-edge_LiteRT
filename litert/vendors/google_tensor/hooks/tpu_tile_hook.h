#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_GOOGLE_TENSOR_HOOKS_TPU_TILE_HOOK_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_GOOGLE_TENSOR_HOOKS_TPU_TILE_HOOK_H_

#include <functional>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/vendors/c/litert_dispatch.h"

namespace litert::google_tensor {

struct TpuTileTimeContext;

struct TpuTileMetricsCollector {
  std::function<LiteRtStatus(LiteRtDispatchInvocationContext, int)>
      start_metrics;
  std::function<LiteRtStatus(LiteRtDispatchInvocationContext,
                             LiteRtDispatchMetrics&)>
      stop_metrics;
};

TpuTileTimeContext* CreateTpuTileTimeContext();

void DestroyTpuTileTimeContext(TpuTileTimeContext* context);

void ParseTpuTileTimeConfig(absl::string_view input,
                            TpuTileTimeContext* context);

void SetTpuTileMetricsCollector(TpuTileTimeContext* context,
                                TpuTileMetricsCollector collector);

void HandleTpuTileTimeRuntimeStart(TpuTileTimeContext* context,
                                   LiteRtDispatchInvocationContext icontext);

void HandleTpuTileTimeRuntimeStop(TpuTileTimeContext* context,
                                  LiteRtDispatchInvocationContext icontext);

void HandleTpuTileTimeStopAndProcess(TpuTileTimeContext* context);

}  // namespace litert::google_tensor

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_GOOGLE_TENSOR_HOOKS_TPU_TILE_HOOK_H_
