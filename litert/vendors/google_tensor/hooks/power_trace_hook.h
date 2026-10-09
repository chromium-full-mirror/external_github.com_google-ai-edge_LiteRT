#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_GOOGLE_TENSOR_HOOKS_POWER_TRACE_HOOK_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_GOOGLE_TENSOR_HOOKS_POWER_TRACE_HOOK_H_

#include <cstdint>
#include <functional>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/vendors/c/litert_dispatch.h"

namespace litert::google_tensor {

struct PowerTraceContext;

using PowerEnergyReader = std::function<absl::StatusOr<uint64_t>()>;

PowerTraceContext* CreatePowerTraceContext();

void DestroyPowerTraceContext(PowerTraceContext* context);

void ParsePowerTraceConfig(absl::string_view input, PowerTraceContext* context);

void SetPowerEnergyReader(PowerTraceContext* context,
                          PowerEnergyReader energy_reader);

// Invoked when kLiteRtHookTypeRuntimeStart is received.
void HandlePowerRuntimeStart(PowerTraceContext* context,
                             LiteRtDispatchInvocationContext icontext);

// Invoked when kLiteRtHookTypeRuntimeStop is received.
void HandlePowerRuntimeStop(PowerTraceContext* context,
                            LiteRtDispatchInvocationContext icontext);

// Invoked when kLiteRtHookTypeStopAndProcess is received.
void HandlePowerStopAndProcess(PowerTraceContext* context);

}  // namespace litert::google_tensor

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_GOOGLE_TENSOR_HOOKS_POWER_TRACE_HOOK_H_
