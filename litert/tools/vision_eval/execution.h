// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef LITERT_TOOLS_VISION_EVAL_EXECUTION_H_
#define LITERT_TOOLS_VISION_EVAL_EXECUTION_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_options.h"
#include "litert/tools/vision_eval/data.h"

namespace litert::vision {

// Validates accelerator selection, thread count, GPU backend/precision/IR
// settings, and Arm Vulkan ML JIT prerequisites in `backend`.
Expected<void> ValidateBackendOptions(const BackendOptions& backend);

// Derives `BackendOptions` for the reference baseline execution from `config`
// (resetting vendor/GPU/NPU overrides when `baseline_accelerator` is `"cpu"`).
BackendOptions BaselineOptions(const EvaluationConfig& config);

// Validates `backend` and builds LiteRT compilation `Options` for the requested
// hardware accelerators, forcing reference CPU kernels when `reference_cpu` is
// true.
Expected<Options> CreateOptions(const BackendOptions& backend,
                                bool reference_cpu = false);

// Queries host operating system, architecture, and hardware/device metadata for
// run provenance reporting.
DeviceIdentity DeviceInfo();

// Collects `FileIdentity` records for loaded `.so` libraries and shared
// libraries in the runtime, dispatch, and compiler plugin directories of
// `config`.
Expected<std::vector<FileIdentity>> RuntimeLibraries(
    const EvaluationConfig& config);

// Packs finite preprocessed `input` floats into raw tensor bytes matching
// `spec` (`FLOAT32`, or per-tensor quantized `INT8`/`UINT8`).
Expected<std::vector<uint8_t>> EncodeImage(const std::vector<float>& input,
                                           const TensorSpec& spec);

// Owns one compiled model and reuses its input and output tensor buffers across
// samples. The enclosing `Environment` must outlive this object.
class Execution {
 private:
  struct Passkey {};

 public:
  // Constructs an `Execution` taking ownership of `model` for `signature`
  // (restricted to `Execution::Create` via `Passkey`).
  Execution(Passkey, CompiledModel model, int signature)
      : model_(std::move(model)), signature_(signature) {}

  // Compiles the `.tflite` model at `path` for `signature_index` using
  // `environment` and `backend` (forcing reference CPU kernels when
  // `reference_cpu` is true), allocates reusable signature I/O buffers, and
  // populates `identity`.
  static Expected<std::unique_ptr<Execution>> Create(
      Environment& environment, const BackendOptions& backend,
      const std::string& path, int signature_index, bool reference_cpu,
      ExecutionIdentity& identity);

  // Writes raw `inputs` into the bound input buffers, executes `warmups`
  // untimed runs followed by one timed run (recording latency in
  // `milliseconds`), and returns finite dequantized `float` outputs.
  Expected<std::vector<std::vector<float>>> Invoke(
      const std::vector<std::vector<uint8_t>>& inputs, int warmups,
      double& milliseconds);

  // Returns tensor specifications for the active signature's inputs.
  const std::vector<TensorSpec>& inputs() const { return input_specs_; }

  // Returns tensor specifications for the active signature's outputs.
  const std::vector<TensorSpec>& outputs() const { return output_specs_; }

 private:
  // Buffers are destroyed before the model that owns their requirements.
  CompiledModel model_;
  std::vector<TensorBuffer> inputs_;
  std::vector<TensorBuffer> outputs_;
  std::vector<TensorSpec> input_specs_;
  std::vector<TensorSpec> output_specs_;
  int signature_;
};
}  // namespace litert::vision

#endif  // LITERT_TOOLS_VISION_EVAL_EXECUTION_H_
