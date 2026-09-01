/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <cstddef>
#include <memory>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project  // IWYU pragma: keep
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo  // IWYU pragma: keep
#include "stablehlo/transforms/optimization/Passes.h"  // from @stablehlo
#include "tflite/converter/stablehlo/transforms/stablehlo_passes.h"

namespace mlir {
namespace odml {
namespace {

#define GEN_PASS_DEF_SIMPLIFYHLOPASS
#include "tflite/converter/stablehlo/transforms/stablehlo_passes.h.inc"

// Debug-name suffixes of StableHLO aggressive folder patterns that are
// excluded below, because MHLO had no equivalent and they change the converted
// graph:
// - FoldTransposeOpPattern: `mhlo.transpose` only folded splats and identity
//   permutations, while this materializes the transpose of any constant element
//   by element, ignoring `foldOpElementLimit`. On large weights (e.g.
//   `F.linear` kernels) that dominates conversion time. Splats are folded by
//   prepare-hlo's `FoldSplatTranspose` and identity permutations by the
//   canonicalization patterns below.
// - FoldConcatenateAdjacentSplatsOpPattern: merges adjacent identical splat
//   operands of `concatenate`, changing the operand structure of the emitted
//   `tfl.concatenation` ops.
constexpr llvm::StringLiteral kExcludedFolderPatterns[] = {
    "::FoldTransposeOpPattern",
    "::FoldConcatenateAdjacentSplatsOpPattern",
};

class SimplifyHloPass : public impl::SimplifyHloPassBase<SimplifyHloPass> {
 public:
  void runOnOperation() override;
};

// transpose(broadcast_in_dim(x)) -> broadcast_in_dim(x) with permuted
// broadcast dimensions. Port of MHLO's `EliminateBroadcastInDimTranspose`
// canonicalization. StableHLO instead rewrites such transposes into reshapes
// when they only move unit dimensions (`TransposeIsReshape`), which then block
// merging the broadcast with following broadcasts and leave a
// `tfl.broadcast_to` that can't be fused into its users. Given a higher benefit
// so that it is tried before `TransposeIsReshape`.
struct TransposeOfBroadcastInDim
    : public OpRewritePattern<stablehlo::TransposeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::TransposeOp op,
                                PatternRewriter& rewriter) const override {
    auto broadcast =
        op.getOperand().getDefiningOp<stablehlo::BroadcastInDimOp>();
    if (!broadcast) return failure();
    ArrayRef<int64_t> permutation = op.getPermutation();
    SmallVector<int64_t> broadcast_dims;
    for (int64_t dim : broadcast.getBroadcastDimensions()) {
      broadcast_dims.push_back(llvm::find(permutation, dim) -
                               permutation.begin());
    }
    rewriter.replaceOpWithNewOp<stablehlo::BroadcastInDimOp>(
        op, op.getType(), broadcast.getOperand(), broadcast_dims);
    return success();
  }
};

void SimplifyHloPass::runOnOperation() {
  MLIRContext* context = &getContext();
  RewritePatternSet patterns(context);

  // Fold float constants too, as MHLO's folders did (e.g. the `sign`,
  // `compare` and `select` on constants in JAX's floor_divide lowering, which
  // the floor_div legalization patterns expect to be folded).
  stablehlo::StablehloAggressiveFolderPassOptions folder_options;
  folder_options.optimizeFloat = true;
  stablehlo::populateStablehloAggressiveFolderPatterns(context, &patterns,
                                                       folder_options);

  // Exclude the patterns listed above. They are found by debug name, so fail
  // loudly if any of them ever stops matching.
  auto& native_patterns = patterns.getNativePatterns();
  for (llvm::StringLiteral name : kExcludedFolderPatterns) {
    const size_t num_patterns = native_patterns.size();
    llvm::erase_if(native_patterns,
                   [&](const std::unique_ptr<RewritePattern>& pattern) {
                     return pattern->getDebugName().ends_with(name);
                   });
    const size_t num_erased = num_patterns - native_patterns.size();
    if (num_erased != 1) {
      getOperation().emitError()
          << "expected to exclude exactly one StableHLO pattern named '*"
          << name << "', but excluded " << num_erased;
      return signalPassFailure();
    }
  }

  stablehlo::populateStablehloCanonicalizationPatterns(context, &patterns);
  patterns.add<TransposeOfBroadcastInDim>(context, /*benefit=*/10);

  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> CreateSimplifyHloPass() {
  return std::make_unique<SimplifyHloPass>();
}

static PassRegistration<SimplifyHloPass> pass;

}  // namespace odml
}  // namespace mlir
