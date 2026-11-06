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

#include "litert/tools/vision_eval/data.h"

#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "litert/cc/litert_expected.h"
#include "litert/core/filesystem.h"

namespace litert::vision {
namespace {
namespace fs = ::litert::internal;

class DataTest : public testing::Test {
 protected:
  void SetUp() override {
    const auto* test = testing::UnitTest::GetInstance()->current_test_info();
    root_ = fs::Join({testing::TempDir(), test->name()});
    ASSERT_TRUE(fs::MkDir(root_));
  }
  Expected<EvaluationConfig> Args(std::vector<std::string> arguments) {
    arguments.insert(arguments.begin(), "litert_vision_eval");
    std::vector<char*> argv;
    for (auto& argument : arguments) argv.push_back(argument.data());
    return ParseArgs(argv.size(), argv.data());
  }
  std::string root_;
};

TEST_F(DataTest, TextAndBinaryRoundTripPreserveOptionalPresence) {
  RunSummary summary;
  summary.set_status(FAILED);
  summary.set_error("No valid depth pixels");
  summary.add_metrics()->mutable_depth()->set_empty_images(1);
  for (auto format : {TEXT_PROTO, BINARY_PROTO}) {
    const auto path = fs::Join({root_, (format == TEXT_PROTO ? "out.txtpb" : "out.binpb")});
    ASSERT_TRUE(WriteProto(path, summary, format));
    RunSummary parsed;
    ASSERT_TRUE(ReadProto(path, parsed));
    EXPECT_EQ(parsed.SerializeAsString(), summary.SerializeAsString());
    EXPECT_FALSE(parsed.metrics(0).depth().has_abs_rel());
  }
  ASSERT_TRUE(WriteProto(fs::Join({root_, "missing/out"}), summary, TEXT_PROTO));
  EXPECT_FALSE(ReadProto(fs::Join({root_, "missing/out"}), summary));
}

TEST_F(DataTest, ConfigLoadsBeforeExplicitOverrides) {
  EvaluationConfig config;
  config.set_manifest("samples.txtpb");
  config.set_output_dir("results");
  config.set_model("model.tflite");
  config.mutable_backend()->set_accelerator("npu");
  config.mutable_backend()->set_num_threads(8);
  config.mutable_preprocessing()->add_mean(0);
  config.mutable_preprocessing()->add_mean(0);
  config.mutable_preprocessing()->add_mean(0);
  const auto path = fs::Join({root_, "run.txtpb"});
  ASSERT_TRUE(WriteProto(path, config, TEXT_PROTO));
  auto parsed = Args({"--num_threads=2", "--config=" + path,
                      "--accelerator=cpu", "--summary_format=binaryproto"});
  ASSERT_TRUE(parsed);
  EXPECT_EQ(parsed->backend().num_threads(), 2);
  EXPECT_EQ(parsed->backend().accelerator(), "cpu");
  EXPECT_EQ(parsed->summary_format(), BINARY_PROTO);
  EXPECT_EQ(parsed->preprocessing().mean_size(), 3);
  EXPECT_FALSE(Args({"--config=" + path, "--unknown=1"}));
  EXPECT_FALSE(Args({"--config=" + path, "--warmup_runs=-1"}));
  EXPECT_FALSE(Args({"--config=" + path, "--atol=nan"}));
  EXPECT_FALSE(Args({"--config=" + path, "--dump_inputs=maybe"}));
  EXPECT_FALSE(Args({"--config=" + path, "--task=classification", "--task=depth"}));
}

TEST_F(DataTest, RawTensorSizeAndFileExistenceAreChecked) {
  TensorSpec spec;
  spec.set_type(FLOAT32);
  spec.add_shape(1);
  spec.add_shape(2);
  float values[] = {1, 2};
  const auto path = fs::Join({root_, "input.raw"});
  ASSERT_TRUE(WriteBytes(path, values, sizeof(values)));
  auto read = ReadFloats(path, spec);
  ASSERT_TRUE(read);
  EXPECT_EQ(*read, (std::vector<float>{1, 2}));
  auto identity = Identify(path);
  ASSERT_TRUE(identity);
  EXPECT_TRUE(VerifyFile(*identity));
  EXPECT_FALSE(VerifyFile(FileIdentity()));
  ASSERT_TRUE(WriteBytes(path, values, sizeof(float)));
  EXPECT_FALSE(ReadTensor(path, spec));
  ASSERT_TRUE(fs::RemoveFile(path));
  EXPECT_FALSE(Identify(path));
  EXPECT_FALSE(VerifyFile(*identity));
  spec.set_shape(1, -1);
  EXPECT_FALSE(TensorBytes(spec));
  spec.set_shape(1, 2147483647);
  EXPECT_FALSE(TensorBytes(spec));
}

TEST_F(DataTest, ManifestRejectsDuplicateBindingsAndUnsafeIds) {
  DatasetManifest manifest;
  auto* input = manifest.add_inputs();
  input->set_name("image");
  input->add_shape(1);
  auto* sample = manifest.add_samples();
  sample->set_id("one");
  sample->set_label(0);
  auto* file = sample->add_inputs();
  file->set_name("image");
  file->mutable_file()->set_path("one/input.raw");
  EXPECT_TRUE(ValidateManifest(manifest, AUTO));
  *sample->add_inputs() = *file;
  EXPECT_FALSE(ValidateManifest(manifest, AUTO));
  sample->mutable_inputs()->RemoveLast();
  sample->set_id("../escape");
  EXPECT_FALSE(ValidateManifest(manifest, AUTO));
  EXPECT_FALSE(SafeName(""));
  EXPECT_FALSE(SafeName("a/b"));
  EXPECT_TRUE(SafeName("ade20k_validation_000003"));
}
}  // namespace
}  // namespace litert::vision
