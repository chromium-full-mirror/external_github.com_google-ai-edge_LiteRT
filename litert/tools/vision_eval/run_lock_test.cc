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

#include "litert/tools/vision_eval/run_lock.h"

#include <string>

#include <gtest/gtest.h>
#include "litert/core/filesystem.h"

namespace litert::vision {
namespace {
namespace fs = ::litert::internal;

TEST(RunLockTest, PreventsConcurrentLockAndReleasesOnDestructionOrRelease) {
  const std::string dir = fs::Join({testing::TempDir(), "run_lock_test"});
  ASSERT_TRUE(fs::MkDir(dir));
  const std::string lock_path = fs::Join({dir, "run.lock"});

  {
    RunLock first;
    ASSERT_TRUE(first.Acquire(lock_path));

    RunLock second;
    EXPECT_FALSE(second.Acquire(lock_path));

    first.Release();
    EXPECT_TRUE(second.Acquire(lock_path));
  }

  RunLock after_scope;
  EXPECT_TRUE(after_scope.Acquire(lock_path));
  // Re-acquiring on the same instance releases the old descriptor cleanly.
  EXPECT_TRUE(after_scope.Acquire(lock_path));
}

TEST(RunLockTest, FailsWhenParentDirectoryDoesNotExist) {
  const std::string missing_path =
      fs::Join({testing::TempDir(), "missing_parent_dir", "run.lock"});
  RunLock lock;
  EXPECT_FALSE(lock.Acquire(missing_path));
}

}  // namespace
}  // namespace litert::vision
