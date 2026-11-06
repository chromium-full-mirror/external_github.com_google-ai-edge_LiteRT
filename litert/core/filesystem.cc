// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "litert/core/filesystem.h"

#include <chrono>  // NOLINT
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>  // NOLINT
#include <limits>
#include <string>
#include <system_error>  // NOLINT
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_expected.h"

namespace litert::internal {

namespace {

using StdPath = std::filesystem::path;

StdPath MakeStdPath(absl::string_view path) {
  return StdPath(std::string(path.begin(), path.end()));
}

bool StdExists(const StdPath& std_path) {
  return std::filesystem::exists(std_path);
}

size_t StdSize(const StdPath& std_path) {
  return std::filesystem::file_size(std_path);
}

LiteRtStatus StdIFRead(const StdPath& std_path, char* data, size_t size) {
  std::FILE* f = std::fopen(std_path.string().c_str(), "rb");
  if (!f) {
    return kLiteRtStatusErrorFileIO;
  }

  size_t read_bytes = std::fread(data, 1, size, f);
  std::fclose(f);
  if (read_bytes != size) {
    return kLiteRtStatusErrorFileIO;
  }

  return kLiteRtStatusOk;
}

}  // namespace

void Touch(absl::string_view path) {
  if (std::FILE* f = std::fopen(MakeStdPath(path).string().c_str(), "wb")) {
    std::fclose(f);
  }
}

std::string Join(const std::vector<absl::string_view>& paths) {
  StdPath std_path;
  for (auto subpath : paths) {
    std_path /= MakeStdPath(subpath);
  }
  return std_path.generic_string();
}

std::string Stem(absl::string_view path) {
  return MakeStdPath(path).stem().generic_string();
}

bool Exists(absl::string_view path) { return StdExists(MakeStdPath(path)); }

Expected<bool> ExistsWithStatus(absl::string_view path) {
  std::error_code ec;
  const bool exists = std::filesystem::exists(MakeStdPath(path), ec);
  if (ec) {
    return Error(kLiteRtStatusErrorFileIO,
                 absl::StrCat("Cannot inspect ", path, ": ", ec.message()));
  }
  return exists;
}

Expected<bool> IsRegularFile(absl::string_view path) {
  std::error_code ec;
  const bool regular = std::filesystem::is_regular_file(MakeStdPath(path), ec);
  if (ec == std::errc::no_such_file_or_directory ||
      ec == std::errc::not_a_directory) {
    return false;
  }
  if (ec) {
    return Error(kLiteRtStatusErrorFileIO,
                 absl::StrCat("Cannot inspect ", path, ": ", ec.message()));
  }
  return regular;
}

Expected<std::string> AbsolutePath(absl::string_view path) {
  std::error_code ec;
  const auto absolute = std::filesystem::absolute(MakeStdPath(path), ec);
  if (ec) {
    return Error(kLiteRtStatusErrorFileIO,
                 absl::StrCat("Cannot resolve ", path, ": ", ec.message()));
  }
  return absolute.lexically_normal().generic_string();
}

Expected<std::string> CanonicalPath(absl::string_view path) {
  std::error_code ec;
  const auto canonical = std::filesystem::canonical(MakeStdPath(path), ec);
  if (ec) {
    return Error(kLiteRtStatusErrorFileIO,
                 absl::StrCat("Cannot resolve ", path, ": ", ec.message()));
  }
  return canonical.generic_string();
}

Expected<void> RenameFile(absl::string_view source,
                         absl::string_view destination) {
#if defined(_WIN32)
  if (!MoveFileExW(MakeStdPath(source).c_str(), MakeStdPath(destination).c_str(),
                  MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
    return Error(kLiteRtStatusErrorFileIO,
                 absl::StrCat("Cannot rename ", source, " to ", destination,
                              ": Windows error ", GetLastError()));
  }
#else
  std::error_code ec;
  std::filesystem::rename(MakeStdPath(source), MakeStdPath(destination), ec);
  if (ec) {
    return Error(kLiteRtStatusErrorFileIO,
                 absl::StrCat("Cannot rename ", source, " to ", destination,
                              ": ", ec.message()));
  }
#endif
  return {};
}

Expected<size_t> Size(absl::string_view path) {
  const auto std_path = MakeStdPath(path);
  std::error_code ec;
  if (std::filesystem::is_directory(std_path, ec)) {
    return Error(kLiteRtStatusErrorFileIO,
                 absl::StrCat("Cannot get file size of directory: ", path));
  }
  ec.clear();
  const auto bytes = std::filesystem::file_size(std_path, ec);
  if (ec) {
    const auto status = ec == std::errc::no_such_file_or_directory ||
                                ec == std::errc::not_a_directory
                            ? kLiteRtStatusErrorNotFound
                            : kLiteRtStatusErrorFileIO;
    return Error(
        status,
        absl::StrCat("Cannot get file size: ", path, ": ", ec.message()));
  }
  if (bytes > std::numeric_limits<size_t>::max()) {
    return Error(kLiteRtStatusErrorFileIO, "File size exceeds size_t");
  }
  return static_cast<size_t>(bytes);
}

Expected<absl::Time> GetLastWriteTime(absl::string_view path) {
  auto std_path = MakeStdPath(path);
  std::error_code ec;
  auto ftime = std::filesystem::last_write_time(std_path, ec);
  if (ec) {
    return Error(kLiteRtStatusErrorFileIO,
                 absl::StrCat("Failed to get last write time: ", path,
                              ", error: ", ec.message()));
  }
  return absl::Now() +
         absl::FromChrono(ftime -
                          std::filesystem::file_time_type::clock::now());
}

Expected<void> TouchFile(absl::string_view path) {
  auto std_path = MakeStdPath(path);
  std::error_code ec;
  std::filesystem::last_write_time(
      std_path, std::filesystem::file_time_type::clock::now(), ec);
  if (ec) {
    return Error(kLiteRtStatusErrorFileIO,
                 absl::StrCat("Failed to touch file: ", path,
                              ", error: ", ec.message()));
  }
  return {};
}

Expected<OwningBufferRef<uint8_t>> LoadBinaryFile(absl::string_view path) {
  auto std_path = MakeStdPath(path);

  if (!StdExists(std_path)) {
    return Error(kLiteRtStatusErrorNotFound,
                 absl::StrCat("File not found: ", path));
  }

  OwningBufferRef<uint8_t> buf(StdSize(std_path));
  if (auto status = StdIFRead(std_path, buf.StrData(), buf.Size());
      status != kLiteRtStatusOk) {
    return Error(status, absl::StrCat("Failed to read: ", path));
  }

  return buf;
}

Expected<std::vector<std::string>> ListDir(absl::string_view path) {
  std::error_code ec;
  std::filesystem::directory_iterator it(MakeStdPath(path), ec), end;
  if (ec) {
    return Error(ec == std::errc::no_such_file_or_directory
                     ? kLiteRtStatusErrorNotFound
                     : kLiteRtStatusErrorFileIO,
                 absl::StrCat("Cannot list directory: ", path, ": ", ec.message()));
  }
  std::vector<std::string> res;
  while (it != end) {
    const bool regular = it->is_regular_file(ec);
    if (ec) break;
    if (regular) res.push_back(it->path().generic_string());
    it.increment(ec);
    if (ec) break;
  }
  if (ec) {
    return Error(kLiteRtStatusErrorFileIO,
                 absl::StrCat("Cannot list directory: ", path, ": ", ec.message()));
  }
  return res;
}

Expected<std::vector<std::string>> RecursiveListDir(absl::string_view path) {
  auto std_path = MakeStdPath(path);
  if (!StdExists(std_path)) {
    return Error(kLiteRtStatusErrorNotFound,
                 absl::StrCat("Directory not found: ", path));
  }
  std::vector<std::string> res;
  std::error_code ec;
  for (const auto& entry :
       std::filesystem::recursive_directory_iterator(std_path, ec)) {
    if (std::filesystem::is_regular_file(entry)) {
      res.push_back(entry.path().generic_string());
    }
  }
  if (ec) {
    return Error(
        kLiteRtStatusErrorFileIO,
        absl::StrCat("Recursive directory iteration failed: ", ec.message()));
  }
  return res;
}

Expected<std::string> Filename(absl::string_view path) {
  auto std_path = MakeStdPath(path);
  if (!StdExists(std_path)) {
    return Error(kLiteRtStatusErrorNotFound,
                 absl::StrCat("File not found: ", path));
  }
  return std_path.filename().generic_string();
}

bool IsDir(absl::string_view path) {
  auto std_path = MakeStdPath(path);
  return std::filesystem::is_directory(std_path);
}

Expected<void> MkDir(absl::string_view path) {
  std::error_code ec;
  std::filesystem::create_directories(MakeStdPath(path), ec);
  if (ec) {
    return Error(ec == std::errc::file_exists ? kLiteRtStatusErrorAlreadyExists
                                            : kLiteRtStatusErrorFileIO,
                 absl::StrCat("Failed to create directory: ", path, ": ",
                              ec.message()));
  }
  return {};
}

Expected<std::string> Parent(absl::string_view path) {
  auto std_path = MakeStdPath(path);
  return std_path.parent_path().generic_string();
}

Expected<void> RemoveFile(absl::string_view path) {
  auto std_path = MakeStdPath(path);
  std::error_code ec;
  std::filesystem::remove(std_path, ec);
  if (ec) {
    return Error(kLiteRtStatusErrorFileIO,
                 absl::StrCat("Failed to remove file: ", path,
                              ", error: ", ec.message()));
  }
  return {};
}

Expected<std::string> Relative(absl::string_view path, absl::string_view base) {
  auto std_path = MakeStdPath(path);
  auto std_base = MakeStdPath(base);
  std::error_code ec;
  auto rel = std::filesystem::relative(std_path, std_base, ec);
  if (ec) {
    return Error(kLiteRtStatusErrorFileIO,
                 absl::StrCat("Failed to compute relative path for ", path,
                              " w.r.t ", base, ": ", ec.message()));
  }
  return rel.generic_string();
}

Expected<void> RmDir(std::string path_to_remove) {
  std::error_code error_code;
  std::uintmax_t count =
      std::filesystem::remove_all(path_to_remove, error_code);
  if (error_code) {
    return Error(kLiteRtStatusErrorFileIO,
                 absl::StrCat("Could not remove: ", path_to_remove,
                              ", error: ", error_code.message()));
  }

  auto remaining = ExistsWithStatus(path_to_remove);
  if (!remaining) return remaining.Error();
  if (!*remaining) {
    if (count > 0) {
      LITERT_LOG(LITERT_INFO,
                 "Successfully removed directory and its contents: %s (%ju "
                 "items deleted)",
                 path_to_remove.c_str(), count);
    }
    // If count == 0 and it doesn't exist, it means it never existed. Still Ok.
    return {};
  } else {
    return Error(kLiteRtStatusErrorFileIO,
                 absl::StrCat("Could not fully remove: ", path_to_remove));
  }
}

}  // namespace litert::internal
