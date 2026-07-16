// Copyright (c) 2026 The Brave Authors. All rights reserved.
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

// Drives BraveLitertPassageEmbedder through the passage_embeddings mojom
// PassageEmbedder interface, on CPU and (when the prebuilt accelerator is
// available) the Metal GPU. Same license-gated inputs as the LiteRT PoC:
//
//   npm run test -- brave_unit_tests --filter='*BraveLitertPassageEmbedder*' \
//     --litert-poc-model=/path/to/embeddinggemma-300M_seq256.tflite \
//     --litert-poc-tokenizer=/path/to/sentencepiece.model \
//     --litert-poc-runtime-lib-dir=\
//         src/third_party/litert/src/litert/prebuilt/macos_arm64
//
// Without the model/tokenizer switches the tests skip; without the
// runtime-lib-dir the Gpu/* instantiation skips.

#include "brave/browser/history_embeddings/brave_litert_passage_embedder.h"

#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "base/command_line.h"
#include "base/files/file_path.h"
#include "base/files/file_util.h"
#include "base/logging.h"
#include "base/test/task_environment.h"
#include "base/test/test_future.h"
#include "services/passage_embeddings/public/mojom/passage_embeddings.mojom.h"
#include "testing/gtest/include/gtest/gtest.h"

namespace brave_history_embeddings {
namespace {

constexpr char kModelPathSwitch[] = "litert-poc-model";
constexpr char kTokenizerSwitch[] = "litert-poc-tokenizer";
constexpr char kRuntimeLibDirSwitch[] = "litert-poc-runtime-lib-dir";

constexpr char kSentenceA[] = "The cat slept quietly on the warm windowsill.";
constexpr char kSentenceB[] = "A kitten dozed peacefully on the sunny ledge.";
constexpr char kSentenceC[] =
    "Quarterly revenue growth beat every analyst forecast this year.";

double CosineSimilarity(const std::vector<float>& a,
                        const std::vector<float>& b) {
  double dot = 0.0;
  double norm_a = 0.0;
  double norm_b = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    dot += static_cast<double>(a[i]) * b[i];
    norm_a += static_cast<double>(a[i]) * a[i];
    norm_b += static_cast<double>(b[i]) * b[i];
  }
  return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
}

}  // namespace

// Parameterized on use_gpu, so the Cpu/* and Gpu/* instantiations form two
// parallel suites.
class BraveLitertPassageEmbedderTest : public testing::TestWithParam<bool> {
 protected:
  bool use_gpu() const { return GetParam(); }

  void SetUp() override {
    const auto* cmd = base::CommandLine::ForCurrentProcess();
    const base::FilePath model_path = cmd->GetSwitchValuePath(kModelPathSwitch);
    const base::FilePath tokenizer_path =
        cmd->GetSwitchValuePath(kTokenizerSwitch);
    if (model_path.empty() || tokenizer_path.empty()) {
      GTEST_SKIP() << "Pass --" << kModelPathSwitch << " and --"
                   << kTokenizerSwitch << "; see file header.";
    }
    runtime_lib_dir_ = cmd->GetSwitchValuePath(kRuntimeLibDirSwitch);
    if (use_gpu() && runtime_lib_dir_.empty()) {
      GTEST_SKIP() << "Pass --" << kRuntimeLibDirSwitch
                   << " to run the GPU suite.";
    }

    std::optional<std::vector<uint8_t>> model =
        base::ReadFileToBytes(model_path);
    ASSERT_TRUE(model.has_value()) << "Cannot read " << model_path;
    model_ = std::move(*model);
    std::optional<std::vector<uint8_t>> sp =
        base::ReadFileToBytes(tokenizer_path);
    ASSERT_TRUE(sp.has_value()) << "Cannot read " << tokenizer_path;
    sentencepiece_ = std::move(*sp);
  }

  std::unique_ptr<BraveLitertPassageEmbedder> CreateEmbedder() {
    return BraveLitertPassageEmbedder::Create(model_, sentencepiece_, use_gpu(),
                                              runtime_lib_dir_);
  }

  // Synchronously embeds `passages` through the mojom interface.
  std::vector<passage_embeddings::mojom::PassageEmbeddingsResultPtr> Embed(
      BraveLitertPassageEmbedder* embedder,
      const std::vector<std::string>& passages) {
    base::test::TestFuture<
        std::vector<passage_embeddings::mojom::PassageEmbeddingsResultPtr>>
        future;
    embedder->GenerateEmbeddings(
        passages, passage_embeddings::mojom::PassagePriority::kUserInitiated,
        future.GetCallback());
    return future.Take();
  }

  base::test::TaskEnvironment task_environment_;
  std::vector<uint8_t> model_;
  std::vector<uint8_t> sentencepiece_;
  base::FilePath runtime_lib_dir_;
};

// The embedder produces one finite, non-trivial embedding per passage.
TEST_P(BraveLitertPassageEmbedderTest, EmbedsPassages) {
  std::unique_ptr<BraveLitertPassageEmbedder> embedder = CreateEmbedder();
  ASSERT_TRUE(embedder);

  auto results = Embed(embedder.get(), {kSentenceA, kSentenceB});
  ASSERT_EQ(results.size(), 2u);
  for (const auto& result : results) {
    ASSERT_FALSE(result->embeddings.empty());
    bool any_nonzero = false;
    for (float value : result->embeddings) {
      ASSERT_TRUE(std::isfinite(value));
      any_nonzero |= (value != 0.0f);
    }
    EXPECT_TRUE(any_nonzero);
  }
}

// The embeddings carry semantics: paraphrases (A, B) are closer than an
// unrelated sentence (C).
TEST_P(BraveLitertPassageEmbedderTest, SimilarPassagesAreCloserThanUnrelated) {
  std::unique_ptr<BraveLitertPassageEmbedder> embedder = CreateEmbedder();
  ASSERT_TRUE(embedder);

  auto results = Embed(embedder.get(), {kSentenceA, kSentenceB, kSentenceC});
  ASSERT_EQ(results.size(), 3u);
  const double similar =
      CosineSimilarity(results[0]->embeddings, results[1]->embeddings);
  const double unrelated =
      CosineSimilarity(results[0]->embeddings, results[2]->embeddings);
  LOG(INFO) << "litert-embedder cosine(A,B)=" << similar
            << " cosine(A,C)=" << unrelated;
  EXPECT_GT(similar, unrelated);
}

INSTANTIATE_TEST_SUITE_P(Cpu,
                         BraveLitertPassageEmbedderTest,
                         testing::Values(false));
INSTANTIATE_TEST_SUITE_P(Gpu,
                         BraveLitertPassageEmbedderTest,
                         testing::Values(true));

}  // namespace brave_history_embeddings
