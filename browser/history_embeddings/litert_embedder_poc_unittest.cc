// Copyright (c) 2026 The Brave Authors. All rights reserved.
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

// Proof-of-concept: drive EmbeddingGemma through LiteRT's CompiledModel API on
// the CPU and the Metal GPU accelerator, embedding real sentences (tokenized
// with the model's SentencePiece vocabulary) and comparing the two backends.
//
// The .tflite and sentencepiece.model are large / license-gated, so they are
// not checked in. Pass them on the command line:
//
//   npm run test -- brave_unit_tests --filter='*LitertEmbedder*' \
//     --litert-poc-model=/path/to/embeddinggemma-300M_seq256.tflite \
//     --litert-poc-tokenizer=/path/to/sentencepiece.model
//
// Without those switches the tests skip. Use the plain (no vendor suffix)
// model; the .qualcomm/.mediatek/.google.tensor variants are NPU AOT builds.
//
// The GPU (Metal on macOS) suite additionally needs the prebuilt accelerator
// library; point LiteRT at its directory (fed to the runtime via the
// kRuntimeLibraryDir environment option, which gpu_registry uses to locate
// libLiteRt*Accelerator). Without it the Gpu/* and match tests skip:
//
//     --litert-poc-runtime-lib-dir=\
//         src/third_party/litert/src/litert/prebuilt/macos_arm64

#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "base/command_line.h"
#include "base/files/file_path.h"
#include "base/files/file_util.h"
#include "base/logging.h"
#include "testing/gtest/include/gtest/gtest.h"
#include "third_party/litert/src/litert/cc/litert_buffer_ref.h"
#include "third_party/litert/src/litert/cc/litert_common.h"
#include "third_party/litert/src/litert/cc/litert_compiled_model.h"
#include "third_party/litert/src/litert/cc/litert_element_type.h"
#include "third_party/litert/src/litert/cc/litert_environment.h"
#include "third_party/litert/src/litert/cc/litert_environment_options.h"
#include "third_party/litert/src/litert/cc/litert_event.h"
#include "third_party/litert/src/litert/cc/litert_layout.h"
#include "third_party/litert/src/litert/cc/litert_model.h"
#include "third_party/litert/src/litert/cc/litert_options.h"
#include "third_party/litert/src/litert/cc/litert_ranked_tensor_type.h"
#include "third_party/litert/src/litert/cc/litert_tensor_buffer.h"
#include "third_party/litert/src/litert/cc/litert_tensor_buffer_types.h"
#include "third_party/litert/src/litert/cc/options/litert_gpu_options.h"
#include "third_party/sentencepiece/src/src/sentencepiece_processor.h"

namespace brave_history_embeddings {
namespace {

constexpr char kModelPathSwitch[] = "litert-poc-model";
constexpr char kTokenizerSwitch[] = "litert-poc-tokenizer";
constexpr char kRuntimeLibDirSwitch[] = "litert-poc-runtime-lib-dir";

// Fixed sentences for the semantic sanity check: A and B are paraphrases, C is
// unrelated. A meaningful embedder puts A closer to B than to C.
constexpr char kSentenceA[] = "The cat slept quietly on the warm windowsill.";
constexpr char kSentenceB[] = "A kitten dozed peacefully on the sunny ledge.";
constexpr char kSentenceC[] =
    "Quarterly revenue growth beat every analyst "
    "forecast this year.";

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

// Shared helpers for the CPU/GPU fixtures. Methods use gtest ASSERT_* macros,
// so they must be invoked from within a running test (via
// ASSERT_NO_FATAL_FAILURE at the call sites).
class LitertEmbedderPocHelper {
 protected:
  // A LiteRT environment and a model compiled against it. The environment must
  // outlive the model, so they travel together.
  struct LoadedModel {
    std::optional<litert::Environment> env;
    std::optional<litert::CompiledModel> model;
  };

  // Reads the model bytes and loads the SentencePiece tokenizer. Callers gate
  // on HasInputs()/skip before calling this.
  void LoadInputs() {
    const auto* cmd = base::CommandLine::ForCurrentProcess();
    std::optional<std::vector<uint8_t>> bytes =
        base::ReadFileToBytes(cmd->GetSwitchValuePath(kModelPathSwitch));
    ASSERT_TRUE(bytes.has_value()) << "Cannot read the .tflite model.";
    // The runtime references the model bytes past Create(), so keep them alive
    // for the fixture's lifetime.
    model_data_ = std::move(*bytes);

    sp_ = std::make_unique<sentencepiece::SentencePieceProcessor>();
    const auto status =
        sp_->Load(cmd->GetSwitchValuePath(kTokenizerSwitch).AsUTF8Unsafe());
    ASSERT_TRUE(status.ok())
        << "Cannot load sentencepiece.model: " << status.ToString();
  }

  // Builds an environment + `CompiledModel` from `model_data_` for
  // `accelerator`.
  void BuildModel(litert::HwAccelerators accelerator, LoadedModel* out) {
    const auto* cmd = base::CommandLine::ForCurrentProcess();
    const std::string runtime_lib_dir =
        cmd->GetSwitchValuePath(kRuntimeLibDirSwitch).AsUTF8Unsafe();
    std::vector<litert::EnvironmentOptions::Option> options;
    if (!runtime_lib_dir.empty()) {
      // gpu_registry composes the accelerator path from kRuntimeLibraryDir.
      options.emplace_back(litert::EnvironmentOptions::Tag::kRuntimeLibraryDir,
                           runtime_lib_dir.c_str());
    }
    litert::EnvironmentOptions env_options(
        litert::Span<const litert::EnvironmentOptions::Option>(options.data(),
                                                               options.size()));
    auto env = litert::Environment::Create(env_options);
    ASSERT_TRUE(env) << env.Error().Message();
    out->env = std::move(*env);

    auto compile_options = litert::Options::Create();
    ASSERT_TRUE(compile_options) << compile_options.Error().Message();
    if (accelerator == litert::HwAccelerators::kGpu) {
      compile_options->SetHardwareAccelerators(litert::HwAccelerators::kGpu);
      auto gpu_options = compile_options->GetGpuOptions();
      ASSERT_TRUE(gpu_options) << gpu_options.Error().Message();
      // EmbeddingGemma is mixed-precision (quantized FC/Conv weights). Both of
      // these are required for non-zero GPU output: without the quantized-ops
      // option the accelerator mishandles the quantized weights, and with the
      // default fp16 precision the dequantized values collapse to zero.
      ASSERT_EQ(gpu_options->EnableAllowSrcQuantizedFcConvOps(true),
                kLiteRtStatusOk);
      ASSERT_TRUE(
          gpu_options->SetPrecision(litert::GpuOptions::Precision::kFp32))
          << "SetPrecision failed";
    } else {
      compile_options->SetHardwareAccelerators(accelerator);
    }

    auto model = litert::CompiledModel::Create(
        *out->env,
        litert::BufferRef<uint8_t>(model_data_.data(), model_data_.size()),
        *compile_options);
    ASSERT_TRUE(model) << "CompiledModel::Create failed: "
                       << model.Error().Message();
    out->model = std::move(*model);
  }

  // Tokenizes `text` into a `window`-length token id sequence, matching the
  // upstream GemmaModelExecutor layout: [bos] + tokens + [eos] + pad.
  void Tokenize(const std::string& text,
                size_t window,
                std::vector<int32_t>* out) {
    std::vector<int> ids;
    const auto status = sp_->Encode(text, &ids);
    ASSERT_TRUE(status.ok()) << status.ToString();
    ASSERT_GE(window, 2u);

    const int pad_id = sp_->pad_id() >= 0 ? sp_->pad_id() : 0;
    out->clear();
    out->reserve(window);
    out->push_back(sp_->bos_id());
    const size_t max_content = window - 2;
    for (size_t i = 0; i < std::min(ids.size(), max_content); ++i) {
      out->push_back(ids[i]);
    }
    out->push_back(sp_->eos_id());
    out->resize(window, pad_id);
  }

  static size_t ElementCount(const litert::RankedTensorType& type) {
    size_t elements = 1;
    for (auto dim : type.Layout().Dimensions()) {
      elements *= static_cast<size_t>(dim);
    }
    return elements;
  }

  // Creates a host-memory buffer matching `reference`'s tensor type (bound to
  // `env`) and appends it to `*out`.
  void MakeHostBuffer(litert::Environment& env,
                      const litert::TensorBuffer& reference,
                      std::vector<litert::TensorBuffer>* out) {
    auto type = reference.TensorType();
    ASSERT_TRUE(type) << type.Error().Message();
    const litert::ElementType element_type = (*type).ElementType();
    const size_t element_bytes = element_type == litert::ElementType::Int64
                                     ? sizeof(int64_t)
                                     : sizeof(int32_t);
    auto buffer = litert::TensorBuffer::CreateManaged(
        env, litert::TensorBufferType::kHostMemory, *type,
        ElementCount(*type) * element_bytes);
    ASSERT_TRUE(buffer) << buffer.Error().Message();
    out->push_back(std::move(*buffer));
  }

  // Embeds `text` with `lm` and writes the (single Float32) output to
  // `*embedding`.
  void EmbedText(LoadedModel& lm,
                 const std::string& text,
                 std::vector<float>* embedding) {
    litert::Environment& env = *lm.env;
    litert::CompiledModel& model = *lm.model;

    // Discover the signature's tensor types from the accelerator's own buffers,
    // then run against host-memory I/O buffers so the runtime owns any layout
    // conversion (GPU PHWC4) and host<->device transfer.
    auto ref_inputs = model.CreateInputBuffers();
    ASSERT_TRUE(ref_inputs) << ref_inputs.Error().Message();
    auto ref_outputs = model.CreateOutputBuffers();
    ASSERT_TRUE(ref_outputs) << ref_outputs.Error().Message();
    ASSERT_EQ(ref_inputs->size(), 1u)
        << "PoC assumes a single token-id input tensor.";

    auto input_type = (*ref_inputs)[0].TensorType();
    ASSERT_TRUE(input_type) << input_type.Error().Message();
    std::vector<int32_t> tokens;
    ASSERT_NO_FATAL_FAILURE(Tokenize(text, ElementCount(*input_type), &tokens));

    std::vector<litert::TensorBuffer> inputs;
    std::vector<litert::TensorBuffer> outputs;
    ASSERT_NO_FATAL_FAILURE(MakeHostBuffer(env, (*ref_inputs)[0], &inputs));
    for (const litert::TensorBuffer& ref : *ref_outputs) {
      ASSERT_NO_FATAL_FAILURE(MakeHostBuffer(env, ref, &outputs));
    }

    // Write token ids, honoring the input tensor's integer width.
    if ((*input_type).ElementType() == litert::ElementType::Int64) {
      std::vector<int64_t> tokens64(tokens.begin(), tokens.end());
      auto w = inputs[0].Write<int64_t>(
          litert::Span<const int64_t>(tokens64.data(), tokens64.size()));
      ASSERT_TRUE(w) << w.Error().Message();
    } else {
      auto w = inputs[0].Write<int32_t>(
          litert::Span<const int32_t>(tokens.data(), tokens.size()));
      ASSERT_TRUE(w) << w.Error().Message();
    }

    // Run asynchronously and wait on any per-output completion event before
    // reading, so a GPU read-back does not race device work. On CPU no event is
    // attached and this is effectively synchronous.
    bool async = true;
    auto status = model.RunAsync(inputs, outputs, async);
    ASSERT_TRUE(status) << "RunAsync failed: " << status.Error().Message();
    for (litert::TensorBuffer& buf : outputs) {
      if (!buf.HasEvent()) {
        continue;
      }
      auto event = buf.GetEvent();
      ASSERT_TRUE(event) << event.Error().Message();
      ASSERT_TRUE(event->Wait()) << "Waiting on output event failed.";
    }

    embedding->clear();
    for (litert::TensorBuffer& buf : outputs) {
      auto type = buf.TensorType();
      ASSERT_TRUE(type) << type.Error().Message();
      if ((*type).ElementType() != litert::ElementType::Float32) {
        continue;
      }
      auto packed = buf.PackedSize();
      ASSERT_TRUE(packed) << packed.Error().Message();
      std::vector<float> values(*packed / sizeof(float));
      auto read =
          buf.Read<float>(litert::Span<float>(values.data(), values.size()));
      ASSERT_TRUE(read) << read.Error().Message();
      embedding->insert(embedding->end(), values.begin(), values.end());
    }
    ASSERT_FALSE(embedding->empty()) << "No Float32 output produced.";
  }

  bool HasModelAndTokenizer() const {
    const auto* cmd = base::CommandLine::ForCurrentProcess();
    return !cmd->GetSwitchValuePath(kModelPathSwitch).empty() &&
           !cmd->GetSwitchValuePath(kTokenizerSwitch).empty();
  }

  bool HasGpuRuntimeLibDir() const {
    return !base::CommandLine::ForCurrentProcess()
                ->GetSwitchValuePath(kRuntimeLibDirSwitch)
                .empty();
  }

  std::vector<uint8_t> model_data_;
  std::unique_ptr<sentencepiece::SentencePieceProcessor> sp_;
};

// Runs the same checks on whichever accelerator the parameter selects, so the
// Cpu/* and Gpu/* instantiations form two parallel suites.
class BraveLitertEmbedderPocTest
    : public LitertEmbedderPocHelper,
      public testing::TestWithParam<litert::HwAccelerators> {
 protected:
  void SetUp() override {
    if (!HasModelAndTokenizer()) {
      GTEST_SKIP() << "Pass --" << kModelPathSwitch << " and --"
                   << kTokenizerSwitch << "; see file header.";
    }
    if (GetParam() == litert::HwAccelerators::kGpu && !HasGpuRuntimeLibDir()) {
      GTEST_SKIP() << "Pass --" << kRuntimeLibDirSwitch
                   << " to run the GPU suite.";
    }
    ASSERT_NO_FATAL_FAILURE(LoadInputs());
    ASSERT_NO_FATAL_FAILURE(BuildModel(GetParam(), &model_));
  }

  LoadedModel model_;
};

// The model compiles on the target accelerator and produces a finite,
// non-trivial embedding for a real sentence.
TEST_P(BraveLitertEmbedderPocTest, EmbedsRealText) {
  std::vector<float> embedding;
  ASSERT_NO_FATAL_FAILURE(EmbedText(model_, kSentenceA, &embedding));

  bool any_nonzero = false;
  for (float value : embedding) {
    ASSERT_TRUE(std::isfinite(value)) << "Non-finite value in embedding.";
    any_nonzero |= (value != 0.0f);
  }
  EXPECT_TRUE(any_nonzero) << "Embedding is all zeros.";
  LOG(INFO) << "litert-poc produced " << embedding.size() << " floats.";
}

// The embedding carries semantics: paraphrases (A, B) are more similar than an
// unrelated sentence (C).
TEST_P(BraveLitertEmbedderPocTest, SimilarSentencesAreCloserThanUnrelated) {
  std::vector<float> a;
  std::vector<float> b;
  std::vector<float> c;
  ASSERT_NO_FATAL_FAILURE(EmbedText(model_, kSentenceA, &a));
  ASSERT_NO_FATAL_FAILURE(EmbedText(model_, kSentenceB, &b));
  ASSERT_NO_FATAL_FAILURE(EmbedText(model_, kSentenceC, &c));

  const double similar = CosineSimilarity(a, b);
  const double unrelated = CosineSimilarity(a, c);
  LOG(INFO) << "litert-poc cosine(A,B)=" << similar
            << " cosine(A,C)=" << unrelated;
  EXPECT_GT(similar, unrelated)
      << "Paraphrase pair was not closer than the unrelated sentence.";
}

// Inference is deterministic for a fixed input.
TEST_P(BraveLitertEmbedderPocTest, IsDeterministic) {
  std::vector<float> a;
  std::vector<float> b;
  ASSERT_NO_FATAL_FAILURE(EmbedText(model_, kSentenceA, &a));
  ASSERT_NO_FATAL_FAILURE(EmbedText(model_, kSentenceA, &b));
  EXPECT_EQ(a, b) << "Same input produced different embeddings.";
}

INSTANTIATE_TEST_SUITE_P(Cpu,
                         BraveLitertEmbedderPocTest,
                         testing::Values(litert::HwAccelerators::kCpu));
INSTANTIATE_TEST_SUITE_P(Gpu,
                         BraveLitertEmbedderPocTest,
                         testing::Values(litert::HwAccelerators::kGpu));

// Cross-accelerator correctness: the GPU must run the whole graph (no CPU
// fallback) AND produce the same embedding as the CPU reference.
class BraveLitertEmbedderMatchTest : public LitertEmbedderPocHelper,
                                     public testing::Test {
 protected:
  void SetUp() override {
    if (!HasModelAndTokenizer() || !HasGpuRuntimeLibDir()) {
      GTEST_SKIP() << "Needs --" << kModelPathSwitch << ", --"
                   << kTokenizerSwitch << " and --" << kRuntimeLibDirSwitch
                   << ".";
    }
    ASSERT_NO_FATAL_FAILURE(LoadInputs());
  }
};

TEST_F(BraveLitertEmbedderMatchTest, GpuMatchesCpuReference) {
  LoadedModel cpu;
  LoadedModel gpu;
  ASSERT_NO_FATAL_FAILURE(BuildModel(litert::HwAccelerators::kCpu, &cpu));
  ASSERT_NO_FATAL_FAILURE(BuildModel(litert::HwAccelerators::kGpu, &gpu));

  // Prove it actually ran on the GPU: a fully-accelerated model left no ops on
  // CPU. If GPU registration/delegation had failed this would be false.
  auto fully = gpu.model->IsFullyAccelerated();
  ASSERT_TRUE(fully) << fully.Error().Message();
  EXPECT_TRUE(*fully) << "GPU model was not fully accelerated (CPU fallback).";

  std::vector<float> cpu_embedding;
  std::vector<float> gpu_embedding;
  ASSERT_NO_FATAL_FAILURE(EmbedText(cpu, kSentenceA, &cpu_embedding));
  ASSERT_NO_FATAL_FAILURE(EmbedText(gpu, kSentenceA, &gpu_embedding));
  ASSERT_EQ(cpu_embedding.size(), gpu_embedding.size());

  const double cosine = CosineSimilarity(cpu_embedding, gpu_embedding);
  LOG(INFO) << "litert-poc cpu/gpu cosine=" << cosine;
  EXPECT_GT(cosine, 0.99)
      << "GPU embedding diverges from the CPU reference (cosine=" << cosine
      << ").";
}

}  // namespace brave_history_embeddings
