// Copyright (c) 2026 The Brave Authors. All rights reserved.
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#include "brave/browser/history_embeddings/brave_litert_passage_embedder.h"

#include <algorithm>
#include <utility>

#include "base/logging.h"
#include "base/memory/ptr_util.h"
#include "third_party/abseil-cpp/absl/strings/string_view.h"
#include "third_party/litert/src/litert/cc/litert_buffer_ref.h"
#include "third_party/litert/src/litert/cc/litert_common.h"
#include "third_party/litert/src/litert/cc/litert_element_type.h"
#include "third_party/litert/src/litert/cc/litert_environment_options.h"
#include "third_party/litert/src/litert/cc/litert_event.h"
#include "third_party/litert/src/litert/cc/litert_layout.h"
#include "third_party/litert/src/litert/cc/litert_options.h"
#include "third_party/litert/src/litert/cc/litert_ranked_tensor_type.h"
#include "third_party/litert/src/litert/cc/litert_tensor_buffer.h"
#include "third_party/litert/src/litert/cc/litert_tensor_buffer_types.h"
#include "third_party/litert/src/litert/cc/options/litert_gpu_options.h"
#include "third_party/sentencepiece/src/src/sentencepiece_processor.h"

namespace brave_history_embeddings {

namespace {

// [bos] + [eos] control tokens, matching the upstream GemmaModelExecutor.
constexpr size_t kControlTokensCount = 2;

size_t ElementCount(const litert::RankedTensorType& type) {
  size_t elements = 1;
  for (auto dim : type.Layout().Dimensions()) {
    elements *= static_cast<size_t>(dim);
  }
  return elements;
}

size_t ElementBytes(litert::ElementType type) {
  return type == litert::ElementType::Int64 ? sizeof(int64_t) : sizeof(int32_t);
}

}  // namespace

BraveLitertPassageEmbedder::BraveLitertPassageEmbedder(
    std::vector<uint8_t> tflite_model,
    litert::Environment environment,
    litert::CompiledModel model,
    std::unique_ptr<sentencepiece::SentencePieceProcessor> tokenizer,
    size_t input_window_size)
    : tflite_model_(std::move(tflite_model)),
      environment_(std::move(environment)),
      model_(std::move(model)),
      tokenizer_(std::move(tokenizer)),
      input_window_size_(input_window_size) {}

BraveLitertPassageEmbedder::~BraveLitertPassageEmbedder() = default;

// static
std::unique_ptr<BraveLitertPassageEmbedder> BraveLitertPassageEmbedder::Create(
    base::span<const uint8_t> tflite_model,
    base::span<const uint8_t> sentencepiece_model,
    bool use_gpu,
    const base::FilePath& gpu_runtime_lib_dir) {
  auto tokenizer = std::make_unique<sentencepiece::SentencePieceProcessor>();
  const auto load_status = tokenizer->LoadFromSerializedProto(absl::string_view(
      reinterpret_cast<const char*>(sentencepiece_model.data()),
      sentencepiece_model.size()));
  if (!load_status.ok()) {
    LOG(ERROR) << "LiteRT embedder: cannot load SentencePiece model: "
               << load_status.ToString();
    return nullptr;
  }

  // The runtime references the model bytes past compilation; keep our own copy.
  std::vector<uint8_t> model_bytes(tflite_model.begin(), tflite_model.end());

  std::vector<litert::EnvironmentOptions::Option> env_options;
  const std::string runtime_lib_dir = gpu_runtime_lib_dir.AsUTF8Unsafe();
  if (use_gpu && !runtime_lib_dir.empty()) {
    // gpu_registry composes the accelerator path from kRuntimeLibraryDir.
    env_options.emplace_back(
        litert::EnvironmentOptions::Tag::kRuntimeLibraryDir,
        runtime_lib_dir.c_str());
  }
  auto environment = litert::Environment::Create(litert::EnvironmentOptions(
      litert::Span<const litert::EnvironmentOptions::Option>(
          env_options.data(), env_options.size())));
  if (!environment) {
    LOG(ERROR) << "LiteRT embedder: cannot create environment: "
               << environment.Error().Message();
    return nullptr;
  }

  auto compile_options = litert::Options::Create();
  if (!compile_options) {
    LOG(ERROR) << "LiteRT embedder: cannot create options: "
               << compile_options.Error().Message();
    return nullptr;
  }
  if (use_gpu) {
    compile_options->SetHardwareAccelerators(litert::HwAccelerators::kGpu);
    auto gpu_options = compile_options->GetGpuOptions();
    if (!gpu_options) {
      LOG(ERROR) << "LiteRT embedder: cannot get GPU options: "
                 << gpu_options.Error().Message();
      return nullptr;
    }
    // EmbeddingGemma is mixed-precision (quantized FC/Conv weights). Both knobs
    // are required for correct GPU output: without the quantized-ops option the
    // accelerator mishandles the quantized weights, and the default fp16
    // precision collapses the dequantized values to zero.
    gpu_options->EnableAllowSrcQuantizedFcConvOps(true);
    if (!gpu_options->SetPrecision(litert::GpuOptions::Precision::kFp32)) {
      LOG(ERROR) << "LiteRT embedder: cannot set fp32 precision.";
      return nullptr;
    }
  } else {
    compile_options->SetHardwareAccelerators(litert::HwAccelerators::kCpu);
  }

  auto model = litert::CompiledModel::Create(
      *environment,
      litert::BufferRef<uint8_t>(model_bytes.data(), model_bytes.size()),
      *compile_options);
  if (!model) {
    LOG(ERROR) << "LiteRT embedder: CompiledModel::Create failed: "
               << model.Error().Message();
    return nullptr;
  }

  // Derive the token window from the model's single input tensor.
  auto input_buffers = model->CreateInputBuffers();
  if (!input_buffers || input_buffers->size() != 1) {
    LOG(ERROR) << "LiteRT embedder: expected a single input tensor.";
    return nullptr;
  }
  auto input_type = (*input_buffers)[0].TensorType();
  if (!input_type) {
    LOG(ERROR) << "LiteRT embedder: cannot read input tensor type.";
    return nullptr;
  }
  const size_t input_window_size = ElementCount(*input_type);

  return base::WrapUnique(new BraveLitertPassageEmbedder(
      std::move(model_bytes), std::move(*environment), std::move(*model),
      std::move(tokenizer), input_window_size));
}

bool BraveLitertPassageEmbedder::Tokenize(const std::string& text,
                                          std::vector<int32_t>* out) const {
  if (input_window_size_ < kControlTokensCount) {
    return false;
  }
  std::vector<int> ids;
  if (!tokenizer_->Encode(text, &ids).ok()) {
    return false;
  }
  const int pad_id = tokenizer_->pad_id() >= 0 ? tokenizer_->pad_id() : 0;
  out->clear();
  out->reserve(input_window_size_);
  out->push_back(tokenizer_->bos_id());
  const size_t max_content = input_window_size_ - kControlTokensCount;
  for (size_t i = 0; i < std::min(ids.size(), max_content); ++i) {
    out->push_back(ids[i]);
  }
  out->push_back(tokenizer_->eos_id());
  out->resize(input_window_size_, pad_id);
  return true;
}

bool BraveLitertPassageEmbedder::Embed(const std::vector<int32_t>& tokens,
                                       std::vector<float>* embedding) {
  auto ref_inputs = model_.CreateInputBuffers();
  auto ref_outputs = model_.CreateOutputBuffers();
  if (!ref_inputs || !ref_outputs || ref_inputs->size() != 1) {
    return false;
  }

  // Run against host-memory I/O buffers so the runtime owns any layout
  // conversion (GPU PHWC4) and host<->device transfer.
  std::vector<litert::TensorBuffer> inputs;
  std::vector<litert::TensorBuffer> outputs;
  auto make_host_buffer = [&](const litert::TensorBuffer& reference,
                              std::vector<litert::TensorBuffer>* out) -> bool {
    auto type = reference.TensorType();
    if (!type) {
      return false;
    }
    auto buffer = litert::TensorBuffer::CreateManaged(
        environment_, litert::TensorBufferType::kHostMemory, *type,
        ElementCount(*type) * ElementBytes((*type).ElementType()));
    if (!buffer) {
      return false;
    }
    out->push_back(std::move(*buffer));
    return true;
  };
  if (!make_host_buffer((*ref_inputs)[0], &inputs)) {
    return false;
  }
  for (const litert::TensorBuffer& ref : *ref_outputs) {
    if (!make_host_buffer(ref, &outputs)) {
      return false;
    }
  }

  auto input_type = inputs[0].TensorType();
  if (!input_type) {
    return false;
  }
  if ((*input_type).ElementType() == litert::ElementType::Int64) {
    std::vector<int64_t> tokens64(tokens.begin(), tokens.end());
    if (!inputs[0].Write<int64_t>(
            litert::Span<const int64_t>(tokens64.data(), tokens64.size()))) {
      return false;
    }
  } else if (!inputs[0].Write<int32_t>(
                 litert::Span<const int32_t>(tokens.data(), tokens.size()))) {
    return false;
  }

  // Run asynchronously and wait on any per-output completion event before
  // reading, so a GPU read-back does not race device work.
  bool async = true;
  if (!model_.RunAsync(inputs, outputs, async)) {
    return false;
  }
  for (litert::TensorBuffer& buf : outputs) {
    if (!buf.HasEvent()) {
      continue;
    }
    auto event = buf.GetEvent();
    if (!event || !event->Wait()) {
      return false;
    }
  }

  embedding->clear();
  for (litert::TensorBuffer& buf : outputs) {
    auto type = buf.TensorType();
    if (!type || (*type).ElementType() != litert::ElementType::Float32) {
      continue;
    }
    auto packed = buf.PackedSize();
    if (!packed) {
      return false;
    }
    std::vector<float> values(*packed / sizeof(float));
    if (!buf.Read<float>(litert::Span<float>(values.data(), values.size()))) {
      return false;
    }
    embedding->insert(embedding->end(), values.begin(), values.end());
  }
  return !embedding->empty();
}

void BraveLitertPassageEmbedder::GenerateEmbeddings(
    const std::vector<std::string>& passages,
    passage_embeddings::mojom::PassagePriority priority,
    GenerateEmbeddingsCallback callback) {
  std::vector<passage_embeddings::mojom::PassageEmbeddingsResultPtr> results;
  results.reserve(passages.size());
  for (const std::string& passage : passages) {
    std::vector<int32_t> tokens;
    std::vector<float> embedding;
    if (!Tokenize(passage, &tokens) || !Embed(tokens, &embedding)) {
      // Per the mojom contract, a failure returns an empty results array.
      std::move(callback).Run({});
      return;
    }
    auto result = passage_embeddings::mojom::PassageEmbeddingsResult::New();
    result->embeddings = std::move(embedding);
    results.push_back(std::move(result));
  }
  std::move(callback).Run(std::move(results));
}

}  // namespace brave_history_embeddings
