// Copyright (c) 2026 The Brave Authors. All rights reserved.
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef BRAVE_BROWSER_HISTORY_EMBEDDINGS_BRAVE_LITERT_PASSAGE_EMBEDDER_H_
#define BRAVE_BROWSER_HISTORY_EMBEDDINGS_BRAVE_LITERT_PASSAGE_EMBEDDER_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "base/containers/span.h"
#include "base/files/file_path.h"
#include "services/passage_embeddings/public/mojom/passage_embeddings.mojom.h"
#include "third_party/litert/src/litert/cc/litert_compiled_model.h"
#include "third_party/litert/src/litert/cc/litert_environment.h"

namespace sentencepiece {
class SentencePieceProcessor;
}

namespace brave_history_embeddings {

// A native implementation of passage_embeddings::mojom::PassageEmbedder that
// runs EmbeddingGemma through LiteRT's CompiledModel (on the Metal GPU
// accelerator, or CPU), instead of the WASM worker used by
// BraveBatchPassageEmbedder. Inputs are tokenized with the model's
// SentencePiece vocabulary using the same layout as the upstream
// GemmaModelExecutor ([bos] + tokens + [eos] + pad).
//
// This is the "CompiledModel as the engine" path: it drops into
// BravePassageEmbeddingsService in place of the WASM embedder while the
// in-browser controller / SchedulingEmbedder / extraction pipeline stay the
// same.
class BraveLitertPassageEmbedder
    : public passage_embeddings::mojom::PassageEmbedder {
 public:
  ~BraveLitertPassageEmbedder() override;

  BraveLitertPassageEmbedder(const BraveLitertPassageEmbedder&) = delete;
  BraveLitertPassageEmbedder& operator=(const BraveLitertPassageEmbedder&) =
      delete;

  // Builds an embedder from the raw EmbeddingGemma `.tflite` bytes and the
  // SentencePiece model bytes. When `use_gpu` is true, the model is compiled
  // for the Metal GPU accelerator loaded from `gpu_runtime_lib_dir` (the
  // directory holding the prebuilt libLiteRt*Accelerator); otherwise it runs on
  // CPU. Returns nullptr on failure.
  static std::unique_ptr<BraveLitertPassageEmbedder> Create(
      base::span<const uint8_t> tflite_model,
      base::span<const uint8_t> sentencepiece_model,
      bool use_gpu,
      const base::FilePath& gpu_runtime_lib_dir);

  // passage_embeddings::mojom::PassageEmbedder:
  void GenerateEmbeddings(const std::vector<std::string>& passages,
                          passage_embeddings::mojom::PassagePriority priority,
                          GenerateEmbeddingsCallback callback) override;

 private:
  BraveLitertPassageEmbedder(
      std::vector<uint8_t> tflite_model,
      litert::Environment environment,
      litert::CompiledModel model,
      std::unique_ptr<sentencepiece::SentencePieceProcessor> tokenizer,
      size_t input_window_size);

  // Tokenizes `text` into an `input_window_size_`-length token id sequence
  // ([bos] + tokens + [eos] + pad), returning false if tokenization fails.
  bool Tokenize(const std::string& text, std::vector<int32_t>* out) const;

  // Runs the model for a single already-tokenized passage, appending the
  // Float32 output to `*embedding`. Returns false on failure.
  bool Embed(const std::vector<int32_t>& tokens, std::vector<float>* embedding);

  // The runtime references the model bytes past compilation, so they are kept
  // alive here.
  std::vector<uint8_t> tflite_model_;
  litert::Environment environment_;
  litert::CompiledModel model_;
  std::unique_ptr<sentencepiece::SentencePieceProcessor> tokenizer_;
  size_t input_window_size_ = 0;
};

}  // namespace brave_history_embeddings

#endif  // BRAVE_BROWSER_HISTORY_EMBEDDINGS_BRAVE_LITERT_PASSAGE_EMBEDDER_H_
