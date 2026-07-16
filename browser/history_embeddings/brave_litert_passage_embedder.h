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
#include "base/functional/callback.h"
#include "base/memory/scoped_refptr.h"
#include "base/task/sequenced_task_runner.h"
#include "mojo/public/cpp/bindings/pending_receiver.h"
#include "mojo/public/cpp/bindings/receiver.h"
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
//
// Because loading (reading a large .tflite) and inference are expensive, the
// service constructs this via base::SequenceBound on a background sequence: the
// file-loading constructor reads the model files, compiles, and binds the mojo
// receiver on that sequence, so GenerateEmbeddings never runs on the UI thread.
class BraveLitertPassageEmbedder
    : public passage_embeddings::mojom::PassageEmbedder {
 public:
  ~BraveLitertPassageEmbedder() override;

  BraveLitertPassageEmbedder(const BraveLitertPassageEmbedder&) = delete;
  BraveLitertPassageEmbedder& operator=(const BraveLitertPassageEmbedder&) =
      delete;

  // Builds an unbound embedder directly from in-memory model bytes. Used by
  // tests; returns nullptr on failure. See the file-loading constructor below
  // for the service path.
  static std::unique_ptr<BraveLitertPassageEmbedder> Create(
      base::span<const uint8_t> tflite_model,
      base::span<const uint8_t> sentencepiece_model,
      bool use_gpu,
      const base::FilePath& gpu_runtime_lib_dir);

  // Service path, intended to run on a background sequence (via
  // base::SequenceBound). Reads the `.tflite` and SentencePiece files from
  // disk, compiles for GPU (`use_gpu` + `gpu_runtime_lib_dir` = the prebuilt
  // accelerator directory) or CPU, and binds `receiver`. `load_callback` (with
  // the success bool) and `on_disconnect` are posted to `reply_task_runner`
  // (the caller's sequence).
  BraveLitertPassageEmbedder(
      const base::FilePath& tflite_model_path,
      const base::FilePath& sentencepiece_model_path,
      bool use_gpu,
      const base::FilePath& gpu_runtime_lib_dir,
      mojo::PendingReceiver<passage_embeddings::mojom::PassageEmbedder>
          receiver,
      scoped_refptr<base::SequencedTaskRunner> reply_task_runner,
      base::OnceCallback<void(bool)> load_callback,
      base::OnceClosure on_disconnect);

  // passage_embeddings::mojom::PassageEmbedder:
  void GenerateEmbeddings(const std::vector<std::string>& passages,
                          passage_embeddings::mojom::PassagePriority priority,
                          GenerateEmbeddingsCallback callback) override;

 private:
  BraveLitertPassageEmbedder();

  // Compiles `tflite_model` and loads `sentencepiece_model` into this instance.
  // Returns false on failure.
  bool Init(base::span<const uint8_t> tflite_model,
            base::span<const uint8_t> sentencepiece_model,
            bool use_gpu,
            const base::FilePath& gpu_runtime_lib_dir);

  // Tokenizes `text` into an `input_window_size_`-length token id sequence
  // ([bos] + tokens + [eos] + pad), returning false if tokenization fails.
  bool Tokenize(const std::string& text, std::vector<int32_t>* out) const;

  // Runs the model for a single already-tokenized passage, appending the
  // Float32 output to `*embedding`. Returns false on failure.
  bool Embed(const std::vector<int32_t>& tokens, std::vector<float>* embedding);

  // The runtime references the model bytes past compilation, so they are kept
  // alive here.
  std::vector<uint8_t> tflite_model_;
  std::optional<litert::Environment> environment_;
  std::optional<litert::CompiledModel> model_;
  std::unique_ptr<sentencepiece::SentencePieceProcessor> tokenizer_;
  size_t input_window_size_ = 0;

  // Bound only on the service path.
  mojo::Receiver<passage_embeddings::mojom::PassageEmbedder> receiver_{this};
};

}  // namespace brave_history_embeddings

#endif  // BRAVE_BROWSER_HISTORY_EMBEDDINGS_BRAVE_LITERT_PASSAGE_EMBEDDER_H_
