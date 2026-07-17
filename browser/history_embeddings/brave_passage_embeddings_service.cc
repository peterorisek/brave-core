// Copyright (c) 2026 The Brave Authors. All rights reserved.
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#include "brave/browser/history_embeddings/brave_passage_embeddings_service.h"

#include <utility>

#include "base/check.h"
#include "base/command_line.h"
#include "base/feature_list.h"
#include "base/files/file_path.h"
#include "base/functional/bind.h"
#include "base/logging.h"
#include "base/task/sequenced_task_runner.h"
#include "base/task/thread_pool.h"
#include "brave/services/passage_embeddings/brave_litert_passage_embedder.h"
#include "brave/browser/history_embeddings/features.h"
#include "components/history_embeddings/core/history_embeddings_features.h"

namespace passage_embeddings {

namespace {

// Optional override for the model location: the path to the EmbeddingGemma
// `.tflite`. `sentencepiece.model` is read as a sibling in the same directory.
// Only consulted when kBraveHistoryEmbeddingsLitertGpu is enabled.
constexpr char kLitertModelSwitch[] = "history-embeddings-litert-model";

constexpr char kSentencePieceModelName[] = "sentencepiece.model";

// PoC dev defaults so launching only needs the feature flag (no switches).
// These are local absolute paths and will be replaced by component delivery.
// TODO(litert-poc): remove once the model + accelerator are shipped.
constexpr char kDefaultModelPath[] =
    "/Users/darkdh/Projects/embeddinggemma-300m/"
    "embeddinggemma-300M_seq256_mixed-precision.tflite";
constexpr char kDefaultAcceleratorDir[] =
    "/Users/darkdh/Projects/brave-browser/src/third_party/litert/src/litert/"
    "prebuilt/macos_arm64";

}  // namespace

// static
bool BravePassageEmbeddingsService::ShouldUseLitertEmbedder() {
  return base::FeatureList::IsEnabled(kBraveHistoryEmbeddingsLitertGpu);
}

BravePassageEmbeddingsService::BravePassageEmbeddingsService(
    BackgroundWebContentsFactory background_web_contents_factory)
    : background_web_contents_factory_(
          std::move(background_web_contents_factory)) {
  CHECK(base::FeatureList::IsEnabled(history_embeddings::kHistoryEmbeddings));
}

BravePassageEmbeddingsService::~BravePassageEmbeddingsService() = default;

void BravePassageEmbeddingsService::BindLocalAIReceiver(
    mojo::PendingReceiver<local_ai::mojom::LocalAIService> receiver) {
  if (batch_embedder_) {
    batch_embedder_->BindLocalAIReceiver(std::move(receiver));
  }
}

void BravePassageEmbeddingsService::BindPassageEmbedder(
    mojo::PendingReceiver<mojom::PassageEmbedder> receiver,
    local_ai::mojom::ModelFilesPtr model_files,
    base::OnceCallback<void(bool)> callback) {
  CHECK(model_files);

  // When enabled, run EmbeddingGemma natively via CompiledModel (GPU/CPU)
  // instead of the WASM worker. `model_files` (the WASM-format buffers) is
  // ignored here -- the .tflite comes from the switch (or the PoC default), the
  // tokenizer is read as a sibling, and the prebuilt GPU accelerator from the
  // hardcoded PoC directory, until component delivery lands.
  if (ShouldUseLitertEmbedder()) {
    if (litert_embedder_) {
      DVLOG(1) << "BindPassageEmbedder called while a LiteRT embedder is "
                  "already bound; failing";
      std::move(callback).Run(false);
      return;
    }
    base::FilePath model_path =
        base::CommandLine::ForCurrentProcess()->GetSwitchValuePath(
            kLitertModelSwitch);
    if (model_path.empty()) {
      model_path = base::FilePath::FromASCII(kDefaultModelPath);
    }
    // Load + compile + run on a background sequence so the large model read and
    // inference never touch the UI thread. The embedder uses the GPU only if
    // the prebuilt accelerator is present in the accelerator directory.
    litert_embedder_ = base::SequenceBound<
        brave_history_embeddings::BraveLitertPassageEmbedder>(
        base::ThreadPool::CreateSequencedTaskRunner(
            {base::MayBlock(), base::TaskPriority::USER_VISIBLE}),
        model_path, model_path.DirName().Append(kSentencePieceModelName),
        /*use_gpu=*/true, base::FilePath::FromASCII(kDefaultAcceleratorDir),
        std::move(receiver), base::SequencedTaskRunner::GetCurrentDefault(),
        std::move(callback),
        base::BindOnce(
            &BravePassageEmbeddingsService::OnLitertEmbedderDisconnected,
            weak_ptr_factory_.GetWeakPtr()));
    return;
  }

  // Upstream's controller binds one embedder at a time — the base class
  // gates on `!embedder_remote_`. If we see a second Bind while a load
  // is in flight or an embedder is already active, fail the extra
  // request defensively.
  if (batch_embedder_) {
    DVLOG(1) << "BindPassageEmbedder called while a BatchEmbedder is already "
                "bound; failing";
    std::move(callback).Run(false);
    return;
  }

  batch_embedder_ = std::make_unique<BraveBatchPassageEmbedder>(
      std::move(receiver), background_web_contents_factory_,
      std::move(model_files), std::move(callback),
      base::BindOnce(
          &BravePassageEmbeddingsService::OnBatchEmbedderDisconnected,
          weak_ptr_factory_.GetWeakPtr()));
}

void BravePassageEmbeddingsService::LoadModels(
    mojom::PassageEmbeddingsLoadModelsParamsPtr model_params,
    mojom::PassageEmbedderParamsPtr params,
    mojo::PendingReceiver<mojom::PassageEmbedder> model,
    LoadModelsCallback callback) {
  // The upstream mojom is shaped for tflite + sentencepiece; Brave's
  // model files are not carried in this struct. The controller calls
  // BindPassageEmbedder directly and never binds service_remote_, so
  // this override is unreachable in practice. Fail defensively.
  DVLOG(1) << "LoadModels called on BravePassageEmbeddingsService; "
              "this path is unused. Failing.";
  std::move(callback).Run(false);
}

void BravePassageEmbeddingsService::OnBatchEmbedderDisconnected() {
  DVLOG(3) << "BraveBatchPassageEmbedder disconnected; tearing down";
  batch_embedder_.reset();
}

void BravePassageEmbeddingsService::OnLitertEmbedderDisconnected() {
  DVLOG(3) << "BraveLitertPassageEmbedder disconnected; tearing down";
  litert_embedder_.Reset();
}

}  // namespace passage_embeddings
