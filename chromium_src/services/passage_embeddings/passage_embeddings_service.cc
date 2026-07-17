/* Copyright (c) 2026 The Brave Authors. All rights reserved.
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/. */

#include "brave/components/local_ai/buildflags/buildflags.h"

#if !BUILDFLAG(ENABLE_LOCAL_AI)

#include <services/passage_embeddings/passage_embeddings_service.cc>

#else  // BUILDFLAG(ENABLE_LOCAL_AI)

// Brave runs EmbeddingGemma through LiteRT's CompiledModel (Metal GPU or CPU)
// rather than upstream's tflite PassageEmbedder. This replaces the service's
// LoadModels body while keeping the upstream class, mojom, and sandboxed
// utility process (launched via ServiceProcessHost) intact.

#include <memory>
#include <utility>

#include "base/functional/callback_helpers.h"
#include "base/task/sequenced_task_runner.h"
#include "base/task/thread_pool.h"
#include "base/threading/sequence_bound.h"
#include "brave/services/passage_embeddings/brave_litert_passage_embedder.h"
#include "build/build_config.h"
#include "services/passage_embeddings/passage_embedder.h"
#include "services/passage_embeddings/passage_embeddings_service.h"

#if BUILDFLAG(IS_MAC)
#include "base/apple/bundle_locations.h"
#endif

namespace passage_embeddings {

namespace {

// The prebuilt LiteRT GPU accelerator is bundled next to the framework (see
// //brave/third_party/litert:litert_metal_accelerator_bundle_data). The
// embedder dlopens it from here; empty elsewhere means CPU.
base::FilePath GetAcceleratorLibraryDir() {
#if BUILDFLAG(IS_MAC)
  return base::apple::FrameworkBundlePath().Append("Libraries");
#else
  return base::FilePath();
#endif
}

}  // namespace

PassageEmbeddingsService::PassageEmbeddingsService(
    mojo::PendingReceiver<mojom::PassageEmbeddingsService> receiver)
    : receiver_(this, std::move(receiver)),
      task_runner_(base::ThreadPool::CreateUpdateableSequencedTaskRunner(
          {base::MayBlock(), base::TaskPriority::BEST_EFFORT,
           base::ThreadPolicy::MUST_USE_FOREGROUND})) {}

PassageEmbeddingsService::~PassageEmbeddingsService() = default;

void PassageEmbeddingsService::OnEmbedderDisconnect() {
  embedder_.reset();
}

void PassageEmbeddingsService::LoadModels(
    mojom::PassageEmbeddingsLoadModelsParamsPtr model_params,
    mojom::PassageEmbedderParamsPtr embedder_params,
    mojo::PendingReceiver<mojom::PassageEmbedder> receiver,
    LoadModelsCallback callback) {
  auto runner = base::ThreadPool::CreateSequencedTaskRunner(
      {base::MayBlock(), base::TaskPriority::USER_VISIBLE});

  // The embedder owns its own mojo receiver and tears itself down on
  // disconnect, so no member is needed to hold it: the SequenceBound lives
  // inside the disconnect closure. When the pipe drops, the closure runs on
  // this sequence, destroys the SequenceBound, and the embedder is deleted on
  // its own sequence.
  auto owner = std::make_unique<base::SequenceBound<
      brave_history_embeddings::BraveLitertPassageEmbedder>>();
  auto* bound = owner.get();
  *bound =
      base::SequenceBound<brave_history_embeddings::BraveLitertPassageEmbedder>(
          std::move(runner), std::move(model_params->embeddings_model),
          std::move(model_params->sp_model), /*use_gpu=*/true,
          GetAcceleratorLibraryDir(), std::move(receiver),
          base::SequencedTaskRunner::GetCurrentDefault(), std::move(callback),
          base::DoNothingWithBoundArgs(std::move(owner)));
}

void PassageEmbeddingsService::OnModelsLoaded(LoadModelsCallback callback,
                                              bool success) {
  std::move(callback).Run(success);
}

}  // namespace passage_embeddings

#endif  // BUILDFLAG(ENABLE_LOCAL_AI)
