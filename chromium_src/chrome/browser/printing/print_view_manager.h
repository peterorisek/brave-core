/* Copyright (c) 2026 The Brave Authors. All rights reserved.
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/. */

#ifndef BRAVE_CHROMIUM_SRC_CHROME_BROWSER_PRINTING_PRINT_VIEW_MANAGER_H_
#define BRAVE_CHROMIUM_SRC_CHROME_BROWSER_PRINTING_PRINT_VIEW_MANAGER_H_

// `GetPrintPreviewParams()` now requires `print_preview_rfh_` to match the
// requesting frame. Our screenshot print preview extractor drives print preview
// directly, without going through the PrintPreviewNow() / PrintPreviewDone()
// dialog flow that normally sets `print_preview_rfh_`, so expose a narrow pair
// of accessors it can use to satisfy that requirement.
#define SetPrintPreviewRenderFrameHost                  \
  SetPrintPreviewRenderFrameHost_Unused();              \
                                                        \
 public:                                                \
  void SetPrintPreviewRenderFrameHostForExtraction(     \
      content::RenderFrameHost* rfh);                   \
  void ClearPrintPreviewRenderFrameHostForExtraction(); \
                                                        \
 private:                                               \
  void SetPrintPreviewRenderFrameHost

#include <chrome/browser/printing/print_view_manager.h>  // IWYU pragma: export

#undef SetPrintPreviewRenderFrameHost

#endif  // BRAVE_CHROMIUM_SRC_CHROME_BROWSER_PRINTING_PRINT_VIEW_MANAGER_H_
