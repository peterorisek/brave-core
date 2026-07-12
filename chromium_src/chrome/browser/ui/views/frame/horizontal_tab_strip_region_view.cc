/* Copyright (c) 2020 The Brave Authors. All rights reserved.
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/. */

#include "chrome/browser/ui/views/frame/horizontal_tab_strip_region_view.h"

#include "brave/browser/ui/views/frame/brave_tab_strip_region_view.h"
#include "brave/browser/ui/views/tabs/brave_browser_tab_strip_controller.h"
#include "brave/browser/ui/views/tabs/brave_new_tab_button.h"
#include "brave/browser/ui/views/tabs/brave_tab_hover_card_controller.h"
#include "brave/browser/ui/views/tabs/brave_tab_strip.h"

#define CreateHorizontalTabStripRegionView \
  CreateHorizontalTabStripRegionView_ChromiumImpl

#include <chrome/browser/ui/views/frame/horizontal_tab_strip_region_view.cc>

#undef CreateHorizontalTabStripRegionView

// We don't support the unified tab strip yet, but otherwise we do want to
// return our subclass
std::unique_ptr<TabStripRegionView> CreateHorizontalTabStripRegionView(
    BrowserView* browser_view) {
  if (base::FeatureList::IsEnabled(tabs::kTabStripUnification)) {
    return CreateHorizontalTabStripRegionView_ChromiumImpl(browser_view);
  }
  return std::make_unique<BraveHorizontalTabStripRegionView>(browser_view);
}
