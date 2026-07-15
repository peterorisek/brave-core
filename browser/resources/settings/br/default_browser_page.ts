/* Copyright (c) 2020 The Brave Authors. All rights reserved.
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/. */

import type { PropertyValues } from '//resources/lit/v3_0/lit.rollup.js'

import {
  SettingsDefaultBrowserPageElement
} from '../default_browser_page/default_browser_page.js'

const modifyDefaultBrowserPage = (root: ShadowRoot) => {
  // Stop the row thinking it's the first, since this item is added to
  // Brave's "Get Started" section.
  for (const row of root.querySelectorAll('.cr-row.first')) {
    row.classList.remove('first')
  }

  // Replace settings-section with its children, since we want its
  // controls in our page but not the header that it also now includes.
  const settingsSection = root.querySelector('settings-section')
  if (!settingsSection) {
    throw new Error(
      '[Settings] Missing settings-section on default_browser_page')
  }
  settingsSection.replaceWith(...settingsSection.childNodes)
}

// `firstUpdated` is `protected` on ReactiveElement, so reach it through an
// untyped view of the prototype to patch it from outside the class hierarchy.
const proto = SettingsDefaultBrowserPageElement.prototype as unknown as {
  firstUpdated?: (changedProperties: PropertyValues) => void
}
const originalFirstUpdated = proto.firstUpdated
proto.firstUpdated = function(
    this: SettingsDefaultBrowserPageElement,
    changedProperties: PropertyValues) {
  originalFirstUpdated?.call(this, changedProperties)
  if (this.shadowRoot) {
    modifyDefaultBrowserPage(this.shadowRoot)
  }
}
