/* Copyright (c) 2026 The Brave Authors. All rights reserved.
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/. */

import {
  ColorScheme,
  WelcomePageHandler,
  WelcomePageInterface,
  WelcomePageReceiver,
  WelcomePageRemote,
  WelcomePageHandlerInterface,
} from 'gen/brave/browser/ui/webui/brave_welcome_page/brave_welcome_page.mojom.m.js'

import {
  Theme,
  ChromeColor,
  ThemeColorPickerHandler,
  ThemeColorPickerHandlerInterface,
  ThemeColorPickerClientInterface,
} from 'chrome://resources/cr_components/theme_color_picker/theme_color_picker.mojom-webui.js'

import { addWebUiListener, sendWithPromise } from 'chrome://resources/js/cr.js'
import { ThemeColorPickerProxy } from './theme_color_picker_proxy'
import {
  createInterfaceApi,
  endpointsFor,
  eventsFor,
  state,
} from '$web-common/api'

export { ColorScheme, Theme, ChromeColor }

export type SkColor = ChromeColor['seed']

export enum P3APhase {
  Welcome = 0,
  Import = 1,
  Consent = 2,
  Finished = 3,
}

export interface DefaultBrowserInfo {
  canBeDefault: boolean
  isDefault: boolean
  isDisabledByPolicy: boolean
  isUnknownError: boolean
}

export const importDataTypes = [
  'autofillFormData',
  'extensions',
  'favorites',
  'history',
  'passwords',
  'search',
] as const

export type ImportDataType = (typeof importDataTypes)[number]

export interface BrowserProfile {
  index: number
  name: string
  profileName: string
  autofillFormData: boolean
  extensions: boolean
  favorites: boolean
  history: boolean
  passwords: boolean
  search: boolean
}

export type ImportDataStatus = '' | 'inProgress' | 'succeeded' | 'failed'

interface EventSource<T> {
  addListeners: (listeners: Partial<T>) => () => void
}

function eventSourceFromRouter<T>(router: any): EventSource<T> {
  return {
    addListeners(listeners) {
      const ids: number[] = []
      for (const [key, val] of Object.entries(listeners)) {
        ids.push(router[key].addListener(val))
      }
      return () => {
        for (const id of ids) {
          router.removeListener(id)
        }
      }
    },
  }
}

export type ThemeColorPickerEventSource =
  EventSource<ThemeColorPickerClientInterface>

interface ApiInit {
  welcomePageHandler: WelcomePageHandlerInterface
  getWelcomePageRemote?: (page: WelcomePageInterface) => WelcomePageRemote
  themeColorPickerHandler: ThemeColorPickerHandlerInterface
  themeColorPickerEventSource: ThemeColorPickerEventSource
  messages: {
    getDefaultBrowserInfo: () => Promise<DefaultBrowserInfo>
    getBrowserProfilesForImport: () => Promise<BrowserProfile[]>
    setAsDefaultBrowser: () => void
    importData: (profileIndex: number, types: Set<ImportDataType>) => void
  }
}

function defaultInit(): ApiInit {
  const themeColorPickerProxy = ThemeColorPickerProxy.getInstance()
  return {
    welcomePageHandler: WelcomePageHandler.getRemote(),
    getWelcomePageRemote: (page) => {
      return new WelcomePageReceiver(page).$.bindNewPipeAndPassRemote()
    },
    themeColorPickerHandler: ThemeColorPickerHandler.getRemote(),
    themeColorPickerEventSource: eventSourceFromRouter(
      themeColorPickerProxy.callbackRouter,
    ),
    messages: {
      getDefaultBrowserInfo() {
        return sendWithPromise('requestDefaultBrowserState')
      },
      getBrowserProfilesForImport() {
        return sendWithPromise('initializeImportDialog')
      },
      setAsDefaultBrowser() {
        chrome.send('setAsDefaultBrowser')
      },
      importData(profileIndex, types) {
        chrome.send('importData', [profileIndex, importTypesToDict(types)])
      },
    },
  }
}

export function createWelcomeApi(init = defaultInit()) {
  const {
    welcomePageHandler,
    themeColorPickerHandler,
    themeColorPickerEventSource,
  } = init

  const api = createInterfaceApi({
    endpoints: {
      ...endpointsFor(themeColorPickerHandler, {
        getChromeColors: {
          response: (r) => r.colors,
          placeholderData: [] as ChromeColor[],
        },
      }),
      theme: state<Theme | undefined>(),
      ...endpointsFor(welcomePageHandler, {
        getColorScheme: {
          response: (r) => r.colorScheme,
          prefetchWithArgs: [],
          placeholderData: ColorScheme.kSystem,
        },
        setColorScheme: {
          mutationResponse: () => {},
          onMutate: ([colorScheme]: [ColorScheme]) => {
            api.getColorScheme.update(colorScheme)
          },
        },
        getVerticalTabsEnabled: {
          response: (r) => r.enabled,
          prefetchWithArgs: [],
          placeholderData: false,
        },
        setVerticalTabsEnabled: {
          mutationResponse: () => {},
          onMutate: ([enabled]: [boolean]) => {
            api.getVerticalTabsEnabled.update(enabled)
          },
        },
        getWebDiscoveryEnabled: {
          response: (r) => r.enabled,
          prefetchWithArgs: [],
          placeholderData: false,
        },
        setWebDiscoveryEnabled: {
          mutationResponse: () => {},
          onMutate: ([enabled]: [boolean]) => {
            api.getWebDiscoveryEnabled.update(enabled)
          },
        },
        getP3AEnabled: {
          response: (r) => r.enabled,
          prefetchWithArgs: [],
          placeholderData: false,
        },
        setP3AEnabled: {
          mutationResponse: () => {},
          onMutate: ([enabled]: [boolean]) => {
            api.getP3AEnabled.update(enabled)
          },
        },
        getCrashReportsEnabled: {
          response: (r) => r.enabled,
          prefetchWithArgs: [],
          placeholderData: false,
        },
        setCrashReportsEnabled: {
          mutationResponse: () => {},
          onMutate: ([enabled]: [boolean]) => {
            api.getCrashReportsEnabled.update(enabled)
          },
        },
      }),
      getDefaultBrowserState: {
        query: () => init.messages.getDefaultBrowserInfo(),
      },
      getBrowserProfilesForImport: {
        query: () => init.messages.getBrowserProfilesForImport(),
      },
      importDataStatus: state<ImportDataStatus>(''),
    },

    events: {
      ...eventsFor(
        WelcomePageInterface,
        {
          onColorSchemeChanged() {
            api.getColorScheme.invalidate()
          },
          onVerticalTabsEnabledChanged() {
            api.getVerticalTabsEnabled.invalidate()
          },
          onWebDiscoveryEnabledChanged() {
            api.getWebDiscoveryEnabled.invalidate()
          },
          onP3AEnabledChanged() {
            api.getP3AEnabled.invalidate()
          },
          onCrashReportsEnabledChanged() {
            api.getCrashReportsEnabled.invalidate()
          },
        },
        (observer) => {
          if (init.getWelcomePageRemote) {
            welcomePageHandler.setWelcomePage(
              init.getWelcomePageRemote(observer),
            )
          }
        },
      ),
    },

    actions: {
      setAsDefaultBrowser: init.messages.setAsDefaultBrowser,
      importData: init.messages.importData,
      setThemeColor: (
        seed: ChromeColor['seed'],
        variant: ChromeColor['variant'],
      ) => {
        themeColorPickerHandler.setSeedColor(seed, variant)
      },
      setGreyThemeColor: () => {
        themeColorPickerHandler.setGreyDefaultColor()
      },
    },
  })

  themeColorPickerEventSource.addListeners({
    setTheme(theme) {
      api.theme.update(theme)
    },
  })

  themeColorPickerHandler.updateTheme()

  addWebUiListener('import-data-status-changed', (status: unknown) => {
    switch (status) {
      case 'inProgress':
      case 'failed':
      case 'succeeded':
        api.importDataStatus.update(status)
        break
      default:
        api.importDataStatus.update('')
        break
    }
  })

  return api
}

export type WelcomeApi = ReturnType<typeof createWelcomeApi>

function importTypesToDict(types: Set<ImportDataType>) {
  let dict: any = {}
  for (const type of types) {
    dict[importTypeToDialogKey(type)] = true
  }
  return dict
}

function importTypeToDialogKey(type: ImportDataType) {
  switch (type) {
    case 'autofillFormData':
      return 'import_dialog_autofill_form_data'
    case 'extensions':
      return 'import_dialog_extensions'
    case 'favorites':
      return 'import_dialog_bookmarks'
    case 'history':
      return 'import_dialog_history'
    case 'passwords':
      return 'import_dialog_saved_passwords'
    case 'search':
      return 'import_dialog_search_engine'
  }
}
