/*
Copyright (C) 2026 Rodrigo Jose Hernandez Cordoba

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef AEONGUI_RESOURCELOADER_H
#define AEONGUI_RESOURCELOADER_H

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>
#include "aeongui/Platform.hpp"

namespace AeonGUI
{
    /** @brief Callback used to retrieve a resource (image, document, etc.)
     *         from an embedder-defined backing store.
     *
     *  The callback is given the path or URL the library is trying to open
     *  and is expected to fill @p aBytes with the resource contents and
     *  return @c true on success. Returning @c false means the loader does
     *  not handle this resource and the library should fall back to its
     *  default filesystem-based loading.
     */
    using ResourceLoader = std::function<bool ( const std::string& aPathOrUrl,
                           std::vector<uint8_t>& aBytes ) >;

    /** @brief Install a global resource loader.
     *
     *  Pass an empty std::function to remove the current loader.
     *  Calls to SetResourceLoader replace any previously installed loader.
     */
    AEONGUI_DLL void SetResourceLoader ( ResourceLoader aLoader );

    /** @brief Get the currently installed resource loader.
     *  @return A reference to the loader. The function may be empty.
     */
    AEONGUI_DLL const ResourceLoader& GetResourceLoader();

    /** @brief Convenience helper that invokes the installed loader.
     *
     *  @param aPathOrUrl The path or URL to fetch.
     *  @param aBytes Output buffer that will be filled with the data on success.
     *  @return true if a loader is installed and it returned true. false
     *          indicates either no loader is installed or the loader did not
     *          handle this resource.
     */
    AEONGUI_DLL bool TryLoadResource ( const std::string& aPathOrUrl,
                                       std::vector<uint8_t>& aBytes );
}

#endif
