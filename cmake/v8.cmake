# Copyright (C) 2020 Rodrigo Jose Hernandez Cordoba
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

find_package(PkgConfig)
if(PKG_CONFIG_FOUND)
    pkg_check_modules(V8 QUIET IMPORTED_TARGET GLOBAL v8 v8_libplatform)
    if(V8_FOUND)
        set(V8_TARGET PkgConfig::V8)
    else()
        find_path(V8_INCLUDE_DIR NAMES "v8.h" PATH_SUFFIXES "nodejs/deps/v8/include" REQUIRED)

        if(V8_INCLUDE_DIR AND EXISTS "${V8_INCLUDE_DIR}/v8-version.h")
            file(STRINGS "${V8_INCLUDE_DIR}/v8-version.h" V8_MAJOR_VERSION REGEX "^#define V8_MAJOR_VERSION [0-9]+.*$")
            string(REGEX REPLACE "^#define V8_MAJOR_VERSION ([0-9]+).*$" "\\1" V8_MAJOR_VERSION "${V8_MAJOR_VERSION}")
            file(STRINGS "${V8_INCLUDE_DIR}/v8-version.h" V8_MINOR_VERSION REGEX "^#define V8_MINOR_VERSION [0-9]+.*$")
            string(REGEX REPLACE "^#define V8_MINOR_VERSION ([0-9]+).*$" "\\1" V8_MINOR_VERSION  "${V8_MINOR_VERSION}")
            file(STRINGS "${V8_INCLUDE_DIR}/v8-version.h" V8_BUILD_NUMBER REGEX "^#define V8_BUILD_NUMBER [0-9]+.*$")
            string(REGEX REPLACE "^#define V8_BUILD_NUMBER ([0-9]+).*$" "\\1" V8_BUILD_NUMBER "${V8_BUILD_NUMBER}")
            file(STRINGS "${V8_INCLUDE_DIR}/v8-version.h" V8_PATCH_LEVEL REGEX "^#define V8_PATCH_LEVEL [0-9]+.*$")
            string(REGEX REPLACE "^#define V8_PATCH_LEVEL ([0-9]+).*$" "\\1" V8_PATCH_LEVEL "${V8_PATCH_LEVEL}")
            set(V8_VERSION_STRING "${V8_MAJOR_VERSION}.${V8_MINOR_VERSION}.${V8_BUILD_NUMBER}.${V8_PATCH_LEVEL}")
            message(STATUS "V8 Version ${V8_VERSION_STRING}")
        endif()

        find_library(v8 V8_LIBRARY)
        find_library(v8_libbase V8LIBBASE_LIBRARY)
        find_library(v8_libplatform V8LIBPLATFORM_LIBRARY)
        add_library(V8::V8 SHARED IMPORTED)
        set_target_properties(V8::V8 PROPERTIES
          INTERFACE_INCLUDE_DIRECTORIES "${V8_INCLUDE_DIR}"
          IMPORTED_LOCATION "${V8_LIBRARY}"
          IMPORTED_LINK_INTERFACE_LIBRARIES
          "${V8LIBBASE_LIBRARY};${V8LIBPLATFORM_LIBRARY}")
        set(V8_TARGET V8::V8)
    endif()
else()
    find_package(V8 REQUIRED)
    set(V8_TARGET V8::V8 V8::V8LIBBASE V8::V8LIBPLATFORM)
endif()
