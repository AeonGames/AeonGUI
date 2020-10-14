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
    endif()
endif()
if(NOT V8_FOUND)
    find_package(V8 REQUIRED)
    set(V8_TARGET V8::V8)
    if(TARGET V8::V8LIBBASE)
        set(V8_TARGET "${V8_TARGET} V8::V8LIBBASE")
    endif()
    if(TARGET V8::V8LIBPLATFORM)
        set(V8_TARGET "${V8_TARGET} V8::V8LIBPLATFORM")
    endif()
endif()
