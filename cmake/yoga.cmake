# Copyright (C) 2026 Rodrigo Jose Hernandez Cordoba
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

# Locate or fetch Facebook Yoga (flexbox layout engine) used by the optional
# HTML support. Sets up a single import target named ``yogacore`` that callers
# can link against regardless of how Yoga was discovered.
#
#   1. Try ``find_package(yoga CONFIG)`` (vcpkg, system installs that ship a
#      CMake config).
#   2. Try ``pkg_check_modules(YOGA yoga)`` (Homebrew, distros that ship a .pc).
#   3. Fall back to ``FetchContent`` from the upstream Git repository, mirroring
#      how googletest is handled in ``tests/CMakeLists.txt``.

if(TARGET yogacore)
    return()
endif()

set(_aeongui_yoga_found FALSE)

# 1. CMake config (vcpkg ships ``yogaConfig.cmake``).
find_package(yoga CONFIG QUIET)
if(yoga_FOUND)
    if(TARGET yoga::yogacore)
        add_library(yogacore INTERFACE IMPORTED)
        target_link_libraries(yogacore INTERFACE yoga::yogacore)
        set(_aeongui_yoga_found TRUE)
        message(STATUS "Yoga: using CMake package (target yoga::yogacore)")
    elseif(TARGET yoga::yoga)
        add_library(yogacore INTERFACE IMPORTED)
        target_link_libraries(yogacore INTERFACE yoga::yoga)
        set(_aeongui_yoga_found TRUE)
        message(STATUS "Yoga: using CMake package (target yoga::yoga)")
    endif()
endif()

# 2. pkg-config.
if(NOT _aeongui_yoga_found AND PKG_CONFIG_FOUND)
    pkg_check_modules(YOGA QUIET yoga)
    if(YOGA_FOUND)
        add_library(yogacore INTERFACE IMPORTED)
        target_include_directories(yogacore INTERFACE ${YOGA_INCLUDE_DIRS})
        target_link_libraries(yogacore INTERFACE ${YOGA_LIBRARIES})
        if(YOGA_LIBRARY_DIRS)
            target_link_directories(yogacore INTERFACE ${YOGA_LIBRARY_DIRS})
        endif()
        set(_aeongui_yoga_found TRUE)
        message(STATUS "Yoga: using pkg-config (${YOGA_VERSION})")
    endif()
endif()

# 3. FetchContent fallback.
if(NOT _aeongui_yoga_found)
    message(STATUS "Yoga: not found via find_package or pkg-config, fetching from upstream")
    include(FetchContent)
    set(YOGA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        yoga
        GIT_REPOSITORY https://github.com/facebook/yoga.git
        GIT_TAG        v3.2.1
        GIT_SHALLOW    TRUE
        # Yoga's top-level CMakeLists unconditionally
        # ``add_subdirectory(tests)`` and the test sources don't compile
        # cleanly under newer GCC (-Werror=array-bounds in
        # YGPersistenceTest.cpp on GCC 15.2).  Point SOURCE_SUBDIR at the
        # library-only subdirectory so we never see the test targets.
        SOURCE_SUBDIR  yoga
    )
    FetchContent_MakeAvailable(yoga)
    if(TARGET yogacore)
        # Upstream defines target named ``yogacore`` directly; nothing else needed.
        message(STATUS "Yoga: using FetchContent (target yogacore)")
    elseif(TARGET yoga)
        add_library(yogacore INTERFACE IMPORTED)
        target_link_libraries(yogacore INTERFACE yoga)
        message(STATUS "Yoga: using FetchContent (target yoga)")
    else()
        message(FATAL_ERROR "Yoga FetchContent did not produce expected target")
    endif()
endif()
