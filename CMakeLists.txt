cmake_minimum_required(VERSION 3.12.0)
project(AeonGUI VERSION 0.9.0 LANGUAGES NONE)

function(add_docs_target)
  find_package(Doxygen)
  if(DOXYGEN_FOUND)
    set(DOXYGEN_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/docs")
    configure_file(${PROJECT_SOURCE_DIR}/cmake/Doxyfile.in ${CMAKE_BINARY_DIR}/Doxyfile @ONLY)
    add_custom_target(docs
      COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_BINARY_DIR}/Doxyfile
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
      COMMENT "Generating API documentation with Doxygen"
      VERBATIM
    )
  endif()
endfunction()

option(DOCS_ONLY "Configure only the docs target" OFF)
if(DOCS_ONLY)
  message(STATUS "DOCS_ONLY is ON: skipping full project dependencies and targets")
  add_docs_target()
  return()
endif()

find_package(Git QUIET)
if(GIT_FOUND)
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    OUTPUT_VARIABLE GIT_COMMIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
endif()
if(NOT GIT_COMMIT_HASH)
  set(GIT_COMMIT_HASH "0000000000000000000000000000000000000000")
endif()

option(BUILD_SCRIPTS_ONLY "Generate only packaging scripts (PKGBUILDs, brew formula, vcpkg port)" OFF)
if(BUILD_SCRIPTS_ONLY)
  message(STATUS "BUILD_SCRIPTS_ONLY is ON: generating packaging templates only")
  configure_file("${PROJECT_SOURCE_DIR}/cmake/PKGBUILD-arch.in"
                 "${CMAKE_BINARY_DIR}/PKGBUILD-arch" @ONLY NEWLINE_STYLE UNIX)
  configure_file("${PROJECT_SOURCE_DIR}/cmake/PKGBUILD-msys2.in"
                 "${CMAKE_BINARY_DIR}/PKGBUILD-msys2" @ONLY NEWLINE_STYLE UNIX)
  configure_file("${PROJECT_SOURCE_DIR}/cmake/vcpkg-port-vcpkg.json.in"
                 "${CMAKE_BINARY_DIR}/vcpkg-port/vcpkg.json" @ONLY)
  configure_file("${PROJECT_SOURCE_DIR}/cmake/vcpkg-port-portfile.cmake.in"
                 "${CMAKE_BINARY_DIR}/vcpkg-port/portfile.cmake" @ONLY)
  configure_file("${PROJECT_SOURCE_DIR}/cmake/aeongui.rb.in"
                 "${CMAKE_BINARY_DIR}/aeongui.rb" @ONLY)
  return()
endif()

enable_language(CXX C)

enable_testing()

if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
  cmake_policy(SET CMP0072 NEW)
  cmake_policy(SET CMP0074 NEW)
endif(COMMAND cmake_policy)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON
    CACHE INTERNAL "Force compilation database")

include(CheckIncludeFile)
include(CheckIncludeFileCXX)
include(CheckIncludeFiles)
include(vscode)

set(OpenGL_GL_PREFERENCE GLVND)
find_package(OpenGL)

find_program(BASH_EXECUTABLE bash HINTS ENV MINGW_PREFIX MSYS2_PATH)
find_program(ZSH_EXECUTABLE zsh HINTS ENV MINGW_PREFIX MSYS2_PATH)

find_program(GREP_EXECUTABLE grep
             HINTS ENV
                   MINGW_PREFIX
                   MSYS2_PATH
                   ProgramFiles
             PATH_SUFFIXES usr/bin Git/usr/bin)

if(WIN32)
  find_program(ASTYLE_EXECUTABLE astyle
               HINTS ENV ProgramData
               PATH_SUFFIXES /usr/bin chocolatey/bin)
  find_program(FIND_EXECUTABLE find
               HINTS ENV ProgramFiles
               PATH_SUFFIXES /usr/bin Git/usr/bin)
else()
  find_program(FIND_EXECUTABLE find)
  find_program(ASTYLE_EXECUTABLE astyle)
endif()

set(ASTYLE_OPTIONS "--options=${PROJECT_SOURCE_DIR}/astylerc")
if(IS_DIRECTORY "${PROJECT_SOURCE_DIR}/.git/hooks")
  configure_file("${PROJECT_SOURCE_DIR}/cmake/pre-commit.in"
                 "${PROJECT_SOURCE_DIR}/.git/hooks/pre-commit")
endif()

find_program(XXD_EXECUTABLE xxd HINTS ENV ProgramData PATH_SUFFIXES /usr/bin chocolatey/bin)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

option(USE_ZLIB "Use the zlib compression library" ON)
option(USE_PNG "Enable PNG image support (Requires zlib)" ON)
option(USE_JPEG "Enable JPEG image support" ON)
option(USE_CUDA "Enable CUDA support")
option(BUILD_UNIT_TESTS "Enable Unit Tests using GTest/GMock" ON)
option(DISABLE_DEMOS "Disable building demo applications" OFF)
option(ENABLE_HTML "Enable experimental HTML/XHTML support (requires Yoga)" OFF)
set(AEONGUI_BACKEND "Cairo" CACHE STRING "2D rendering backend (Cairo or Skia)")
set_property(CACHE AEONGUI_BACKEND PROPERTY STRINGS Cairo Skia)
set(AEONGUI_SKIA_ROOT "" CACHE PATH "Skia installation root (used when AEONGUI_BACKEND=Skia, especially on macOS)")
set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "One of Debug|Release|RelWithDebInfo|MinSizeRel")
set(HTTP_PROXY "" CACHE STRING "Specify a proxy server if required for downloads")
set(HTTPS_PROXY "" CACHE STRING "Specify a proxy server if required for downloads")

if(NOT "${HTTP_PROXY}" STREQUAL "")
	set(ENV{http_proxy} "${HTTP_PROXY}")
endif()
if(NOT "${HTTPS_PROXY}" STREQUAL "")
	set(ENV{https_proxy} "${HTTPS_PROXY}")
endif()

# On macOS, prefer the Homebrew skia-aeongui prefix when using Skia backend.
if(APPLE AND AEONGUI_BACKEND STREQUAL "Skia")
  set(AEONGUI_SKIA_EFFECTIVE_ROOT "")
  set(_aeongui_skia_root "${AEONGUI_SKIA_ROOT}")
  if(NOT _aeongui_skia_root)
    set(_aeongui_skia_root "$ENV{SKIA_ROOT}")
  endif()
  if(NOT _aeongui_skia_root)
    if(EXISTS "/opt/homebrew/opt/skia-aeongui")
      set(_aeongui_skia_root "/opt/homebrew/opt/skia-aeongui")
    elseif(EXISTS "/usr/local/opt/skia-aeongui")
      set(_aeongui_skia_root "/usr/local/opt/skia-aeongui")
    endif()
  endif()

  if(_aeongui_skia_root)
    set(AEONGUI_SKIA_EFFECTIVE_ROOT "${_aeongui_skia_root}")
    list(PREPEND CMAKE_PREFIX_PATH "${_aeongui_skia_root}")
    list(PREPEND CMAKE_LIBRARY_PATH "${_aeongui_skia_root}/lib")
    list(PREPEND CMAKE_INCLUDE_PATH "${_aeongui_skia_root}/include")
    if(DEFINED ENV{PKG_CONFIG_PATH} AND NOT "$ENV{PKG_CONFIG_PATH}" STREQUAL "")
      set(ENV{PKG_CONFIG_PATH} "${_aeongui_skia_root}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
    else()
      set(ENV{PKG_CONFIG_PATH} "${_aeongui_skia_root}/lib/pkgconfig")
    endif()
    message(STATUS "Skia backend on macOS: using SKIA_ROOT='${_aeongui_skia_root}'")
  else()
    message(STATUS "Skia backend on macOS: SKIA_ROOT is not set and no default skia-aeongui prefix was found")
  endif()
endif()

set(CMAKE_DEBUG_POSTFIX "d")

if(USE_CUDA)
	message(STATUS "Looking for CUDA toolkit")
	find_package(CUDA)
	add_definitions(-DUSE_CUDA)
endif(USE_CUDA)

if(USE_ZLIB)
	find_package(ZLIB)
  if(ZLIB_FOUND)
    include_directories(${ZLIB_INCLUDE_DIRS})
  else()
    message(FATAL_ERROR "zlib not found, but is required when USE_ZLIB is ON, you can still build AeonGUI without zlib by setting USE_ZLIB to OFF")
  endif()
endif(USE_ZLIB)
if(USE_PNG)
	find_package(PNG)
  if(PNG_FOUND)
    include_directories(${PNG_INCLUDE_DIRS})
  else()
    message(FATAL_ERROR "libpng not found, but is required when USE_PNG is ON, you can still build AeonGUI without libpng by setting USE_PNG to OFF")
  endif()
endif(USE_PNG)
if(USE_JPEG)
	find_package(JPEG)
  if(JPEG_FOUND)
    include_directories(${JPEG_INCLUDE_DIRS})
  else()
    message(FATAL_ERROR "libjpeg not found, but is required when USE_JPEG is ON, you can still build AeonGUI without libjpeg by setting USE_JPEG to OFF")
  endif()
endif(USE_JPEG)

# Install pkgconf from vcpkg on MSVC if the following does not work on Visual Studio Builds
# vcpkg.exe --triplet=x64-windows install pkgconf
find_package(PkgConfig)
if(PKG_CONFIG_FOUND)
    if(BASH_EXECUTABLE AND MSYS)
      # In order to properly use pkg-config in a msys2 environment we need to jump
      # through some hoops. CMake seems to call pkg-config without any of the bash
      # environment set up, so we need to set the proper pkg config path inside the
      # cmake process address space.
      execute_process(COMMAND ${BASH_EXECUTABLE} --login -c "echo $PKG_CONFIG_PATH"
                      OUTPUT_VARIABLE PKG_CONFIG_PATH)
      set(ENV{PKG_CONFIG_PATH} ${PKG_CONFIG_PATH})
    elseif(MSVC AND VCPKG_INSTALLED_DIR AND VCPKG_TARGET_TRIPLET)
      set(ENV{PKG_CONFIG_PATH} ${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/lib/pkgconfig)
    endif()
    pkg_check_modules(HARFBUZZ harfbuzz)
    pkg_check_modules(FONTCONFIG fontconfig)
    if(AEONGUI_BACKEND STREQUAL "Cairo")
      pkg_check_modules(CAIRO cairo)
      pkg_check_modules(PANGO pango pangocairo pangofc)
    elseif(AEONGUI_BACKEND STREQUAL "Skia")
      # Help vcpkg's Skia config find Windows SDK libraries (FontSub, Usp10).
      if(MSVC)
        # Determine the SDK platform version (VS generator sets this automatically).
        set(_winsdk_ver "${CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION}")
        if(NOT _winsdk_ver)
          # Fallback: pick the latest installed SDK version.
          file(GLOB _winsdk_versions LIST_DIRECTORIES true
            "C:/Program Files (x86)/Windows Kits/10/Lib/10.*")
          if(_winsdk_versions)
            list(SORT _winsdk_versions COMPARE NATURAL ORDER DESCENDING)
            list(GET _winsdk_versions 0 _winsdk_latest)
            get_filename_component(_winsdk_ver "${_winsdk_latest}" NAME)
          endif()
        endif()
        if(_winsdk_ver)
          set(_winsdk_um "C:/Program Files (x86)/Windows Kits/10/Lib/${_winsdk_ver}/um/x64")
          if(IS_DIRECTORY "${_winsdk_um}")
            list(APPEND CMAKE_LIBRARY_PATH "${_winsdk_um}")
          endif()
        endif()
      endif()
      find_package(unofficial-skia CONFIG QUIET)
      if(NOT unofficial-skia_FOUND)
        pkg_check_modules(SKIA QUIET skia)
        if(NOT SKIA_FOUND)
          message(FATAL_ERROR
            "Skia backend selected but Skia was not found.\n"
            "macOS setup with Aeon tap:\n"
            "  brew tap aeongames/ports\n"
            "  brew install --build-from-source aeongames/ports/skia-aeongui\n"
            "  cmake -S . -B build -DAEONGUI_BACKEND=Skia -DAEONGUI_SKIA_ROOT=\"$(brew --prefix skia-aeongui)\"\n"
            "  export SKIA_ROOT=\"$(brew --prefix skia-aeongui)\"\n"
            "  export PKG_CONFIG_PATH=\"$SKIA_ROOT/lib/pkgconfig:$PKG_CONFIG_PATH\"\n"
            "Then rerun CMake."
          )
        endif()
      else()
        set(SKIA_LIBRARIES unofficial::skia::skia)
      endif()
      pkg_check_modules(PANGO pango pangoft2 pangofc)
    endif()
endif()

find_package(Freetype)
find_package(LibXml2)

find_package(X11)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR APPLE)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -finput-charset=UTF-8 -fexec-charset=UTF-8 -Wfatal-errors -Wall -Drestrict=__restrict__")
  set(CMAKE_CXX_FLAGS_RELEASE "-DNDEBUG -fomit-frame-pointer -O3")
  set(CMAKE_CXX_FLAGS_DEBUG "-DDEBUG -g -O0")
elseif(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -utf-8 -validate-charset -D_CRT_SUPPRESS_RESTRICT -Drestrict=__restrict")
endif()

if(CMAKE_C_COMPILER_ID STREQUAL "GNU" OR CMAKE_C_COMPILER_ID STREQUAL "Clang")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -finput-charset=UTF-8 -fexec-charset=UTF-8 -Wall")
  set(CMAKE_C_FLAGS_RELEASE "-DNDEBUG -fomit-frame-pointer -O3")
	set(CMAKE_C_FLAGS_DEBUG "-DDEBUG -g -O0")
endif()

# _ALIGNED is used by libcss (NetSurf) on struct css_rule to ensure correct
# alignment when casting between base/derived struct types (C inheritance).
# On ARM, GCC warns about casts increasing alignment requirements without this.
if(MSVC)
  add_definitions(-D_USE_MATH_DEFINES "-D_ALIGNED=__declspec(align(8))")
else()
  add_definitions(-D_USE_MATH_DEFINES "-D_ALIGNED=__attribute__((aligned))")
endif()

if(ENABLE_HTML)
  message(STATUS "ENABLE_HTML is ON: enabling experimental HTML/XHTML support")
  include(yoga)
endif()

if(BUILD_UNIT_TESTS)
  add_subdirectory(tests)
endif()

# Capture the AeonGUI source dir at parse time so the helper script can be
# located whether AeonGUI is built standalone or as a sub-project (e.g. via
# FetchContent), where CMAKE_SOURCE_DIR points at the parent project.
set(_AEONGUI_FIX_COMPILE_COMMANDS_SCRIPT
    "${CMAKE_CURRENT_LIST_DIR}/cmake/FixCompileCommands.cmake")

function(fix_compile_commands target)
  if(MINGW)
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND "${CMAKE_COMMAND}"
                -DCOMPILE_COMMANDS_JSON=${CMAKE_BINARY_DIR}/compile_commands.json
                -P "${_AEONGUI_FIX_COMPILE_COMMANDS_SCRIPT}"
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
        COMMENT "Fixing compilation database for ${target}"
        VERBATIM
    )
  endif()
endfunction()
if(MINGW)
  add_custom_target(fix_compile_commands ALL
      COMMAND "${CMAKE_COMMAND}"
              -DCOMPILE_COMMANDS_JSON=${CMAKE_BINARY_DIR}/compile_commands.json
              -P "${_AEONGUI_FIX_COMPILE_COMMANDS_SCRIPT}"
      WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
      COMMENT "Fixing compilation database"
      VERBATIM
  )
endif()


include(parser)
include(opensans)
include_directories(include common demos/common core)
add_subdirectory(css)
add_subdirectory(core)

# Demo native plugin for button.svg
add_library(button_plugin SHARED images/button.c)
target_include_directories(button_plugin PRIVATE include)
set_target_properties(button_plugin PROPERTIES
  OUTPUT_NAME "button"
  PREFIX ""
  DEBUG_POSTFIX ""
  RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/images"
  RUNTIME_OUTPUT_DIRECTORY_DEBUG "${PROJECT_SOURCE_DIR}/images"
  RUNTIME_OUTPUT_DIRECTORY_RELEASE "${PROJECT_SOURCE_DIR}/images"
  LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/images"
)

set(DEMO_DIRS "")
if(NOT DISABLE_DEMOS)
  if(OpenGL_FOUND)
    add_subdirectory(demos/OpenGL)
    list(APPEND DEMO_DIRS ${PROJECT_SOURCE_DIR}/demos/OpenGL)
  endif()
  if(APPLE)
    add_subdirectory(demos/Metal)
    list(APPEND DEMO_DIRS ${PROJECT_SOURCE_DIR}/demos/Metal)
  endif()
  add_subdirectory(demos/Vulkan)
  list(APPEND DEMO_DIRS ${PROJECT_SOURCE_DIR}/demos/Vulkan)
  if(WIN32)
    add_subdirectory(demos/Direct3D12)
    list(APPEND DEMO_DIRS ${PROJECT_SOURCE_DIR}/demos/Direct3D12)
  endif()
endif()
configure_vscode(${DEMO_DIRS})

add_docs_target()

install(TARGETS AeonGUI
    EXPORT aeongui
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
	INCLUDES DESTINATION include
)

install(
  FILES
    include/aeongui/AeonGUI.hpp
    include/aeongui/AABB.hpp
    include/aeongui/Attribute.hpp
    include/aeongui/AttributeMap.hpp
    include/aeongui/Canvas.hpp
    include/aeongui/Color.hpp
    include/aeongui/CSSSelector.hpp
    include/aeongui/DrawType.hpp
    include/aeongui/ElementFactory.hpp
    include/aeongui/FontDatabase.hpp
    include/aeongui/LogLevel.hpp
    include/aeongui/Matrix2x3.hpp
    include/aeongui/Path.hpp
    include/aeongui/Platform.hpp
    include/aeongui/PluginAPI.h
    include/aeongui/RasterImage.hpp
    include/aeongui/ResourceLoader.hpp
    include/aeongui/Rect.hpp
    include/aeongui/StringLiteral.hpp
    include/aeongui/StyleSheet.hpp
    include/aeongui/TextLayout.hpp
    include/aeongui/Transform.hpp
    include/aeongui/Vector2.hpp
	DESTINATION include/aeongui
)

install(
  DIRECTORY include/aeongui/dom/
  DESTINATION include/aeongui/dom
  FILES_MATCHING PATTERN "*.hpp"
)

install(
    EXPORT aeongui
    DESTINATION share/aeongui
    FILE aeongui-config.cmake
)

file(INSTALL LICENSE DESTINATION share/aeongui RENAME copyright)

# --- Packaging templates ---
configure_file("${PROJECT_SOURCE_DIR}/cmake/PKGBUILD-arch.in"
               "${CMAKE_BINARY_DIR}/PKGBUILD-arch" @ONLY NEWLINE_STYLE UNIX)
configure_file("${PROJECT_SOURCE_DIR}/cmake/PKGBUILD-msys2.in"
               "${CMAKE_BINARY_DIR}/PKGBUILD-msys2" @ONLY NEWLINE_STYLE UNIX)
configure_file("${PROJECT_SOURCE_DIR}/cmake/vcpkg-port-vcpkg.json.in"
               "${CMAKE_BINARY_DIR}/vcpkg-port/vcpkg.json" @ONLY)
configure_file("${PROJECT_SOURCE_DIR}/cmake/vcpkg-port-portfile.cmake.in"
               "${CMAKE_BINARY_DIR}/vcpkg-port/portfile.cmake" @ONLY)
configure_file("${PROJECT_SOURCE_DIR}/cmake/aeongui.rb.in"
               "${CMAKE_BINARY_DIR}/aeongui.rb" @ONLY)
