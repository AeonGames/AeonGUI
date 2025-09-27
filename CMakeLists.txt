cmake_minimum_required(VERSION 3.10)
project(AeonGUI CXX C)
enable_testing()

if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
  cmake_policy(SET CMP0072 NEW)
endif(COMMAND cmake_policy)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
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
find_package(Git)

find_program(BASH_EXECUTABLE bash HINTS ENV MINGW_PREFIX MSYS2_PATH)
find_program(ZSH_EXECUTABLE zsh HINTS ENV MINGW_PREFIX MSYS2_PATH)

function(gnu_sed_check validator_result_var item)
  execute_process(COMMAND ${item} --version
                  RESULT_VARIABLE sed_result
                  OUTPUT_VARIABLE sed_output
                  ERROR_VARIABLE sed_error
                  OUTPUT_STRIP_TRAILING_WHITESPACE
                  ERROR_STRIP_TRAILING_WHITESPACE)
  if(sed_output MATCHES "GNU sed")
    message(STATUS "Found GNU sed: ${item}")
    set(${validator_result_var} TRUE PARENT_SCOPE)
  else()
    set(${validator_result_var} FALSE PARENT_SCOPE)
  endif()
endfunction()

find_program(SED_EXECUTABLE NAMES gsed sed
            PATHS ENV MSYSTEM_PREFIX ENV ProgramFiles
            PATH_SUFFIXES
                  bin
                  usr/bin
                  Git/usr/bin
                  VALIDATOR gnu_sed_check
)

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

set(ASTYLE_OPTIONS "--options=${CMAKE_SOURCE_DIR}/astylerc")
configure_file("${CMAKE_SOURCE_DIR}/cmake/pre-commit.in"
               "${CMAKE_SOURCE_DIR}/.git/hooks/pre-commit")

find_program(XXD_EXECUTABLE xxd HINTS ENV ProgramData PATH_SUFFIXES /usr/bin chocolatey/bin)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

option(USE_ZLIB "Use the zlib compression library")
option(USE_PNG "Enable PNG image support (Requires zlib)")
option(USE_CUDA "Enable CUDA support")
option(BUILD_UNIT_TESTS "Enable Unit Tests using GTest/GMock")
set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "One of Debug|Release|RelWithDebInfo|MinSizeRel")
set(HTTP_PROXY "" CACHE STRING "Specify a proxy server if required for downloads")
set(HTTPS_PROXY "" CACHE STRING "Specify a proxy server if required for downloads")

if(NOT "${HTTP_PROXY}" STREQUAL "")
	set(ENV{http_proxy} "${HTTP_PROXY}")
endif()
if(NOT "${HTTPS_PROXY}" STREQUAL "")
	set(ENV{https_proxy} "${HTTPS_PROXY}")
endif()

set(CMAKE_DEBUG_POSTFIX "d")

if(USE_CUDA)
	message(STATUS "Looking for CUDA toolkit")
	find_package(CUDA)
	add_definitions(-DUSE_CUDA)
endif(USE_CUDA)

if(USE_ZLIB)
	find_package(zlib)
endif(USE_ZLIB)
if(USE_PNG)
	find_package(libpng)
endif(USE_PNG)

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
    pkg_check_modules(CAIRO cairo)
endif()

find_package(Freetype)
if(FREETYPE_FOUND)
	include_directories(${FREETYPE_INCLUDE_DIRS})
endif()
find_package(LibXml2)
if(LIBXML2_FOUND)
	include_directories(${LIBXML2_INCLUDE_DIR})
endif()

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

# TODO: Set _ALIGNED to the proper attribute if/when required
add_definitions(-D_USE_MATH_DEFINES  -D_ALIGNED=)

if(BUILD_UNIT_TESTS)
  add_subdirectory(tests)
endif()

function(fix_compile_commands target)
  if(SED_EXECUTABLE)
  add_custom_command(TARGET ${target} POST_BUILD
      COMMAND ${SED_EXECUTABLE} -i 's/\\/\\\(.\\\)\\//\\1:\\//g' ${CMAKE_BINARY_DIR}/compile_commands.json
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
      COMMENT "Fixing compilation Database."
  )
  endif()
endfunction()
if(SED_EXECUTABLE AND MINGW)
  add_custom_target(fix_compile_commands ALL
      COMMAND ${SED_EXECUTABLE} -i 's/\\/\\\(.\\\)\\//\\1:\\//g' ${CMAKE_BINARY_DIR}/compile_commands.json
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
      COMMENT "Fixing compilation Database."
  )
endif()


include(parser)
include_directories(include common demos/common core)
add_subdirectory(css)
add_subdirectory(core)
add_subdirectory(tests)
add_subdirectory(demos/OpenGL)

install(TARGETS AeonGUI
    EXPORT aeongui
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
	INCLUDES DESTINATION include
)

install(
  FILES
    include/aeongui/AeonGUI.h
    include/aeongui/AABB.h
    include/aeongui/AttributeMap.h
    include/aeongui/Attribute.hpp
    include/aeongui/Window.h
    include/aeongui/Document.h
    include/aeongui/Platform.h
    include/aeongui/Canvas.h
    include/aeongui/CairoCanvas.h
    include/aeongui/DrawType.h
    include/aeongui/Color.h
    include/aeongui/Vector2.h
    include/aeongui/Element.h
    include/aeongui/Transform.h
    include/aeongui/Matrix2x3.h
	DESTINATION include/aeongui
)

install(
    EXPORT aeongui
    DESTINATION share/aeongui
    FILE aeongui-config.cmake
)

file(INSTALL LICENSE DESTINATION share/aeongui RENAME copyright)
