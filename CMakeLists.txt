cmake_minimum_required(VERSION 3.0.2)
project(AeonGUI CXX C)

if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
  cmake_policy(SET CMP0072 NEW)
endif(COMMAND cmake_policy)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH  ${CMAKE_BINARY_DIR}/bin)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON
    CACHE INTERNAL "Force compilation database")

include(CheckIncludeFile)
include(CheckIncludeFileCXX)
include(CheckIncludeFiles)
include(duktape)
include(vscode)

find_package(OpenGL)
find_package(Git)
aeongui_configure_duktape()

find_program(BASH_EXECUTABLE bash HINTS ENV MINGW_PREFIX MSYS2_PATH)
find_program(SED_EXECUTABLE sed
             HINTS ENV
                   MINGW_PREFIX
                   MSYS2_PATH
                   ProgramFiles
             PATH_SUFFIXES bin usr/bin Git/usr/bin)
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
set(CMAKE_BUILD_TYPE "DEBUG" CACHE STRING "One of DEBUG|RELEASE|RELWITHDEBINFO|MINSIZEREL")
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

find_package(PkgConfig)
if(PKG_CONFIG_FOUND AND BASH_EXECUTABLE AND MSYS)
  # In order to properly use pkg-config in a msys2 environment we need to jump
  # through some hoops. CMake seems to call pkg-config without any of the bash
  # environment set up, so we need to set the proper pkg config path inside the
  # cmake process address space.
  execute_process(COMMAND ${BASH_EXECUTABLE} --login -c "echo $PKG_CONFIG_PATH"
                  OUTPUT_VARIABLE PKG_CONFIG_PATH)
  set(ENV{PKG_CONFIG_PATH} ${PKG_CONFIG_PATH})
endif()
if(PKG_CONFIG_FOUND)
  pkg_check_modules(CAIRO cairo)
#  pkg_check_modules(LIBRSVG librsvg-2.0)
endif()

if(VCPKG_TARGET_TRIPLET AND NOT CAIRO_FOUND)
  # cairo
  find_package(unofficial-cairo CONFIG REQUIRED)
  set(CAIRO_LIBRARIES unofficial::cairo::cairo)
  set(CAIRO_INCLUDE_DIRS "")
  # librsvg
  #find_package(unofficial-librsvg CONFIG REQUIRED)
  #set(LIBRSVG_LIBRARIES unofficial::librsvg::rsvg-2.0)
  #set(LIBRSVG_INCLUDE_DIRS "")
endif()

include(opensans)
find_package(Freetype)
if(FREETYPE_FOUND)
	include_directories(${FREETYPE_INCLUDE_DIRS})
endif()
find_package(LibXml2)
if(LIBXML2_FOUND)
	include_directories(${LIBXML2_INCLUDE_DIR})
endif()

find_package(X11)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_COMPILER_IS_GNUCXX)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -DNDEBUG -fomit-frame-pointer -O3")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -DDEBUG -g -O0")
endif(CMAKE_COMPILER_IS_GNUCXX)

if(CMAKE_COMPILER_IS_GNUC)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")
  set(CMAKE_C_FLAGS_RELEASE "-DNDEBUG -fomit-frame-pointer -O3")
	set(CMAKE_C_FLAGS_DEBUG "-DDEBUG -g -O0")
endif(CMAKE_COMPILER_IS_GNUC)

add_definitions(-D_USE_MATH_DEFINES)

if(BUILD_UNIT_TESTS)
	include(gmock)
endif()

include_directories(include common demos/common core)
add_subdirectory(core)

add_subdirectory(demos/OpenGL)
configure_vscode(${CMAKE_SOURCE_DIR}/demos/OpenGL)

install(TARGETS AeonGUI duktape
    EXPORT aeongui
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
	INCLUDES DESTINATION include
)

install(
  FILES
    include/aeongui/AABB.h
    include/aeongui/AttributeMap.h
    include/aeongui/Window.h
    include/aeongui/Document.h
    include/aeongui/Platform.h
    include/aeongui/Canvas.h
    include/aeongui/CairoCanvas.h
    include/aeongui/Duktape.h
    include/aeongui/DrawType.h
    include/aeongui/Color.h
    include/aeongui/Vector2.h
    include/aeongui/JavaScript.h
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
