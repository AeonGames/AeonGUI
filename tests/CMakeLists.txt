# Copyright (C) 2023,2025 Rodrigo Jose Hernandez Cordoba
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

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  include(FetchContent)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        52eb810 # release-1.17.0
  )
  # For Windows: Prevent overriding the parent project's compiler/linker settings
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
endif()

find_package(GTest CONFIG REQUIRED)

include_directories(${GTEST_INCLUDE_DIR}
                    ${PROJECT_SOURCE_DIR}/include)
set(TESTED_SRCS
  DocumentTest.cpp
  DOMMatrixReadOnlyTest.cpp
  DOMMatrixTest.cpp
  DOMPointReadOnlyTest.cpp
  DOMPointTest.cpp
  DOMRectReadOnlyTest.cpp
  DOMRectTest.cpp
  ErrorHandlingTest.cpp
  HitTestTest.cpp
  Location.cpp
  NodeTest.cpp
  RasterImageTest.cpp
  SMILAnimationTest.cpp
  SVGAnimatedTypesTest.cpp
  SVGImageElementTest.cpp
  SVGLengthTest.cpp
  SVGScriptElementTest.cpp
  ThreadSafetyTest.cpp
  UIEventsTest.cpp
)

if(ENABLE_HTML)
  list(APPEND TESTED_SRCS HTMLLayoutSpikeTest.cpp HTMLElementFactoryTest.cpp HTMLLayoutEngineTest.cpp HTMLRenderTest.cpp)
endif()

set(TEST_SRCS
    Main.cpp
)

source_group("Tested" FILES ${TESTED_SRCS})
source_group("Tests" FILES ${TEST_SRCS})

add_executable(unit-tests ${TEST_SRCS} ${TESTED_SRCS})
target_link_libraries(unit-tests PRIVATE AeonGUI GTest::gtest GTest::gmock)

if(ENABLE_HTML)
  target_link_libraries(unit-tests PRIVATE yogacore)
endif()

if(MSVC)
  # VCPKG compiles GTest/GMock DLLs.
  set_target_properties(
    unit-tests
    PROPERTIES
      COMPILE_FLAGS
      "-DSOURCE_PATH=\"\\\"${PROJECT_SOURCE_DIR}\\\"\" -D_CRT_SECURE_NO_WARNINGS -wd4251 -wd4275"
    )
else()
  set_target_properties(
    unit-tests
    PROPERTIES COMPILE_FLAGS "-DSOURCE_PATH=\"\\\"${PROJECT_SOURCE_DIR}\\\"\"")
endif()
add_test(NAME unit-tests COMMAND unit-tests)
set_tests_properties(unit-tests PROPERTIES
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
)

# Test plugin shared library for SVGScriptElement tests
add_library(test_plugin SHARED test_plugin.c)
target_include_directories(test_plugin PRIVATE ${PROJECT_SOURCE_DIR}/include)
set_target_properties(test_plugin PROPERTIES
  PREFIX ""
  DEBUG_POSTFIX ""
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

configure_vscode(${PROJECT_SOURCE_DIR}/tests)
