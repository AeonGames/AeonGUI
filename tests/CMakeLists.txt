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

enable_testing()

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        b796f7d # release-1.13.0
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

find_package(GTest CONFIG REQUIRED)

include_directories(${GTEST_INCLUDE_DIR}
                    ${CMAKE_SOURCE_DIR}/include
                    ${CAIRO_INCLUDE_DIRS})
set(TESTED_SRCS
  Location.cpp
)

set(TEST_SRCS
    Main.cpp
)

source_group("Tested" FILES ${TESTED_SRCS})
source_group("Tests" FILES ${TEST_SRCS})

add_executable(unit-tests ${TEST_SRCS} ${TESTED_SRCS})
target_link_libraries(unit-tests PRIVATE AeonGUI GTest::gtest GTest::gmock)

if(MSVC)
  # VCPKG compiles GTest/GMock DLLs.
  set_target_properties(
    unit-tests
    PROPERTIES
      COMPILE_FLAGS
      "-DSOURCE_PATH=\"\\\"${CMAKE_SOURCE_DIR}\\\"\" -D_CRT_SECURE_NO_WARNINGS -wd4251 -wd4275"
    )
else()
  set_target_properties(
    unit-tests
    PROPERTIES COMPILE_FLAGS "-DSOURCE_PATH=\"\\\"${CMAKE_SOURCE_DIR}\\\"\"")
endif()
add_test(NAME unit-tests COMMAND unit-tests)
configure_vscode(${CMAKE_SOURCE_DIR}/tests)
