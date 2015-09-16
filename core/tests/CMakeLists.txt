include_directories(${GTEST_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/include)
set(TEST_SRCS
	OverlayTest.cpp
    )
source_group("Tests" FILES ${TEST_SRCS})
add_executable(core-tests ${TEST_SRCS})
add_dependencies(core-tests AeonGUI ${GTEST_LIBRARY} ${GMOCK_LIBRARY} ${GMOCK_MAIN_LIBRARY})
target_link_libraries(core-tests AeonGUI ${GTEST_LIBRARY} ${GMOCK_MAIN_LIBRARY})
set_target_properties(core-tests PROPERTIES
    COMPILE_FLAGS "-D_CRT_SECURE_NO_WARNINGS")
add_test(NAME core-tests COMMAND core-tests)
