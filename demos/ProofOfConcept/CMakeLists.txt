set(POC_SOURCES Window.h Window.cpp)
add_executable(PoC ${POC_SOURCES})
target_link_libraries(PoC ${OPENGL_LIBRARIES})