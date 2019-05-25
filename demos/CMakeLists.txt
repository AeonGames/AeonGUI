# Avoid compilation errors due to missing Linux implementation.
if(WIN32)
    subdirs(OpenGL)
endif()
