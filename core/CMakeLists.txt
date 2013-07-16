SET(AEONGUI_HEADERS
../include/AeonGUI.h
../include/KeyListener.h
../include/MouseListener.h
../include/Widget.h
../include/Button.h
../include/ScrollBar.h
../include/MainWindow.h
../include/Static.h
../include/Image.h
../include/Font.h
../include/Renderer.h
../include/Integer.h
../include/Rect.h
../include/Color.h
)

SET(AEONGUI_SOURCES
AeonGUI.cpp
Renderer.cpp
Font.cpp
Widget.cpp
Button.cpp
ScrollBar.cpp
MainWindow.cpp
Image.cpp
Rect.cpp
Color.cpp
)

if(USE_CUDA)
# Set Arch to sm_20 for printf inside kernel 
# set(CUDA_NVCC_FLAGS -arch=sm_20;${CUDA_NVCC_FLAGS})
set(CUDA_KERNELS cuda/Renderer.cu)
cuda_compile(CUDA_WRAPPERS ${CUDA_KERNELS})
set(AEONGUI_SOURCES ${AEONGUI_SOURCES} ${CUDA_KERNELS})
source_group("CUDA Kernels" FILES ${CUDA_KERNELS})
include_directories(${CUDA_TOOLKIT_INCLUDE} cuda)
set(AEONGUI_LIBRARIES ${AEONGUI_LIBRARIES} ${CUDA_CUDART_LIBRARY})
endif(USE_CUDA)

SET(AEONGUI_RESOURCES
resources/close.c
resources/maximize.c
resources/minimize.c
resources/restore.c
resources/scrolldown.c
resources/scrolldownpressed.c
resources/scrollup.c
resources/scrolluppressed.c
resources/scrollleft.c
resources/scrollleftpressed.c
resources/scrollright.c
resources/scrollrightpressed.c
resources/close.h
resources/maximize.h
resources/minimize.h
resources/restore.h
resources/scrolldown.h
resources/scrolldownpressed.h
resources/scrollup.h
resources/scrolluppressed.h
resources/scrollleft.h
resources/scrollleftpressed.h
resources/scrollright.h
resources/scrollrightpressed.h
resources/close_down.c
resources/maximize_down.c
resources/minimize_down.c
resources/restore_down.c
resources/close_down.h
resources/maximize_down.h
resources/minimize_down.h
resources/restore_down.h
)

set(AEONGUI_LIBRARIES ${AEONGUI_LIBRARIES} pcx)
if(USE_PNG)
set(AEONGUI_LIBRARIES ${AEONGUI_LIBRARIES} ${PNG_LIBRARIES} ${ZLIB_LIBRARIES})
endif(USE_PNG)

SOURCE_GROUP("Resource Files" FILES ${AEONGUI_RESOURCES})
add_library(AeonGUI ${AEONGUI_HEADERS} ${AEONGUI_SOURCES} ${AEONGUI_RESOURCES})
target_link_libraries(AeonGUI ${AEONGUI_LIBRARIES})
