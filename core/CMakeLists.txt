
find_package(BISON)
find_package(FLEX)

set(AEONGUI_HEADERS
    ../include/aeongui/AeonGUI.h
    ../include/aeongui/Platform.h
    ../include/aeongui/Rect.h
    ../include/aeongui/ElementFactory.h
    ../include/aeongui/StringLiteral.h
    ../include/aeongui/Document.h
    ../include/aeongui/Window.h
    ../include/aeongui/Canvas.h
    ../include/aeongui/CairoCanvas.h
    ../include/aeongui/Path.h
    ../include/aeongui/CairoPath.h
    ../include/aeongui/AABB.h
    ../include/aeongui/Matrix2x3.h
    ../include/aeongui/Transform.h
    ../include/aeongui/Vector2.h
    ../include/aeongui/DrawType.h
    ../include/aeongui/JavaScript.h
    ../include/aeongui/JsObjectWrap.h
    ../include/aeongui/Color.h
)

set(AEONGUI_SOURCES
    AeonGUI.cpp
    ElementFactory.cpp
    Window.cpp
    Document.cpp
    Rect.cpp
    Transform.cpp
    Vector2.cpp
    Matrix2x3.cpp
    AABB.cpp
    Canvas.cpp
    CairoCanvas.cpp
    Path.cpp
    CairoPath.cpp
    JavaScript.cpp
    JsObjectWrap.cpp
    Color.cpp
    dom/EventTarget.cpp
    dom/Node.cpp
    dom/Element.cpp
    dom/SVGElement.cpp
    dom/SVGSVGElement.cpp
    dom/SVGGradientElement.cpp
    dom/SVGLinearGradientElement.cpp
    dom/SVGGElement.cpp
    dom/SVGUseElement.cpp
    dom/SVGStopElement.cpp
    dom/SVGDefsElement.cpp
    dom/SVGPathElement.cpp
    dom/SVGGeometryElement.cpp
    dom/SVGGraphicsElement.cpp
    dom/SVGRectElement.cpp
    dom/SVGLineElement.cpp
    dom/SVGPolylineElement.cpp
    dom/SVGPolygonElement.cpp
    dom/SVGCircleElement.cpp
    dom/SVGEllipseElement.cpp
    dom/Script.cpp
    dom/Text.cpp
    dom/EventTarget.h
    dom/Node.h
    dom/Element.h
    dom/SVGGElement.h
    dom/SVGSVGElement.h
    dom/SVGElement.h
    dom/SVGGradientElement.h
    dom/SVGLinearGradientElement.h
    dom/SVGUseElement.h
    dom/SVGStopElement.h
    dom/SVGDefsElement.h
    dom/SVGPathElement.h
    dom/SVGGraphicsElement.h
    dom/SVGGeometryElement.h
    dom/SVGRectElement.h
    dom/SVGLineElement.h
    dom/SVGPolylineElement.h
    dom/SVGPolygonElement.h
    dom/SVGCircleElement.h
    dom/SVGEllipseElement.h
    dom/Script.h
    dom/Text.h
)

if(USE_CUDA)
	# Set Arch to sm_20 for printf inside kernel
	# set(CUDA_NVCC_FLAGS -arch=sm_20;${CUDA_NVCC_FLAGS})
	set(CUDA_KERNELS cuda/Renderer.cu)
	# set(CUDA_PROPAGATE_HOST_FLAGS OFF)
	cuda_compile(CUDA_WRAPPERS ${CUDA_KERNELS})
	set(AEONGUI_SOURCES ${AEONGUI_SOURCES} ${CUDA_WRAPPERS})
	source_group("CUDA Kernels" FILES ${CUDA_KERNELS})
	include_directories(${CUDA_TOOLKIT_INCLUDE} cuda)
	set(AEONGUI_LIBRARIES ${AEONGUI_LIBRARIES} ${CUDA_CUDART_LIBRARY})
endif(USE_CUDA)


parser_code(path_data ${CMAKE_CURRENT_SOURCE_DIR}/parsers ${CMAKE_CURRENT_SOURCE_DIR}/parsers)
parser_code(style ${CMAKE_CURRENT_SOURCE_DIR}/parsers ${CMAKE_CURRENT_SOURCE_DIR}/parsers)

include_directories(${CAIRO_INCLUDE_DIRS} ${FREETYPE_INCLUDE_DIR_freetype2} ${FREETYPE_INCLUDE_DIR_ft2build} ${V8_INCLUDE_DIRS})
add_library(AeonGUI SHARED ${AEONGUI_HEADERS} ${AEONGUI_SOURCES} ${AEONGUI_RESOURCES} ${BISON_path_data_parser_OUTPUTS} ${FLEX_path_data_lexer_OUTPUTS} ${BISON_style_parser_OUTPUTS} ${FLEX_style_lexer_OUTPUTS})
set_target_properties(AeonGUI PROPERTIES COMPILE_FLAGS "-DNOMINMAX -D_CRT_SECURE_NO_WARNINGS")
target_link_libraries(AeonGUI PUBLIC ${CAIRO_LIBRARIES} ${LIBXML2_LIBRARIES} ${FREETYPE_LIBRARIES} ${V8_TARGET})

if(V8_SNAPSHOT_BLOB)
    add_custom_command(TARGET AeonGUI
        COMMAND ${CMAKE_COMMAND} -E copy ${V8_SNAPSHOT_BLOB} $<TARGET_FILE_DIR:AeonGUI>/
        MAIN_DEPENDENCY ${V8_SNAPSHOT_BLOB}
        COMMENT "Copying V8 snapshot blob")
endif()

target_precompile_headers(AeonGUI
  PUBLIC
    ${AEONGUI_HEADERS}
  PRIVATE
    [["v8.h"]]
    [["libplatform/libplatform.h"]]
    <cairo.h>
    <string>
    <unordered_map>
    <iostream>
    <algorithm>
    <cmath>
    <limits>
    <vector>
    <functional>
    <memory>
    <utility>
    <tuple>
    <regex>
)

fix_compile_commands(AeonGUI)

if(BUILD_UNIT_TESTS)
	subdirs(tests)
endif()
