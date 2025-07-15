find_package(BISON)
find_package(FLEX)
add_subdirectory(idl)

set(AEONGUI_HEADERS
    ../include/aeongui/AeonGUI.h
    ../include/aeongui/Platform.h
    ../include/aeongui/Rect.h
    ../include/aeongui/ElementFactory.h
    ../include/aeongui/StringLiteral.h
    ../include/aeongui/Canvas.h
    ../include/aeongui/CairoCanvas.h
    ../include/aeongui/CairoPath.h
    ../include/aeongui/Path.h
    ../include/aeongui/AABB.h
    ../include/aeongui/Matrix2x3.h
    ../include/aeongui/Transform.h
    ../include/aeongui/Vector2.h
    ../include/aeongui/DrawType.h
    ../include/aeongui/Color.h
    ../include/aeongui/StyleSheet.h
)
set(AEONGUI_DOM_HEADERS
    ../include/aeongui/dom/Window.hpp
    ../include/aeongui/dom/Document.hpp
    ../include/aeongui/dom/DOMString.hpp
    ../include/aeongui/dom/AnyType.hpp
    ../include/aeongui/dom/Event.hpp
    ../include/aeongui/dom/EventTarget.hpp
    ../include/aeongui/dom/EventListener.hpp
    ../include/aeongui/dom/Node.hpp
    ../include/aeongui/dom/Element.hpp
    ../include/aeongui/dom/SVGGElement.hpp
    ../include/aeongui/dom/SVGSVGElement.hpp
    ../include/aeongui/dom/SVGElement.hpp
    ../include/aeongui/dom/SVGGradientElement.hpp
    ../include/aeongui/dom/SVGLinearGradientElement.hpp
    ../include/aeongui/dom/SVGUseElement.hpp
    ../include/aeongui/dom/SVGStopElement.hpp
    ../include/aeongui/dom/SVGDefsElement.hpp
    ../include/aeongui/dom/SVGPathElement.hpp
    ../include/aeongui/dom/SVGGraphicsElement.hpp
    ../include/aeongui/dom/SVGGeometryElement.hpp
    ../include/aeongui/dom/SVGRectElement.hpp
    ../include/aeongui/dom/SVGLineElement.hpp
    ../include/aeongui/dom/SVGPolylineElement.hpp
    ../include/aeongui/dom/SVGPolygonElement.hpp
    ../include/aeongui/dom/SVGCircleElement.hpp
    ../include/aeongui/dom/SVGEllipseElement.hpp
    ../include/aeongui/dom/Text.hpp
    ../include/aeongui/dom/USVString.hpp
)

set(AEONGUI_SOURCES
    AeonGUI.cpp
    ElementFactory.cpp
    Rect.cpp
    Transform.cpp
    Vector2.cpp
    Matrix2x3.cpp
    AABB.cpp
    Canvas.cpp
    CairoCanvas.cpp
    Path.cpp
    CairoPath.cpp
    Color.cpp
    StyleSheet.cpp
    CSSSelectHandler.cpp
    CSSSelectHandler.hpp
)
set(AEONGUI_DOM_SOURCES
    dom/Window.cpp
    dom/Document.cpp
    dom/Event.cpp
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
    dom/Text.cpp
    dom/Attribute.cpp
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
#parser_code(style ${CMAKE_CURRENT_SOURCE_DIR}/parsers ${CMAKE_CURRENT_SOURCE_DIR}/parsers)

include_directories(${CAIRO_INCLUDE_DIRS} ${FREETYPE_INCLUDE_DIR_freetype2} ${FREETYPE_INCLUDE_DIR_ft2build} ${LIBXML2_INCLUDE_DIRS})
add_library(AeonGUI SHARED ${AEONGUI_HEADERS} ${AEONGUI_SOURCES} ${AEONGUI_DOM_HEADERS} ${AEONGUI_DOM_SOURCES} ${AEONGUI_RESOURCES} ${BISON_path_data_parser_OUTPUTS} ${FLEX_path_data_lexer_OUTPUTS})
set_target_properties(AeonGUI PROPERTIES COMPILE_FLAGS "-DNOMINMAX -D_CRT_SECURE_NO_WARNINGS")
target_link_libraries(AeonGUI PRIVATE libcss ${pkgcfg_lib_CAIRO_cairo} ${LIBXML2_LIBRARIES} ${FREETYPE_LIBRARIES})

target_precompile_headers(AeonGUI
  PUBLIC
    ${AEONGUI_HEADERS}
    ${AEONGUI_DOM_HEADERS}
  PRIVATE
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

if(MINGW)
  fix_compile_commands(AeonGUI)
endif()
