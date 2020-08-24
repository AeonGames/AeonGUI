
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
    ../include/aeongui/V8.h
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
    V8.cpp
    Color.cpp
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

if(FLEX_FOUND AND BISON_FOUND)
    message(STATUS "Parser code will be generated from Flex and Bison code.")
    message(STATUS "Build the 'update-parser-code' target if you want to update the pre-generated code.")

    bison_target(path_data_parser ${CMAKE_CURRENT_SOURCE_DIR}/parsers/path_data.ypp ${CMAKE_CURRENT_BINARY_DIR}/path_data_parser.cpp COMPILE_FLAGS "-v -d")
    flex_target(path_data_lexer ${CMAKE_CURRENT_SOURCE_DIR}/parsers/path_data.l ${CMAKE_CURRENT_BINARY_DIR}/path_data_lexer.cpp COMPILE_FLAGS "-Cf")
    add_flex_bison_dependency(path_data_lexer path_data_parser)
    list(APPEND AEONGUI_SOURCES ${BISON_path_data_parser_OUTPUTS} ${FLEX_path_data_lexer_OUTPUTS} parsers/dstype.h)

    bison_target(style_parser ${CMAKE_CURRENT_SOURCE_DIR}/parsers/style.ypp ${CMAKE_CURRENT_BINARY_DIR}/style_parser.cpp COMPILE_FLAGS "-v -d")
    flex_target(style_lexer ${CMAKE_CURRENT_SOURCE_DIR}/parsers/style.l ${CMAKE_CURRENT_BINARY_DIR}/style_lexer.cpp COMPILE_FLAGS "-Cf")
    add_flex_bison_dependency(style_lexer style_parser)
    list(APPEND AEONGUI_SOURCES ${BISON_style_parser_OUTPUTS} ${FLEX_style_lexer_OUTPUTS})

    if(MSVC)
        set_source_files_properties(${FLEX_style_lexer_OUTPUTS} ${FLEX_path_data_lexer_OUTPUTS} PROPERTIES COMPILE_FLAGS -FIcstdint)
    endif()

    add_custom_target(update-parser-code
        COMMAND
            ${CMAKE_COMMAND} -E copy ${BISON_path_data_parser_OUTPUTS} ${FLEX_path_data_lexer_OUTPUTS} ${CMAKE_SOURCE_DIR}/core/parsers
        COMMAND
            ${CMAKE_COMMAND} -E copy ${BISON_style_parser_OUTPUTS} ${FLEX_style_lexer_OUTPUTS} ${CMAKE_SOURCE_DIR}/core/parsers
        DEPENDS
            path_data_parser
            path_data_lexer
            style_parser
            style_lexer
        COMMENT "Updating pre-generated parser code.")
else()
    message(STATUS "Using pre-generated parser code.")
    message(STATUS "Install Flex and Bison if you want to regenerate the parser code at build time.")
    set(PARSER_HEADERS
        parsers/path_data_parser.hpp
        parsers/dstype.h
        parsers/style_parser.hpp
    )
    set(PARSER_SOURCES
        parsers/path_data_lexer.cpp
        parsers/path_data_parser.cpp
        parsers/style_lexer.cpp
        parsers/style_parser.cpp
    )
    list(APPEND AEONGUI_SOURCES ${PARSER_SOURCES} ${PARSER_HEADERS})
    if(MSVC)
        set_source_files_properties(${PARSER_SOURCES} PROPERTIES COMPILE_FLAGS -FIcstdint)
    endif()
endif()

include_directories(${CAIRO_INCLUDE_DIRS} ${FREETYPE_INCLUDE_DIR_freetype2} ${FREETYPE_INCLUDE_DIR_ft2build} ${V8_INCLUDE_DIRS})
add_library(AeonGUI SHARED ${AEONGUI_HEADERS} ${AEONGUI_SOURCES} ${AEONGUI_RESOURCES})
set_target_properties(AeonGUI PROPERTIES COMPILE_FLAGS "-DNOMINMAX -D_CRT_SECURE_NO_WARNINGS")
target_link_libraries(AeonGUI PUBLIC ${CAIRO_LIBRARIES} ${LIBXML2_LIBRARIES} ${FREETYPE_LIBRARIES} ${V8_LIBRARIES})

fix_compile_commands(AeonGUI)

if(BUILD_UNIT_TESTS)
	subdirs(tests)
endif()
