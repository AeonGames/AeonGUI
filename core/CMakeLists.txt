
find_package(BISON)
find_package(FLEX)

set(AEONGUI_HEADERS
		../include/aeongui/Platform.h
        ../include/aeongui/Rect.h
        ../include/aeongui/Element.h
        ../include/aeongui/ElementFactory.h
        ../include/aeongui/StringLiteral.h
		../include/aeongui/Document.h
		../include/aeongui/Window.h
		../include/aeongui/Canvas.h
		../include/aeongui/CairoCanvas.h
		../include/aeongui/AABB.h
		../include/aeongui/Matrix2x3.h
		../include/aeongui/Transform.h
		../include/aeongui/Vector2.h
		../include/aeongui/DrawType.h
		../include/aeongui/JavaScript.h
		../include/aeongui/Duktape.h
		../include/aeongui/Color.h
)

set(AEONGUI_SOURCES
        ElementFactory.cpp
        Window.cpp
        Document.cpp
        Rect.cpp
        Element.cpp
		Transform.cpp
        Vector2.cpp
        Matrix2x3.cpp
        AABB.cpp
        Canvas.cpp
        CairoCanvas.cpp
        JavaScript.cpp
        Duktape.cpp
        Color.cpp
        elements/SVG.cpp
        elements/G.cpp
        elements/Path.cpp
        elements/Rect.cpp
        elements/Line.cpp
        elements/Polyline.cpp
        elements/Polygon.cpp
        elements/Circle.cpp
        elements/Ellipse.cpp
        elements/Script.cpp
        elements/SVG.h
        elements/G.h
        elements/Path.h
        elements/Rect.h
        elements/Line.h
        elements/Polyline.h
        elements/Polygon.h
        elements/Circle.h
        elements/Ellipse.h
        elements/Script.h
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

#add_custom_command(
#	OUTPUT
#		${CMAKE_SOURCE_DIR}/open-sans/OpenSans_Bold_ttf.h
#		${CMAKE_SOURCE_DIR}/open-sans/OpenSans_Light_ttf.h
#		${CMAKE_SOURCE_DIR}/open-sans/OpenSans_BoldItalic_ttf.h
#		${CMAKE_SOURCE_DIR}/open-sans/OpenSans_LightItalic_ttf.h
#		${CMAKE_SOURCE_DIR}/open-sans/OpenSans_ExtraBold_ttf.h
#		${CMAKE_SOURCE_DIR}/open-sans/OpenSans_Regular_ttf.h
#		${CMAKE_SOURCE_DIR}/open-sans/OpenSans_ExtraBoldItalic_ttf.h
#		${CMAKE_SOURCE_DIR}/open-sans/OpenSans_Semibold_ttf.h
#		${CMAKE_SOURCE_DIR}/open-sans/OpenSans_Italic_ttf.h
#		${CMAKE_SOURCE_DIR}/open-sans/OpenSans_SemiboldItalic_ttf.h
#	COMMAND ${XXD_EXECUTABLE} ARGS -i OpenSans-Bold.ttf > OpenSans_Bold_ttf.h
#	COMMAND ${XXD_EXECUTABLE} ARGS -i OpenSans-Light.ttf > OpenSans_Light_ttf.h
#	COMMAND ${XXD_EXECUTABLE} ARGS -i OpenSans-BoldItalic.ttf > OpenSans_BoldItalic_ttf.h
#	COMMAND ${XXD_EXECUTABLE} ARGS -i OpenSans-LightItalic.ttf > OpenSans_LightItalic_ttf.h
#	COMMAND ${XXD_EXECUTABLE} ARGS -i OpenSans-ExtraBold.ttf > OpenSans_ExtraBold_ttf.h
#	COMMAND ${XXD_EXECUTABLE} ARGS -i OpenSans-Regular.ttf > OpenSans_Regular_ttf.h
#	COMMAND ${XXD_EXECUTABLE} ARGS -i OpenSans-ExtraBoldItalic.ttf > OpenSans_ExtraBoldItalic_ttf.h
#	COMMAND ${XXD_EXECUTABLE} ARGS -i OpenSans-Semibold.ttf > OpenSans_Semibold_ttf.h
#	COMMAND ${XXD_EXECUTABLE} ARGS -i OpenSans-Italic.ttf > OpenSans_Italic_ttf.h
#	COMMAND ${XXD_EXECUTABLE} ARGS -i OpenSans-SemiboldItalic.ttf > OpenSans_SemiboldItalic_ttf.h
#	DEPENDS ${XXD_EXECUTABLE}
#	WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/open-sans
#	COMMENT "Generating Open Sans font arrays" VERBATIM)

#include_directories(${CMAKE_SOURCE_DIR}/open-sans)

#set(AEONGUI_RESOURCES
#	${CMAKE_SOURCE_DIR}/open-sans/OpenSans_Bold_ttf.h
#	${CMAKE_SOURCE_DIR}/open-sans/OpenSans_Light_ttf.h
#	${CMAKE_SOURCE_DIR}/open-sans/OpenSans_BoldItalic_ttf.h
#	${CMAKE_SOURCE_DIR}/open-sans/OpenSans_LightItalic_ttf.h
#	${CMAKE_SOURCE_DIR}/open-sans/OpenSans_ExtraBold_ttf.h
#	${CMAKE_SOURCE_DIR}/open-sans/OpenSans_Regular_ttf.h
#	${CMAKE_SOURCE_DIR}/open-sans/OpenSans_ExtraBoldItalic_ttf.h
#	${CMAKE_SOURCE_DIR}/open-sans/OpenSans_Semibold_ttf.h
#	${CMAKE_SOURCE_DIR}/open-sans/OpenSans_Italic_ttf.h
#	${CMAKE_SOURCE_DIR}/open-sans/OpenSans_SemiboldItalic_ttf.h
#	resources/close.c
#	resources/maximize.c
#	resources/minimize.c
#	resources/restore.c
#	resources/scrolldown.c
#	resources/scrolldownpressed.c
#	resources/scrollup.c
#	resources/scrolluppressed.c
#	resources/scrollleft.c
#	resources/scrollleftpressed.c
#	resources/scrollright.c
#	resources/scrollrightpressed.c
#	resources/close.h
#	resources/maximize.h
#	resources/minimize.h
#	resources/restore.h
#	resources/scrolldown.h
#	resources/scrolldownpressed.h
#	resources/scrollup.h
#	resources/scrolluppressed.h
#	resources/scrollleft.h
#	resources/scrollleftpressed.h
#	resources/scrollright.h
#	resources/scrollrightpressed.h
#	resources/close_down.c
#	resources/maximize_down.c
#	resources/minimize_down.c
#	resources/restore_down.c
#	resources/close_down.h
#	resources/maximize_down.h
#	resources/minimize_down.h
#	resources/restore_down.h
#)

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

    add_custom_target(update-parser-code
        COMMAND
            ${CMAKE_COMMAND} -E copy_if_different ${BISON_path_data_parser_OUTPUTS} ${FLEX_path_data_lexer_OUTPUTS} ${CMAKE_SOURCE_DIR}/core/parsers
        COMMAND
            ${CMAKE_COMMAND} -E copy_if_different ${BISON_style_parser_OUTPUTS} ${FLEX_style_lexer_OUTPUTS} ${CMAKE_SOURCE_DIR}/core/parsers
        DEPENDS
            path_data_parser
            path_data_lexer
        COMMENT "Updating pre-generated parser code.")
else()
    message(STATUS "Using pre-generated parser code.")
    message(STATUS "Install Flex and Bison if you want to regenerate the parser code at build time.")
    list(APPEND AEONGUI_SOURCES
        parsers/path_data_lexer.cpp
        parsers/path_data_parser.cpp
        parsers/path_data_parser.hpp
        parsers/dstype.h
        parsers/style_lexer.cpp
        parsers/style_parser.cpp
        parsers/style_parser.hpp
        )
endif()

set(AEONGUI_LIBRARIES ${LIBXML2_LIBRARIES} pcx)
if(USE_PNG)
	set(AEONGUI_LIBRARIES ${AEONGUI_LIBRARIES} ${PNG_LIBRARIES} ${ZLIB_LIBRARIES})
endif(USE_PNG)
source_group("Resource Files" FILES ${AEONGUI_RESOURCES})
include_directories(${CAIRO_INCLUDE_DIRS} ${DUKTAPE_INCLUDE_DIR} ${FREETYPE_INCLUDE_DIR_freetype2} ${FREETYPE_INCLUDE_DIR_ft2build})
add_library(AeonGUI SHARED ${AEONGUI_HEADERS} ${AEONGUI_SOURCES} ${AEONGUI_RESOURCES})
add_dependencies(AeonGUI pcx)
set_target_properties(AeonGUI PROPERTIES COMPILE_FLAGS "-DNOMINMAX -D_CRT_SECURE_NO_WARNINGS")
target_link_libraries(AeonGUI ${AEONGUI_LIBRARIES} ${CAIRO_LIBRARIES} ${DUKTAPE_LIBRARY} ${LIBXML2_LIBRARIES} ${FREETYPE_LIBRARIES})

if(BUILD_UNIT_TESTS)
	subdirs(tests)
endif()
