find_package(BISON)
find_package(FLEX)

set(AEONGUI_HEADERS
    ../include/aeongui/AeonGUI.hpp
    ../include/aeongui/Platform.hpp
    ../include/aeongui/Rect.hpp
    ../include/aeongui/ElementFactory.hpp
    ../include/aeongui/StringLiteral.hpp
    ../include/aeongui/Canvas.hpp
    ../include/aeongui/Path.hpp
    ../include/aeongui/AABB.hpp
    ../include/aeongui/Matrix2x3.hpp
    ../include/aeongui/Transform.hpp
    ../include/aeongui/Vector2.hpp
    ../include/aeongui/DrawType.hpp
    ../include/aeongui/Color.hpp
    ../include/aeongui/RasterImage.hpp
    ../include/aeongui/Cursor.hpp
    ../include/aeongui/ResourceLoader.hpp
    ../include/aeongui/StyleSheet.hpp
    ../include/aeongui/CSSSelector.hpp
    ../include/aeongui/TextLayout.hpp
    ../include/aeongui/FontDatabase.hpp
    ../include/aeongui/PluginAPI.h
)
if(AEONGUI_BACKEND STREQUAL "Cairo")
    set(AEONGUI_BACKEND_HEADERS
        CairoCanvas.hpp
        CairoPath.hpp
        PangoTextLayout.hpp
    )
elseif(AEONGUI_BACKEND STREQUAL "Skia")
    set(AEONGUI_BACKEND_HEADERS
        SkiaCanvas.hpp
        SkiaPath.hpp
        SkiaTextLayout.hpp
        # PangoTextLayout is also used by the HTML layout/paint code
        # under the Skia backend (Pango is linked in either case).
        PangoTextLayout.hpp
    )
endif()
set(AEONGUI_DOM_HEADERS
    ../include/aeongui/dom/DOMException.hpp
    ../include/aeongui/dom/DOMMatrix.hpp
    ../include/aeongui/dom/DOMMatrixReadOnly.hpp
    ../include/aeongui/dom/DOMRect.hpp
    ../include/aeongui/dom/DOMRectReadOnly.hpp
    ../include/aeongui/dom/DOMPoint.hpp
    ../include/aeongui/dom/DOMPointReadOnly.hpp
    ../include/aeongui/dom/Window.hpp
    ../include/aeongui/dom/Location.hpp
    ../include/aeongui/dom/Document.hpp
    ../include/aeongui/dom/DOMString.hpp
    ../include/aeongui/dom/AnyType.hpp
    ../include/aeongui/dom/Event.hpp
    ../include/aeongui/dom/EventTarget.hpp
    ../include/aeongui/dom/EventListener.hpp
    ../include/aeongui/dom/UIEvent.hpp
    ../include/aeongui/dom/EventModifierInit.hpp
    ../include/aeongui/dom/KeyboardEvent.hpp
    ../include/aeongui/dom/MouseEvent.hpp
    ../include/aeongui/dom/WheelEvent.hpp
    ../include/aeongui/dom/FocusEvent.hpp
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
    ../include/aeongui/dom/SVGImageElement.hpp
    ../include/aeongui/dom/SVGTextElement.hpp
    ../include/aeongui/dom/SVGTSpanElement.hpp
    ../include/aeongui/dom/SVGTextPathElement.hpp
    ../include/aeongui/dom/SVGTextPositioningElement.hpp
    ../include/aeongui/dom/SVGTextContentElement.hpp
    ../include/aeongui/dom/SVGAnimationElement.hpp
    ../include/aeongui/dom/SVGAnimateElement.hpp
    ../include/aeongui/dom/SVGSetElement.hpp
    ../include/aeongui/dom/SVGAnimateTransformElement.hpp
    ../include/aeongui/dom/SVGAnimateMotionElement.hpp
    ../include/aeongui/dom/SVGScriptElement.hpp
    ../include/aeongui/dom/SVGFilterElement.hpp
    ../include/aeongui/dom/SVGFEDropShadowElement.hpp
    ../include/aeongui/dom/SVGLength.hpp
    ../include/aeongui/dom/SVGLengthList.hpp
    ../include/aeongui/dom/SVGAnimatedLength.hpp
    ../include/aeongui/dom/SVGAnimatedString.hpp
    ../include/aeongui/dom/SVGAnimatedPreserveAspectRatio.hpp
    ../include/aeongui/dom/SVGAnimatedLengthList.hpp
    ../include/aeongui/dom/SVGAnimatedEnumeration.hpp
    ../include/aeongui/dom/SVGNumberList.hpp
    ../include/aeongui/dom/SVGAnimatedNumberList.hpp
    ../include/aeongui/dom/Text.hpp
    ../include/aeongui/dom/USVString.hpp
)

set(AEONGUI_SOURCES
    AeonGUI.cpp
    CSSSelector.cpp
    ElementFactory.cpp
    Rect.cpp
    Transform.cpp
    Vector2.cpp
    Matrix2x3.cpp
    AABB.cpp
    Canvas.cpp
    Path.cpp
    Color.cpp
    ResourceLoader.cpp
    RasterImage.cpp
    Cursor.cpp
    StyleSheet.cpp
    Canvas.cpp
    TextLayout.cpp
    FontDatabase.cpp
)
if(AEONGUI_BACKEND STREQUAL "Cairo")
    list(APPEND AEONGUI_SOURCES
        CairoCanvas.cpp
        CairoPath.cpp
        PangoTextLayout.cpp
    )
elseif(AEONGUI_BACKEND STREQUAL "Skia")
    list(APPEND AEONGUI_SOURCES
        SkiaCanvas.cpp
        SkiaPath.cpp
        SkiaTextLayout.cpp
        # PangoTextLayout is also used by the HTML layout/paint code
        # under the Skia backend (Pango is linked in either case).
        PangoTextLayout.cpp
    )
endif()

set(AEONGUI_DOM_SOURCES
    dom/DOMMatrix.cpp
    dom/DOMMatrixReadOnly.cpp
    dom/DOMRect.cpp
    dom/DOMRectReadOnly.cpp
    dom/DOMPoint.cpp
    dom/DOMPointReadOnly.cpp
    dom/Window.cpp
    dom/Location.cpp
    dom/Document.cpp
    dom/Event.cpp
    dom/EventListener.cpp
    dom/EventTarget.cpp
    dom/UIEvent.cpp
    dom/KeyboardEvent.cpp
    dom/MouseEvent.cpp
    dom/WheelEvent.cpp
    dom/FocusEvent.cpp
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
    dom/SVGImageElement.cpp
    dom/SVGTextPositioningElement.cpp
    dom/SVGTextContentElement.cpp
    dom/SVGTextElement.cpp
    dom/SVGTSpanElement.cpp
    dom/SVGTextPathElement.cpp
    dom/SVGAnimationElement.cpp
    dom/SVGAnimateElement.cpp
    dom/SVGSetElement.cpp
    dom/SVGAnimateTransformElement.cpp
    dom/SVGAnimateMotionElement.cpp
    dom/SVGScriptElement.cpp
    dom/SVGFilterElement.cpp
    dom/SVGFEDropShadowElement.cpp
    dom/SVGLength.cpp
    dom/SVGLengthList.cpp
    dom/SVGAnimatedLength.cpp
    dom/SVGAnimatedString.cpp
    dom/SVGAnimatedPreserveAspectRatio.cpp
    dom/SVGAnimatedLengthList.cpp
    dom/SVGAnimatedEnumeration.cpp
    dom/SVGNumberList.cpp
    dom/SVGAnimatedNumberList.cpp
    dom/Text.cpp
    dom/Attribute.cpp
)

if(ENABLE_HTML)
    list(APPEND AEONGUI_DOM_HEADERS
        ../include/aeongui/dom/HTMLElement.hpp
        ../include/aeongui/dom/HTMLHtmlElement.hpp
        ../include/aeongui/dom/HTMLBodyElement.hpp
        ../include/aeongui/dom/HTMLDivElement.hpp
        ../include/aeongui/dom/HTMLSpanElement.hpp
        ../include/aeongui/dom/HTMLImageElement.hpp
        ../include/aeongui/dom/HTMLParagraphElement.hpp
    )
    list(APPEND AEONGUI_DOM_SOURCES
        dom/HTMLElement.cpp
        dom/HTMLHtmlElement.cpp
        dom/HTMLBodyElement.cpp
        dom/HTMLDivElement.cpp
        dom/HTMLSpanElement.cpp
        dom/HTMLImageElement.cpp
        dom/HTMLParagraphElement.cpp
    )
    list(APPEND AEONGUI_HEADERS
        ../include/aeongui/HTMLLayoutEngine.hpp
    )
    list(APPEND AEONGUI_SOURCES
        HTMLLayoutEngine.cpp
    )
endif()

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

add_library(AeonGUI SHARED ${AEONGUI_HEADERS} ${AEONGUI_BACKEND_HEADERS} ${AEONGUI_SOURCES} ${AEONGUI_DOM_HEADERS} ${AEONGUI_DOM_SOURCES} ${AEONGUI_RESOURCES} ${BISON_path_data_parser_OUTPUTS} ${FLEX_path_data_lexer_OUTPUTS})
target_include_directories(AeonGUI
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE
        ${PANGO_INCLUDE_DIRS}
        ${HARFBUZZ_INCLUDE_DIRS}
        ${FONTCONFIG_INCLUDE_DIRS}
        ${FREETYPE_INCLUDE_DIR_freetype2}
        ${FREETYPE_INCLUDE_DIR_ft2build}
        ${LIBXML2_INCLUDE_DIRS}
)
if(AEONGUI_BACKEND STREQUAL "Cairo")
    foreach(_dir IN LISTS CAIRO_INCLUDE_DIRS)
        target_include_directories(AeonGUI SYSTEM PRIVATE ${_dir})
    endforeach()
elseif(AEONGUI_BACKEND STREQUAL "Skia")
    # Skia include directories are added below per-target
endif()
set_target_properties(AeonGUI PROPERTIES
    COMPILE_FLAGS "-DNOMINMAX -D_CRT_SECURE_NO_WARNINGS"
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)
if(AEONGUI_BACKEND STREQUAL "Cairo")
    target_link_libraries(AeonGUI PRIVATE libcss ${pkgcfg_lib_CAIRO_cairo} ${pkgcfg_lib_PANGO_pango-1.0} ${pkgcfg_lib_PANGO_pangocairo-1.0} ${pkgcfg_lib_PANGO_pangoft2-1.0} ${pkgcfg_lib_PANGO_gobject-2.0} ${pkgcfg_lib_HARFBUZZ_harfbuzz} ${pkgcfg_lib_FONTCONFIG_fontconfig} ${LIBXML2_LIBRARIES} ${FREETYPE_LIBRARIES})
elseif(AEONGUI_BACKEND STREQUAL "Skia")
    if(APPLE AND AEONGUI_SKIA_EFFECTIVE_ROOT)
        target_include_directories(AeonGUI SYSTEM PRIVATE ${AEONGUI_SKIA_EFFECTIVE_ROOT})
    endif()
    if(TARGET unofficial::skia::skia)
        get_target_property(SKIA_VCPKG_INCLUDE_DIRS unofficial::skia::skia INTERFACE_INCLUDE_DIRECTORIES)
        foreach(_dir IN LISTS SKIA_VCPKG_INCLUDE_DIRS)
            target_include_directories(AeonGUI SYSTEM PRIVATE ${_dir})
        endforeach()
        target_link_libraries(AeonGUI PRIVATE libcss unofficial::skia::skia ${pkgcfg_lib_PANGO_pango-1.0} ${pkgcfg_lib_PANGO_pangoft2-1.0} ${pkgcfg_lib_PANGO_gobject-2.0} ${pkgcfg_lib_HARFBUZZ_harfbuzz} ${pkgcfg_lib_FONTCONFIG_fontconfig} ${LIBXML2_LIBRARIES} ${FREETYPE_LIBRARIES})
        if(MSVC)
            target_link_libraries(AeonGUI PRIVATE FontSub Usp10)
        endif()
    else()
        target_include_directories(AeonGUI SYSTEM PRIVATE ${SKIA_INCLUDE_DIRS})
        # Some Skia distributions (e.g. MSYS2) place headers under
        # <prefix>/include/skia/include/{core,effects,...} while pkg-config
        # only reports <prefix>/include/skia as the include dir. Add the
        # nested include/ dir so that bare <core/Sk*.h> includes resolve.
        if(SKIA_INCLUDEDIR AND EXISTS "${SKIA_INCLUDEDIR}/include")
            target_include_directories(AeonGUI SYSTEM PRIVATE "${SKIA_INCLUDEDIR}/include")
        endif()
        target_link_directories(AeonGUI PRIVATE ${SKIA_LIBRARY_DIRS})
        if(APPLE AND AEONGUI_SKIA_EFFECTIVE_ROOT)
            target_link_directories(AeonGUI PRIVATE ${AEONGUI_SKIA_EFFECTIVE_ROOT}/lib)
        endif()
        target_link_libraries(AeonGUI PRIVATE libcss ${SKIA_LIBRARIES} ${pkgcfg_lib_PANGO_pango-1.0} ${pkgcfg_lib_PANGO_pangoft2-1.0} ${pkgcfg_lib_PANGO_gobject-2.0} ${pkgcfg_lib_HARFBUZZ_harfbuzz} ${pkgcfg_lib_FONTCONFIG_fontconfig} ${LIBXML2_LIBRARIES} ${FREETYPE_LIBRARIES})
    endif()
    if(APPLE)
        find_library(CORETEXT_FRAMEWORK CoreText)
        find_library(COREFOUNDATION_FRAMEWORK CoreFoundation)
        find_library(COREGRAPHICS_FRAMEWORK CoreGraphics)
        find_library(APPLICATIONSERVICES_FRAMEWORK ApplicationServices)
        target_link_libraries(AeonGUI PRIVATE ${CORETEXT_FRAMEWORK} ${COREFOUNDATION_FRAMEWORK} ${COREGRAPHICS_FRAMEWORK} ${APPLICATIONSERVICES_FRAMEWORK})
    endif()
    target_compile_definitions(AeonGUI PUBLIC AEONGUI_USE_SKIA=1)
endif()

if(ENABLE_HTML)
  target_compile_definitions(AeonGUI PUBLIC AEONGUI_ENABLE_HTML=1)
  target_link_libraries(AeonGUI PRIVATE yogacore)
endif()

if(USE_PNG AND PNG_FOUND)
  target_compile_definitions(AeonGUI PRIVATE USE_PNG=1)
  target_link_libraries(AeonGUI PRIVATE PNG::PNG)
  if(MSVC)
    set_source_files_properties(
        RasterImage.cpp
        PROPERTIES
        SKIP_PRECOMPILE_HEADERS ON
        COMPILE_OPTIONS -Drestrict=
    )
  endif()
endif()

if(USE_JPEG AND JPEG_FOUND)
  target_compile_definitions(AeonGUI PRIVATE USE_JPEG=1)
  target_link_libraries(AeonGUI PRIVATE JPEG::JPEG)
endif()

# Build angle-bracket include list for PUBLIC precompiled headers.
# Using angle-bracket includes instead of file paths ensures the exported
# cmake config does not contain absolute build-tree paths (fixes vcpkg).
set(AEONGUI_PCH_PUBLIC_HEADERS "")
foreach(_hdr IN LISTS AEONGUI_HEADERS AEONGUI_DOM_HEADERS)
    string(REGEX REPLACE "^\\.\\./include/" "" _rel "${_hdr}")
    list(APPEND AEONGUI_PCH_PUBLIC_HEADERS "<${_rel}>")
endforeach()

# Build quoted-path include list for PRIVATE backend precompiled headers.
set(AEONGUI_PCH_BACKEND_HEADERS "")
foreach(_hdr IN LISTS AEONGUI_BACKEND_HEADERS)
    list(APPEND AEONGUI_PCH_BACKEND_HEADERS "\"${_hdr}\"")
endforeach()

if(AEONGUI_BACKEND STREQUAL "Cairo")
    target_precompile_headers(AeonGUI
      PUBLIC
        ${AEONGUI_PCH_PUBLIC_HEADERS}
      PRIVATE
        ${AEONGUI_PCH_BACKEND_HEADERS}
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
elseif(AEONGUI_BACKEND STREQUAL "Skia")
    target_precompile_headers(AeonGUI
      PUBLIC
        ${AEONGUI_PCH_PUBLIC_HEADERS}
      PRIVATE
        ${AEONGUI_PCH_BACKEND_HEADERS}
        <include/core/SkCanvas.h>
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
endif()

if(MINGW)
  fix_compile_commands(AeonGUI)
endif()
