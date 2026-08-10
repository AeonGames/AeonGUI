include_directories(${CMAKE_CURRENT_BINARY_DIR} ${CAIRO_INCLUDE_DIRS})

if(NOT WIN32)
    message(STATUS "Direct3D12 demo is only available on Windows - skipping Direct3D12Demo")
    return()
endif()

check_include_file_cxx("d3d12.h" HAS_D3D12_H)
check_include_file_cxx("dxgi1_6.h" HAS_DXGI1_6_H)
check_include_file_cxx("d3dcompiler.h" HAS_D3DCOMPILER_H)

if(NOT HAS_D3D12_H OR NOT HAS_DXGI1_6_H OR NOT HAS_D3DCOMPILER_H)
    message(STATUS "D3D12 headers not found - skipping Direct3D12Demo")
    return()
endif()

add_executable(Direct3D12Demo WindowsDirect3D12.cpp)

set_target_properties(Direct3D12Demo PROPERTIES
    AEONGUI_SVG_FILES "${PROJECT_SOURCE_DIR}/images/text.svg;${PROJECT_SOURCE_DIR}/images/textpath.svg;${PROJECT_SOURCE_DIR}/images/drawing.svg;${PROJECT_SOURCE_DIR}/images/tiger-style.svg;${PROJECT_SOURCE_DIR}/images/gradient.svg;${PROJECT_SOURCE_DIR}/images/curve.svg;${PROJECT_SOURCE_DIR}/images/arc.svg;${PROJECT_SOURCE_DIR}/images/line.svg;${PROJECT_SOURCE_DIR}/images/rasterimage.svg;${PROJECT_SOURCE_DIR}/images/button.svg;${PROJECT_SOURCE_DIR}/images/hover-demo.svg;${PROJECT_SOURCE_DIR}/images/smil-demo.svg;${PROJECT_SOURCE_DIR}/images/forms.xhtml;${PROJECT_SOURCE_DIR}/images/fps.xhtml"
    COMPILE_FLAGS "-DNOMINMAX -D_CRT_SECURE_NO_WARNINGS"
)

target_link_libraries(Direct3D12Demo AeonGUI d3d12 dxgi d3dcompiler)

# Compile images/fps.xhtml into a C++ document the demo can host directly
# (xmlcxx). Run Direct3D12Demo with the literal argument "compiled-fps" to load
# the generated FpsDocument instead of parsing an SVG/XHTML file at runtime.
if(TARGET xmlcxx)
    include(xmlcxx)
    xmlcxx_generate(FPS_DOCUMENT_SRCS
        "${PROJECT_SOURCE_DIR}/images/fps.xhtml"
        CLASS FpsDocument)
    target_sources(Direct3D12Demo PRIVATE ${FPS_DOCUMENT_SRCS})
    target_include_directories(Direct3D12Demo PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
    target_compile_definitions(Direct3D12Demo PRIVATE AEONGUI_HAVE_COMPILED_FPS)
endif()

if(MINGW)
    fix_compile_commands(Direct3D12Demo)
endif()
