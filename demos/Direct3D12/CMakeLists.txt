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
    AEONGUI_SVG_FILES "${CMAKE_SOURCE_DIR}/images/text.svg;${CMAKE_SOURCE_DIR}/images/textpath.svg;${CMAKE_SOURCE_DIR}/images/drawing.svg;${CMAKE_SOURCE_DIR}/images/tiger-style.svg;${CMAKE_SOURCE_DIR}/images/gradient.svg;${CMAKE_SOURCE_DIR}/images/curve.svg;${CMAKE_SOURCE_DIR}/images/arc.svg;${CMAKE_SOURCE_DIR}/images/line.svg;${CMAKE_SOURCE_DIR}/images/rasterimage.svg"
    COMPILE_FLAGS "-DNOMINMAX -D_CRT_SECURE_NO_WARNINGS"
)

target_link_libraries(Direct3D12Demo AeonGUI d3d12 dxgi d3dcompiler)

if(MINGW)
    fix_compile_commands(Direct3D12Demo)
endif()
