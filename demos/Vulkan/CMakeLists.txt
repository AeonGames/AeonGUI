include_directories(${CMAKE_CURRENT_BINARY_DIR} ${CAIRO_INCLUDE_DIRS})

find_package(Vulkan)
if(NOT Vulkan_FOUND)
    message(STATUS "Vulkan SDK not found – skipping VulkanDemo")
    return()
endif()

find_program(GLSLANG_VALIDATOR NAMES glslangValidator HINTS
    $ENV{VULKAN_SDK}/bin
    $ENV{VULKAN_SDK}/Bin)

if(NOT GLSLANG_VALIDATOR)
    # Try glslc (shaderc) as a fallback
    find_program(GLSLC_EXECUTABLE glslc HINTS
        $ENV{VULKAN_SDK}/bin
        $ENV{VULKAN_SDK}/Bin)
endif()

# ── Compile GLSL → SPIR-V ────────────────────────────────────────────────
set(VERT_SHADER "${CMAKE_CURRENT_SOURCE_DIR}/shaders/screen.vert")
set(FRAG_SHADER "${CMAKE_CURRENT_SOURCE_DIR}/shaders/screen.frag")
set(VERT_SPV    "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/screen.vert.spv")
set(FRAG_SPV    "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/screen.frag.spv")

if(GLSLANG_VALIDATOR)
    add_custom_command(
        OUTPUT  ${VERT_SPV}
        COMMAND ${GLSLANG_VALIDATOR} -V ${VERT_SHADER} -o ${VERT_SPV}
        DEPENDS ${VERT_SHADER}
        COMMENT "Compiling screen.vert → SPIR-V (glslangValidator)"
    )
    add_custom_command(
        OUTPUT  ${FRAG_SPV}
        COMMAND ${GLSLANG_VALIDATOR} -V ${FRAG_SHADER} -o ${FRAG_SPV}
        DEPENDS ${FRAG_SHADER}
        COMMENT "Compiling screen.frag → SPIR-V (glslangValidator)"
    )
elseif(GLSLC_EXECUTABLE)
    add_custom_command(
        OUTPUT  ${VERT_SPV}
        COMMAND ${GLSLC_EXECUTABLE} ${VERT_SHADER} -o ${VERT_SPV}
        DEPENDS ${VERT_SHADER}
        COMMENT "Compiling screen.vert → SPIR-V (glslc)"
    )
    add_custom_command(
        OUTPUT  ${FRAG_SPV}
        COMMAND ${GLSLC_EXECUTABLE} ${FRAG_SHADER} -o ${FRAG_SPV}
        DEPENDS ${FRAG_SHADER}
        COMMENT "Compiling screen.frag → SPIR-V (glslc)"
    )
else()
    message(WARNING "Neither glslangValidator nor glslc found – "
                    "SPIR-V shaders will not be compiled. "
                    "Please install the Vulkan SDK or provide pre-compiled .spv files.")
    return()
endif()

add_custom_target(VulkanShaders DEPENDS ${VERT_SPV} ${FRAG_SPV})

# ── Executable (platform-specific) ───────────────────────────────────────
if(WIN32)
    add_executable(VulkanDemo VulkanRenderer.h WindowsVulkan.cpp)
elseif(APPLE)
    add_executable(VulkanDemo VulkanRenderer.h MacOSVulkan.mm)
    set_source_files_properties(MacOSVulkan.mm PROPERTIES COMPILE_FLAGS "-x objective-c++")
    set_target_properties(VulkanDemo PROPERTIES DISABLE_PRECOMPILE_HEADERS ON)
else()
    find_package(X11)
    add_executable(VulkanDemo VulkanRenderer.h LinuxVulkan.cpp)
endif()

set_target_properties(VulkanDemo PROPERTIES
    AEONGUI_SVG_FILES "${PROJECT_SOURCE_DIR}/images/text.svg;${PROJECT_SOURCE_DIR}/images/textpath.svg;${PROJECT_SOURCE_DIR}/images/drawing.svg;${PROJECT_SOURCE_DIR}/images/tiger-style.svg;${PROJECT_SOURCE_DIR}/images/gradient.svg;${PROJECT_SOURCE_DIR}/images/curve.svg;${PROJECT_SOURCE_DIR}/images/arc.svg;${PROJECT_SOURCE_DIR}/images/line.svg;${PROJECT_SOURCE_DIR}/images/rasterimage.svg;${PROJECT_SOURCE_DIR}/images/button.svg;${PROJECT_SOURCE_DIR}/images/hover-demo.svg;${PROJECT_SOURCE_DIR}/images/smil-demo.svg;${PROJECT_SOURCE_DIR}/images/fps.xhtml"
    COMPILE_FLAGS "-DNOMINMAX -D_CRT_SECURE_NO_WARNINGS"
)

add_dependencies(VulkanDemo VulkanShaders)
target_include_directories(VulkanDemo PRIVATE ${Vulkan_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(VulkanDemo AeonGUI Vulkan::Vulkan)

if(APPLE)
    target_link_libraries(VulkanDemo "-framework Cocoa" "-framework QuartzCore")
elseif(UNIX)
    target_link_libraries(VulkanDemo ${X11_LIBRARIES})
endif()

# Compile images/fps.xhtml into a C++ document the demo can host directly
# (xmlcxx). Run VulkanDemo with the literal argument "compiled-fps" to load
# the generated FpsDocument instead of parsing an SVG/XHTML file at runtime.
if(TARGET xmlcxx)
    include(xmlcxx)
    xmlcxx_generate(FPS_DOCUMENT_SRCS
        "${PROJECT_SOURCE_DIR}/images/fps.xhtml"
        CLASS FpsDocument)
    target_sources(VulkanDemo PRIVATE ${FPS_DOCUMENT_SRCS})
    target_include_directories(VulkanDemo PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
    target_compile_definitions(VulkanDemo PRIVATE AEONGUI_HAVE_COMPILED_FPS)
endif()

if(MINGW)
    fix_compile_commands(VulkanDemo)
endif()
