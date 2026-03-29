if(APPLE)
    include_directories(${CMAKE_CURRENT_BINARY_DIR} ${CAIRO_INCLUDE_DIRS})

    # ── Compile .metal → .metallib ────────────────────────────────────────
    find_program(XCRUN_EXECUTABLE xcrun)
    set(METAL_SHADER_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/Shaders.metal")
    set(METAL_AIR           "${CMAKE_CURRENT_BINARY_DIR}/Shaders.air")
    set(METAL_LIB           "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Shaders.metallib")

    if(XCRUN_EXECUTABLE)
        add_custom_command(
            OUTPUT  ${METAL_AIR}
            COMMAND ${XCRUN_EXECUTABLE} -sdk macosx metal -c ${METAL_SHADER_SOURCE} -o ${METAL_AIR}
            DEPENDS ${METAL_SHADER_SOURCE}
            COMMENT "Compiling Metal shader → AIR"
        )
        add_custom_command(
            OUTPUT  ${METAL_LIB}
            COMMAND ${XCRUN_EXECUTABLE} -sdk macosx metallib ${METAL_AIR} -o ${METAL_LIB}
            DEPENDS ${METAL_AIR}
            COMMENT "Linking Metal shader library"
        )
        add_custom_target(MetalShaders DEPENDS ${METAL_LIB})
    else()
        # Fallback: copy the .metal source next to the binary so it can be
        # compiled at runtime by the application.
        add_custom_command(
            OUTPUT  "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Shaders.metal"
            COMMAND ${CMAKE_COMMAND} -E copy ${METAL_SHADER_SOURCE}
                    "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Shaders.metal"
            DEPENDS ${METAL_SHADER_SOURCE}
            COMMENT "Copying Metal shader source"
        )
        add_custom_target(MetalShaders DEPENDS "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Shaders.metal")
    endif()

    # ── Executable ────────────────────────────────────────────────────────
    add_executable(MetalDemo MacOSMetal.mm)
    set_source_files_properties(MacOSMetal.mm PROPERTIES COMPILE_FLAGS "-x objective-c++")
    set_target_properties(MetalDemo PROPERTIES DISABLE_PRECOMPILE_HEADERS ON)
    set_target_properties(MetalDemo PROPERTIES
        AEONGUI_SVG_FILES "${CMAKE_SOURCE_DIR}/images/text.svg;${CMAKE_SOURCE_DIR}/images/textpath.svg;${CMAKE_SOURCE_DIR}/images/drawing.svg;${CMAKE_SOURCE_DIR}/images/tiger-style.svg;${CMAKE_SOURCE_DIR}/images/gradient.svg;${CMAKE_SOURCE_DIR}/images/curve.svg;${CMAKE_SOURCE_DIR}/images/arc.svg;${CMAKE_SOURCE_DIR}/images/line.svg;${CMAKE_SOURCE_DIR}/images/rasterimage.svg;${CMAKE_SOURCE_DIR}/images/button.svg;${CMAKE_SOURCE_DIR}/images/hover-demo.svg"
        COMPILE_FLAGS "-DNOMINMAX -D_CRT_SECURE_NO_WARNINGS"
    )

    add_dependencies(MetalDemo MetalShaders)
    target_link_libraries(MetalDemo AeonGUI "-framework Cocoa" "-framework Metal" "-framework MetalKit" "-framework QuartzCore")
endif()
