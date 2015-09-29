#!/bin/bash
astyle --options=astylerc "core/*.cpp" "demos/*.cpp" "demos/*.h" "include/*.h" "renderers/OpenGL/*.cpp" "renderers/OpenGL/*.h" "common/pcx/*.h" "common/pcx/*.cpp" "core/cuda/*.cu"
