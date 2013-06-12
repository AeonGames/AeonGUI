#!/bin/bash
astyle --options=astylerc "core/*.cpp" "demos/WindowsOpenGL/*.cpp" "demos/LinuxOpenGL/*.cpp" "include/*.h" "renderers/OpenGL/*.cpp" "renderers/OpenGL/*.h" "common/pcx/*.h" "common/pcx/*.cpp"
