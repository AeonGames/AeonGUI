@echo off
rem Astyle.exe can be found here: http://sourceforge.net/projects/astyle/files/
Astyle.exe --options=astylerc demos\WindowsOpenGL\*.cpp include\*.h renderers\OpenGL\*.cpp renderers\OpenGL\*.h
