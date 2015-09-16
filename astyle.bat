@echo off
rem Astyle.exe can be found here: http://sourceforge.net/projects/astyle/files/
Astyle.exe --options=astylerc core\*.cpp demos\*.cpp include\*.h renderers\*.cpp renderers\*.h common\*.h common\*.cpp core\*.cu
