@echo off
echo static const char* %~n1 = > %1.h
for /F "tokens=*" %%A in (%1) do echo "%%A\\n" >> %1.h
echo ; >> %1.h

