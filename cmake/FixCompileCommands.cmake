# Rewrite MinGW-style /<drive>/foo paths in compile_commands.json to
# Windows-style <drive>:/foo so tools like clangd / VS Code can resolve them.
#
# This script is intended to be invoked via `cmake -P` from a custom command,
# avoiding the shell-quoting and MSYS2 path-mangling issues that plague the
# equivalent `sed -i 's/.../.../'` invocation on Windows.
if(NOT DEFINED COMPILE_COMMANDS_JSON)
  message(FATAL_ERROR "FixCompileCommands: COMPILE_COMMANDS_JSON is not set")
endif()
if(NOT EXISTS "${COMPILE_COMMANDS_JSON}")
  return()
endif()

file(READ "${COMPILE_COMMANDS_JSON}" _content)
# /<letter>/  ->  <letter>:/
string(REGEX REPLACE "/([A-Za-z])/" "\\1:/" _content "${_content}")
file(WRITE "${COMPILE_COMMANDS_JSON}" "${_content}")
