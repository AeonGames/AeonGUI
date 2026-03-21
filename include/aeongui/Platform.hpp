/*
Copyright (C) 2014,2019,2025,2026 Rodrigo Jose Hernandez Cordoba

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

/** @file Platform.hpp
 *  @brief Platform-specific DLL import/export macros and compiler helpers.
 */
#ifndef AEONGUI_PLATFORM_H
#define AEONGUI_PLATFORM_H
#ifndef DLL
#ifdef WIN32
#ifdef AeonGUI_EXPORTS
#define DLL __declspec( dllexport )
#else
#define DLL __declspec( dllimport )
#endif
#else
#define DLL
#endif
#endif
#endif

/** @def PRIVATE_TEMPLATE_MEMBERS_START
 *  @brief Suppress MSVC C4251 warnings for private template data members.
 */
/** @def PRIVATE_TEMPLATE_MEMBERS_END
 *  @brief Restore warnings after PRIVATE_TEMPLATE_MEMBERS_START.
 */
#ifdef _MSC_VER
#define PRIVATE_TEMPLATE_MEMBERS_START \
_Pragma("warning(push)") \
_Pragma("warning(disable: 4251)")
#define PRIVATE_TEMPLATE_MEMBERS_END \
_Pragma("warning(pop)")
#else
#define PRIVATE_TEMPLATE_MEMBERS_START
#define PRIVATE_TEMPLATE_MEMBERS_END
#endif
