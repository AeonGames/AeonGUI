#ifndef AEONGUI_PLATFORM_H
#define AEONGUI_PLATFORM_H
#ifndef DLL
#ifdef WIN32
#ifdef DLL_EXPORT
#define DLL __declspec( dllexport )
#else
#define DLL __declspec( dllimport )
#endif
#else
#define DLL
#endif
#endif
#endif
