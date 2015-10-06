#ifndef AEONGUI_LOG_H
#define AEONGUI_LOG_H
/******************************************************************************
Copyright 2015 Rodrigo Hernandez Cordoba

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
******************************************************************************/

#ifdef ANDROID
#include <android/log.h>
#else
#ifdef WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN 1
#endif
#ifndef NOMINMAX
#define NOMINMAX 1
#endif
#include <windows.h>
#endif
#ifdef __cplusplus
#include <cstdio>
#include <cstdarg>
#else
#include <stdio.h>
#include <stdarg.h>
#endif
#endif

#ifdef __cplusplus
#include <cassert>
#else
#include <assert.h>
#endif

///\todo Add __FILE__ __LINE__ and log type support.
#if defined(_DEBUG)

#define AEONGUI_LOG_INFO( fmt, ... )  AEONGUI::log(AEONGUI::LOG_INFO, fmt, ##__VA_ARGS__ )
#define AEONGUI_LOG_WARN( fmt, ... )  AEONGUI::log(AEONGUI::LOG_WARN, fmt, ##__VA_ARGS__ )
#define AEONGUI_LOG_ERROR( fmt, ... )  AEONGUI::log(AEONGUI::LOG_ERROR, fmt, ##__VA_ARGS__ )

namespace AEONGUI
{
    enum LogTypes
    {
#ifdef ANDROID
        LOG_INFO = ANDROID_LOG_INFO,
        LOG_WARN = ANDROID_LOG_WARN,
        LOG_ERROR = ANDROID_LOG_ERROR,
#else
        LOG_INFO = 0,
        LOG_WARN,
        LOG_ERROR,
#endif
    };

#ifdef ANDROID
    inline void log ( int level, const char* fmt, ... )
    {
        va_list args;
        va_start ( args, fmt );
        __android_log_vprint ( level, "AeonEngine", fmt, args );
        va_end ( args );
    }
#else
    inline void log ( int level, const char* fmt, ... )
    {
#ifdef WIN32
        CONSOLE_SCREEN_BUFFER_INFO csbInfo;
        HANDLE console_handle = GetStdHandle ( STD_OUTPUT_HANDLE );
        GetConsoleScreenBufferInfo ( console_handle, &csbInfo );
#endif
        va_list args;
        va_start ( args, fmt );
        switch ( level )
        {
        case LOG_INFO:
#ifndef WIN32
            printf ( "\x1B[32m%s\x1B[m ", "Info:" );
#else
            SetConsoleTextAttribute ( console_handle, FOREGROUND_GREEN | FOREGROUND_INTENSITY );
            printf ( "%s", "Info: " );
#endif
            break;
        case LOG_WARN:
#ifndef WIN32
            printf ( "\x1B[33m%s\x1B[m ", "Warning:" );
#else
            SetConsoleTextAttribute ( console_handle, FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_INTENSITY );
            printf ( "%s", "Warning: " );
#endif
            break;
        case LOG_ERROR:
#ifndef WIN32
            printf ( "\x1B[31m%s\x1B[m ", "Error:" );
#else
            SetConsoleTextAttribute ( console_handle, FOREGROUND_RED | FOREGROUND_INTENSITY );
            printf ( "%s", "Error: " );
#endif
            break;
        default:
            break;
        }
#ifdef WIN32
        SetConsoleTextAttribute ( console_handle, csbInfo.wAttributes );
#endif
        vprintf ( fmt, args );
        printf ( "\n" );
        fflush ( stdout );
        va_end ( args );
#ifdef WIN32
        if ( level == LOG_ERROR )
        {
            char buffer[1024];
            va_start ( args, fmt );
            vsprintf ( buffer, fmt, args );
            MessageBox ( NULL, buffer, TEXT ( "AEONGUI Error" ), MB_OK | MB_ICONERROR );
            va_end ( args );
        }
#endif
    }
#endif
};

#if defined(__gl_h_) || defined(__gl2_h_) || defined(__glcorearb_h_)
static int glError = 0;
#define AEONGUI_OPENGL_CHECK_ERROR()\
    if ( ( glError = glGetError() ) != GL_NO_ERROR ) \
    { \
        AEONGUI_LOG_ERROR ( "OpenGL Error %s (0x%x) Line %d at %s", \
        (glError==GL_INVALID_ENUM) ? "GL_INVALID_ENUM" : \
        (glError==GL_INVALID_VALUE) ? "GL_INVALID_VALUE" : \
        (glError==GL_INVALID_OPERATION) ? "GL_INVALID_OPERATION" : \
        (glError==GL_STACK_OVERFLOW) ? "GL_STACK_OVERFLOW" : \
        (glError==GL_STACK_UNDERFLOW) ? "GL_STACK_UNDERFLOW" : \
        (glError==GL_OUT_OF_MEMORY) ? "GL_OUT_OF_MEMORY" : \
        "Unknown Error Code", \
        glError, __LINE__, __FILE__ ); \
    }
#endif

#else
#define AEONGUI_LOG_INFO( fmt, ... )
#define AEONGUI_LOG_WARN( fmt, ... )
#define AEONGUI_LOG_ERROR( fmt, ... )
#define AEONGUI_OPENGL_CHECK_ERROR()
#endif
#endif
