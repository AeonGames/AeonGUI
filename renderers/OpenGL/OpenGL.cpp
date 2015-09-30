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
#define NOMINMAX
#include <cstdint>
#include <algorithm>
#include "OpenGL.h"
#include "Log.h"

#ifdef WIN32
#include "wglext.h"
#define GLGETPROCADDRESS(glFunctionType,glFunction) \
    if(glFunction==nullptr) { \
        glFunction = (glFunctionType)wglGetProcAddress(#glFunction); \
        if (glFunction == nullptr) { AEONGUI_LOG_ERROR("OpenGL: Unable to load %s function.", #glFunction); if(!contextAvailable){DestroyOpenGLContext();} return false; }}
#else
#include <GL/gl.h>
#include <GL/glx.h>
#include "glxext.h"
#define GLGETPROCADDRESS(glFunctionType,glFunction) \
    if(glFunction==nullptr) { \
    glFunction = ( glFunctionType ) glXGetProcAddress ( (const GLubyte*) #glFunction ); \
    if (glFunction==nullptr) { AEONGUI_LOG_ERROR("%s:%d OpenGL: Unable to load %s.",__FUNCTION__,__LINE__,#glFunction);return false;}}
#endif
#define GLDEFINEFUNCTION(glFunctionType,glFunction) glFunctionType glFunction = nullptr


namespace AeonGUI
{
    GLDEFINEFUNCTION ( PFNGLATTACHSHADERPROC, glAttachShader );
    GLDEFINEFUNCTION ( PFNGLCOMPILESHADERPROC, glCompileShader );
    GLDEFINEFUNCTION ( PFNGLCREATEPROGRAMPROC, glCreateProgram );
    GLDEFINEFUNCTION ( PFNGLCREATESHADERPROC, glCreateShader );
    GLDEFINEFUNCTION ( PFNGLDELETEPROGRAMPROC, glDeleteProgram );
    GLDEFINEFUNCTION ( PFNGLDELETESHADERPROC, glDeleteShader );
    GLDEFINEFUNCTION ( PFNGLENABLEVERTEXATTRIBARRAYPROC, glEnableVertexAttribArray );
    GLDEFINEFUNCTION ( PFNGLGETATTRIBLOCATIONPROC, glGetAttribLocation );
    GLDEFINEFUNCTION ( PFNGLGETPROGRAMIVPROC, glGetProgramiv );
    GLDEFINEFUNCTION ( PFNGLGETPROGRAMINFOLOGPROC, glGetProgramInfoLog );
    GLDEFINEFUNCTION ( PFNGLGETSHADERIVPROC, glGetShaderiv );
    GLDEFINEFUNCTION ( PFNGLGETSHADERINFOLOGPROC, glGetShaderInfoLog );
    GLDEFINEFUNCTION ( PFNGLGETUNIFORMLOCATIONPROC, glGetUniformLocation );
    GLDEFINEFUNCTION ( PFNGLLINKPROGRAMPROC, glLinkProgram );
    GLDEFINEFUNCTION ( PFNGLSHADERSOURCEPROC, glShaderSource );
    GLDEFINEFUNCTION ( PFNGLUSEPROGRAMPROC, glUseProgram );
    GLDEFINEFUNCTION ( PFNGLUNIFORM1IPROC, glUniform1i );
    GLDEFINEFUNCTION ( PFNGLUNIFORMMATRIX4FVPROC, glUniformMatrix4fv );
    GLDEFINEFUNCTION ( PFNGLVERTEXATTRIBPOINTERPROC, glVertexAttribPointer );
    GLDEFINEFUNCTION ( PFNGLGENBUFFERSPROC, glGenBuffers );
    GLDEFINEFUNCTION ( PFNGLBINDBUFFERPROC, glBindBuffer );
    GLDEFINEFUNCTION ( PFNGLBUFFERDATAPROC, glBufferData );
    GLDEFINEFUNCTION ( PFNGLDELETEBUFFERSPROC, glDeleteBuffers );
    GLDEFINEFUNCTION ( PFNGLGENVERTEXARRAYSPROC, glGenVertexArrays );
    GLDEFINEFUNCTION ( PFNGLBINDVERTEXARRAYPROC, glBindVertexArray );
    GLDEFINEFUNCTION ( PFNGLDELETEVERTEXARRAYSPROC, glDeleteVertexArrays );

    bool LoadOpenGLFunctions()
    {
#ifdef WIN32
        const bool contextAvailable = ( wglGetCurrentContext() != nullptr );
#else
        const bool contextAvailable = ( glXGetCurrentContext() != nullptr );
#endif
        if ( !contextAvailable && !CreateOpenGLContext() )
        {
            return false;
        }

        GLGETPROCADDRESS ( PFNGLATTACHSHADERPROC, glAttachShader );
        GLGETPROCADDRESS ( PFNGLCOMPILESHADERPROC, glCompileShader );
        GLGETPROCADDRESS ( PFNGLCREATEPROGRAMPROC, glCreateProgram );
        GLGETPROCADDRESS ( PFNGLCREATESHADERPROC, glCreateShader );
        GLGETPROCADDRESS ( PFNGLDELETEPROGRAMPROC, glDeleteProgram );
        GLGETPROCADDRESS ( PFNGLDELETESHADERPROC, glDeleteShader );
        GLGETPROCADDRESS ( PFNGLENABLEVERTEXATTRIBARRAYPROC, glEnableVertexAttribArray );
        GLGETPROCADDRESS ( PFNGLGETATTRIBLOCATIONPROC, glGetAttribLocation );
        GLGETPROCADDRESS ( PFNGLGETPROGRAMIVPROC, glGetProgramiv );
        GLGETPROCADDRESS ( PFNGLGETPROGRAMINFOLOGPROC, glGetProgramInfoLog );
        GLGETPROCADDRESS ( PFNGLGETSHADERIVPROC, glGetShaderiv );
        GLGETPROCADDRESS ( PFNGLGETSHADERINFOLOGPROC, glGetShaderInfoLog );
        GLGETPROCADDRESS ( PFNGLGETUNIFORMLOCATIONPROC, glGetUniformLocation );
        GLGETPROCADDRESS ( PFNGLLINKPROGRAMPROC, glLinkProgram );
        GLGETPROCADDRESS ( PFNGLSHADERSOURCEPROC, glShaderSource );
        GLGETPROCADDRESS ( PFNGLUSEPROGRAMPROC, glUseProgram );
        GLGETPROCADDRESS ( PFNGLUNIFORM1IPROC, glUniform1i );
        GLGETPROCADDRESS ( PFNGLUNIFORMMATRIX4FVPROC, glUniformMatrix4fv );
        GLGETPROCADDRESS ( PFNGLVERTEXATTRIBPOINTERPROC, glVertexAttribPointer );
        GLGETPROCADDRESS ( PFNGLGENBUFFERSPROC, glGenBuffers );
        GLGETPROCADDRESS ( PFNGLBINDBUFFERPROC, glBindBuffer );
        GLGETPROCADDRESS ( PFNGLBUFFERDATAPROC, glBufferData );
        GLGETPROCADDRESS ( PFNGLDELETEBUFFERSPROC, glDeleteBuffers );
        GLGETPROCADDRESS ( PFNGLGENVERTEXARRAYSPROC, glGenVertexArrays );
        GLGETPROCADDRESS ( PFNGLBINDVERTEXARRAYPROC, glBindVertexArray );
        GLGETPROCADDRESS ( PFNGLDELETEVERTEXARRAYSPROC, glDeleteVertexArrays );

        if ( !contextAvailable )
        {
            DestroyOpenGLContext();
        }
        return true;
    }
}