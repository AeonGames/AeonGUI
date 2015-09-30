#ifndef AEONGUI_OPENGL_H
#define AEONGUI_OPENGL_H
/******************************************************************************
Copyright 2010-2013 Rodrigo Hernandez Cordoba

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

#include "glcorearb.h"
#include <GL/gl.h>

#define GLDECLAREFUNCTION(glFunctionType,glFunction) extern glFunctionType glFunction;

namespace AeonGUI
{
    GLDECLAREFUNCTION ( PFNGLATTACHSHADERPROC, glAttachShader );
    GLDECLAREFUNCTION ( PFNGLCOMPILESHADERPROC, glCompileShader );
    GLDECLAREFUNCTION ( PFNGLCREATEPROGRAMPROC, glCreateProgram );
    GLDECLAREFUNCTION ( PFNGLCREATESHADERPROC, glCreateShader );
    GLDECLAREFUNCTION ( PFNGLDELETEPROGRAMPROC, glDeleteProgram );
    GLDECLAREFUNCTION ( PFNGLDELETESHADERPROC, glDeleteShader );
    GLDECLAREFUNCTION ( PFNGLENABLEVERTEXATTRIBARRAYPROC, glEnableVertexAttribArray );
    GLDECLAREFUNCTION ( PFNGLGETATTRIBLOCATIONPROC, glGetAttribLocation );
    GLDECLAREFUNCTION ( PFNGLGETPROGRAMIVPROC, glGetProgramiv );
    GLDECLAREFUNCTION ( PFNGLGETPROGRAMINFOLOGPROC, glGetProgramInfoLog );
    GLDECLAREFUNCTION ( PFNGLGETSHADERIVPROC, glGetShaderiv );
    GLDECLAREFUNCTION ( PFNGLGETSHADERINFOLOGPROC, glGetShaderInfoLog );
    GLDECLAREFUNCTION ( PFNGLGETUNIFORMLOCATIONPROC, glGetUniformLocation );
    GLDECLAREFUNCTION ( PFNGLLINKPROGRAMPROC, glLinkProgram );
    GLDECLAREFUNCTION ( PFNGLSHADERSOURCEPROC, glShaderSource );
    GLDECLAREFUNCTION ( PFNGLUSEPROGRAMPROC, glUseProgram );
    GLDECLAREFUNCTION ( PFNGLUNIFORM1IPROC, glUniform1i );
    GLDECLAREFUNCTION ( PFNGLUNIFORMMATRIX4FVPROC, glUniformMatrix4fv );
    GLDECLAREFUNCTION ( PFNGLVERTEXATTRIBPOINTERPROC, glVertexAttribPointer );
    GLDECLAREFUNCTION ( PFNGLGENBUFFERSPROC, glGenBuffers );
    GLDECLAREFUNCTION ( PFNGLBINDBUFFERPROC, glBindBuffer );
    GLDECLAREFUNCTION ( PFNGLBUFFERDATAPROC, glBufferData );
    GLDECLAREFUNCTION ( PFNGLDELETEBUFFERSPROC, glDeleteBuffers );
    GLDECLAREFUNCTION ( PFNGLGENVERTEXARRAYSPROC, glGenVertexArrays );
    GLDECLAREFUNCTION ( PFNGLBINDVERTEXARRAYPROC, glBindVertexArray );
    GLDECLAREFUNCTION ( PFNGLDELETEVERTEXARRAYSPROC, glDeleteVertexArrays );

    bool CreateOpenGLContext();
    void DestroyOpenGLContext();
    bool LoadOpenGLFunctions();
}
#endif
