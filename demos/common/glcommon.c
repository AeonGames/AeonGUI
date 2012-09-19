/******************************************************************************
Copyright 2010-2012 Rodrigo Hernandez Cordoba

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
#ifdef WIN32
#include <windows.h>
#endif
#include "glcommon.h"
#include <GL/gl.h>

void DrawCube ( float size )
{
#if 0
    glBegin ( GL_QUADS );
    glColor3f ( 0.0f, 0.0f, 1.0f );
    glVertex3f ( size, size, -size );
    glVertex3f ( -size, size, -size );
    glVertex3f ( -size, size, size );
    glVertex3f ( size, size, size );
    glColor3f ( 1.0f, 0.5f, 0.0f );
    glVertex3f ( size, -size, size );
    glVertex3f ( -size, -size, size );
    glVertex3f ( -size, -size, -size );
    glVertex3f ( size, -size, -size );
    glColor3f ( 1.0f, 0.0f, 0.0f );
    glVertex3f ( size, size, size );
    glVertex3f ( -size, size, size );
    glVertex3f ( -size, -size, size );
    glVertex3f ( size, -size, size );
    glColor3f ( 1.0f, 1.0f, 0.0f );
    glVertex3f ( size, -size, -size );
    glVertex3f ( -size, -size, -size );
    glVertex3f ( -size, size, -size );
    glVertex3f ( size, size, -size );
    glColor3f ( 0.0f, 1.0f, 0.0f );
    glVertex3f ( -size, size, size );
    glVertex3f ( -size, size, -size );
    glVertex3f ( -size, -size, -size );
    glVertex3f ( -size, -size, size );
    glColor3f ( 1.0f, 0.0f, 1.0f );
    glVertex3f ( size, size, -size );
    glVertex3f ( size, size, size );
    glVertex3f ( size, -size, size );
    glVertex3f ( size, -size, -size );
    glEnd();
#else
    float cube [] = {
     size, size, -size ,
     0.0f, 0.0f, 1.0f ,
     -size, size, -size ,
     0.0f, 0.0f, 1.0f ,
     -size, size, size ,
     0.0f, 0.0f, 1.0f ,
     size, size, size ,
     0.0f, 0.0f, 1.0f ,
     size, -size, size ,
     1.0f, 0.5f, 0.0f ,
     -size, -size, size ,
     1.0f, 0.5f, 0.0f ,
     -size, -size, -size ,
     1.0f, 0.5f, 0.0f ,
     size, -size, -size ,
     1.0f, 0.5f, 0.0f ,
     size, size, size ,
     1.0f, 0.0f, 0.0f ,
     -size, size, size ,
     1.0f, 0.0f, 0.0f ,
     -size, -size, size ,
     1.0f, 0.0f, 0.0f ,
     size, -size, size ,
     1.0f, 0.0f, 0.0f ,
     size, -size, -size ,
     1.0f, 1.0f, 0.0f ,
     -size, -size, -size ,
     1.0f, 1.0f, 0.0f ,
     -size, size, -size ,
     1.0f, 1.0f, 0.0f ,
     size, size, -size ,
     1.0f, 1.0f, 0.0f ,
     -size, size, size ,
     0.0f, 1.0f, 0.0f ,
     -size, size, -size ,
     0.0f, 1.0f, 0.0f ,
     -size, -size, -size ,
     0.0f, 1.0f, 0.0f ,
     -size, -size, size ,
     0.0f, 1.0f, 0.0f ,
     size, size, -size ,
     1.0f, 0.0f, 1.0f ,
     size, size, size ,
     1.0f, 0.0f, 1.0f ,
     size, -size, size ,
     1.0f, 0.0f, 1.0f ,
     size, -size, -size ,
     1.0f, 0.0f, 1.0f
    };
    glVertexPointer(3,GL_FLOAT,sizeof(float)*6,cube);
    glColorPointer(3,GL_FLOAT,sizeof(float)*6,cube+3);
    glPushClientAttrib(GL_CLIENT_VERTEX_ARRAY_BIT);
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    glDrawArrays(GL_QUADS,0,24);
    glPopClientAttrib();
#endif
}
