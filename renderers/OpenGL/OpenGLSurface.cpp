﻿/******************************************************************************
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
#include "OpenGLSurface.h"

AeonGUI::OpenGLSurface::OpenGLSurface ( uint32_t aWidth, uint32_t aHeight )
{
}

AeonGUI::OpenGLSurface::~OpenGLSurface()
{
}

uint32_t AeonGUI::OpenGLSurface::Width() const
{
    return 0;
}

uint32_t AeonGUI::OpenGLSurface::Height() const
{
    return 0;
}

void * AeonGUI::OpenGLSurface::MapMemory()
{
    return nullptr;
}

DLL void AeonGUI::OpenGLSurface::UnmapMemory()
{
}

DLL void AeonGUI::OpenGLSurface::ReSize ( uint32_t aWidth, uint32_t aHeight )
{
}
