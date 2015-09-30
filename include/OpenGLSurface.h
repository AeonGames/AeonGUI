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
#ifndef AEONGUI_OPENGLSURFACE_H
#define AEONGUI_OPENGLSURFACE_H

#include "Platform.h"
#include "Surface.h"

namespace AeonGUI
{
    class OpenGLSurface : public Surface
    {
    public:
        DLL OpenGLSurface ( uint32_t aWidth, uint32_t aHeight );
        DLL ~OpenGLSurface();
        DLL uint32_t Width() const override final;
        DLL uint32_t Height() const override final;
        DLL void* MapMemory() override final;
        DLL void UnmapMemory() override final;
        DLL void ReSize ( uint32_t aWidth, uint32_t aHeight ) override final;
    private:
    };
}
#endif