#ifndef AEONGUI_OPENGL_RENDERER
#define AEONGUI_OPENGL_RENDERER
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

#include "Platform.h"
#include "Renderer.h"
#include "Integer.h"

namespace AeonGUI
{
    /// OpenGL 3.2 Renderer class.
    class OpenGLRenderer : public Renderer
    {
    public:
        DLL OpenGLRenderer();
        DLL ~OpenGLRenderer();
        /// No Copying allowed
        OpenGLRenderer ( const OpenGLRenderer& ) = delete;
        /*! \copydoc Renderer::Initialize. */
        DLL bool Initialize () override final;
        /*! \copydoc Renderer::Finalize. */
        DLL void Finalize() override final;
        /*! \copydoc Renderer::BeginRender. */
        DLL void BeginRender() override final;
        /*! \copydoc Renderer::EndRender. */
        DLL void EndRender() override final;
        DLL uint32_t SurfaceWidth() const override final;
        DLL uint32_t SurfaceHeight() const override final;
        DLL void* MapMemory() override final;
        DLL void UnmapMemory() override final;
        DLL void ReSize ( uint32_t aWidth, uint32_t aHeight ) override final;
    private:
        uint32_t mShaderProgram;
        uint32_t mVertexBufferObject;
        uint32_t mVertexArrayObject;
    };
}
#endif
