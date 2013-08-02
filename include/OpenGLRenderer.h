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

#include "Renderer.h"
#include "Integer.h"

namespace AeonGUI
{
    /// OpenGL Renderer class.
    class OpenGLRenderer : public Renderer
    {
    public:
        OpenGLRenderer();
        ~OpenGLRenderer() {};
        /*! \copydoc Renderer::ChangeScreenSize. */
        bool ChangeScreenSize ( int32_t screen_width, int32_t screen_height );
        /*! \copydoc Renderer::Initialize. */
        bool Initialize ();
        /*! \copydoc Renderer::Finalize. */
        void Finalize();
        /*! \copydoc Renderer::BeginRender. */
        void BeginRender();
        /*! \copydoc Renderer::EndRender. */
        void EndRender();
    private:
        unsigned int screen_texture;
        /// Card maximum texture size for texture atlas
        int max_texture_size;
        uint32_t vert_shader;
        uint32_t frag_shader;
        uint32_t shader_program;
        uint32_t vertex_buffer_object;
        uint32_t vertex_array_object;
    };
}
#endif
