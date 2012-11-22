#ifndef AEONGUI_OPENGL_RENDERER
#define AEONGUI_OPENGL_RENDERER
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
#include "Renderer.h"
#include <map>
#include <vector>
#include "Integer.h"
namespace AeonGUI
{
    class OpenGLRenderer : public Renderer
    {
    public:
        OpenGLRenderer();
        ~OpenGLRenderer() {};
        /*! \copydoc Renderer::Initialize. */
        bool Initialize ( int32_t screen_width, int32_t screen_height );
        /*! \copydoc Renderer::Finalize. */
        void Finalize();
        /*! \copydoc Renderer::BeginRender. */
        void BeginRender();
        /*! \copydoc Renderer::EndRender. */
        void EndRender();
        /*! \copydoc Renderer::DrawRect. */
        void DrawRect ( Color color, const Rect* rect );
        /*! \copydoc Renderer::DrawRectOutline. */
        void DrawRectOutline ( Color color, const Rect* rect );
        /*! \copydoc Renderer::DrawImage. */
        void DrawImage ( Color color, int32_t x, int32_t y, Image* image );
        /*! \copydoc Renderer::DrawString. */
        void DrawString ( Color color, int32_t x, int32_t y, const wchar_t* text );
    private:
#if 0
        Image* NewImage ( uint32_t width, uint32_t height, Image::Format format, Image::Type type, void* data );
        void DeleteImage ( Image* image );
        struct TextureData
        {
            uint32_t texture;
            uint32_t tw;
            uint32_t th;
            std::vector<Image*> images;
        };
#endif
        int32_t viewport_w;
        int32_t viewport_h;
        int32_t screen_w;
        int32_t screen_h;
        unsigned int screen_texture;
        uint8_t* screen_bitmap;
        int32_t screen_texture_w;
        int32_t screen_texture_h;
        float screen_texture_ratio_w;
        float screen_texture_ratio_h;
        /// Card maximum texture size for texture atlas
        int max_texture_size;
        uint32_t vert_shader;
        uint32_t frag_shader;
        uint32_t shader_program;
        static uint32_t TypeTable[];
        static uint32_t FormatTable[];
    };
}
#endif
