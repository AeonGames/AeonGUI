#ifndef AEONGUI_RENDERER_H
#define AEONGUI_RENDERER_H
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

#include <cstddef>
#include <string>
#include <map>

#include "Image.h"
#include "Font.h"
#include "Rect.h"
#include "Color.h"

namespace AeonGUI
{
    /*! \brief Interface class Renderer. */
    class Renderer
    {
    public:
        /// Standard constructor.
        Renderer()
        {
            font = NULL;
        };
        /// Virtual destructor so all derived destructors are called accordingly.
        virtual ~Renderer() {};
        /*! \brief Initialization Function.
            \param screen_width [in] Current Screen width resolution.
            \param screen_height [in] Current Screen width resolution.
            \note This function takes the screen or display resolution, not the window size.
            \return true on success, false on failure.
        */
        virtual bool Initialize ( int32_t screen_width, int32_t screen_height ) = 0;
        /*! \brief Finalization Function. */
        virtual void Finalize() = 0;
        /*! \brief Used to do any internal screen size memory updates.
            \param screen_width with to set the screen to.
            \param screen_height with to set the screen to.
            \return true on success, false on failure.
        */
        virtual bool ChangeScreenSize ( int32_t screen_width, int32_t screen_height )
        {
            return true;
        };
        /*! \brief Used to do any pre-rendering initialization. */
        virtual void BeginRender() {};
        /*! \brief Used to do any post-rendering cleanup. */
        virtual void EndRender() {};
        /*! \brief Set default font.
            \param newfont [in] pointer to font object to set as default.
        */
        virtual inline void SetFont ( Font* newfont )
        {
            font = newfont;
        }
        /*! \brief Get a pointer to the current font.
            \return Pointer to current font, may be null
        */
        virtual inline Font* GetFont()
        {
            return font;
        }
        ///\name Drawing Functions
        //@{
        /*! \brief Draws a Rect in screen space.
            \param color [in] Pointer to the color to use for drawing.
            \param rect [in] Pointer to the rect.
        */
        virtual void DrawRect ( Color color, const Rect* rect ) = 0;

        /*! \brief Draws a Rect outline in screen space.
            \param color [in] Pointer to the color to use for drawing.
            \param rect [in] Pointer to the rect.
        */
        virtual void DrawRectOutline ( Color color, const Rect* rect ) = 0;

        /*! \brief Draws an image in screen space.
            \param color [in] Pointer to the color to use for drawing.
            \param rect [in] Pointer to the rect.
        */
        virtual void DrawImage ( Color color, int32_t x, int32_t y, Image* image ) = 0;
        /*! \brief Render some text.
            \param font [in] To the font to use.
            \sa NewFont,DeleteFont.
        */
        virtual void DrawString ( Color color, int32_t x, int32_t y, const wchar_t* text ) = 0;
        //@}
#if 0
        /*! \brief Image object factory method.

            Keeping image data on GPU memory is more efficient than passing it over each time it is needed,
            NewImage and DeleteImage allow the implementor to take advantage of this.
            \sa DeleteImage
        */
        virtual Image* NewImage ( uint32_t width, uint32_t height, Image::Format format, Image::Type type, void* data ) = 0;
        /*! \brief Destroys a previously created image object.
            \sa NewImage,DeleteFont.
        */
        virtual void DeleteImage ( Image* image ) = 0;

        Image* GetImage ( std::string id, uint32_t width, uint32_t height, Image::Format format, Image::Type type, void* data );
        void ReleaseImage ( Image* image );
#endif
    protected:
        /// Current font used to Render Text.
        Font* font;
    };
}
#endif
