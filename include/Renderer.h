#ifndef AEONGUI_RENDERER_H
#define AEONGUI_RENDERER_H
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

#include <cstddef>
#include <string>
#include <map>

#include "Image.h"
#include "Font.h"
#include "Rect.h"
#include "Color.h"

namespace AeonGUI
{
    class Widget;
    enum ResizeAlgorithm
    {
        NEAREST = 0,
        LANCZOS = 1
    };
    /*! \brief Renderer base class. */
    class Renderer
    {
    public:
        /// Standard constructor.
        Renderer();
        /// Virtual destructor so all derived destructors are called accordingly.
        virtual ~Renderer();
        /*! \brief Initialization Function.
            \return true on success, false on failure.
        */
        virtual bool Initialize();
        /*! \brief Finalization Function. */
        virtual void Finalize();
        /*! \brief Used to do any internal screen size memory updates.
            \param screen_width with to set the screen to.
            \param screen_height with to set the screen to.
            \return true on success, false on failure.
        */
        virtual bool ChangeScreenSize ( int32_t screen_width, int32_t screen_height );

        /*! \brief Used to do any pre-rendering initialization. */
        virtual void BeginRender() {};

        /*! \brief Used to do any post-rendering cleanup. */
        virtual void EndRender() {};

        /*! \brief Set default font.
            \param newfont [in] pointer to font object to set as default.
        */
        void SetFont ( Font* newfont );

        /*! \brief Get a pointer to the current font.
            \return Pointer to current font, may be null
        */
        const Font* GetFont();

        /*! \brief Add widget to widget rendering list.
            It is the user resposibility to allocate, initialize, manage and eventually deallocate the memory the widget object resides in.
            \param widget Pointer to the widget object to add.*/
        void AddWidget ( Widget* widget );
        /*! \brief Remove widget to widget rendering list.
            It is the user resposibility to allocate, initialize, manage and eventually deallocate the memory the widget object resides in.
            \param widget Pointer to the widget object to remove.*/
        void RemoveWidget ( Widget* widget );

        /*! Renders the widget list.
            Must be called between BeginRender and EndRender calls.
        */
        void RenderWidgets();

        ///\name Drawing Functions
        //@{
        /*! \brief Draws a Rect in screen space.
            \param color [in] Pointer to the color to use for drawing.
            \param rect [in] Pointer to the rect.
        */
        void DrawRect ( Color color, const Rect* rect );

        /*! \brief Draws a Rect outline in screen space.
            \param color [in] Pointer to the color to use for drawing.
            \param rect [in] Pointer to the rect.
        */
        void DrawRectOutline ( Color color, const Rect* rect );

        /*! \brief Draws an image in screen space.

            If the image contains patch 9 information, this function makes use of that information,
            if not, this function is the same as calling Image::DrawSubImage with the subx,suby,subw and subh arguments set to zero (0).

            \note if you want to save a handful of function calls and you know an image does not contain patch 9 information, you may want to call Image::DrawSubImage with default parameters instead.
            \param image [in] Pointer to the image object to draw.
            \param x [in] X screen coordinate to draw the image.
            \param y [in] Y screen coordinate to draw the image.
            \param w [in] Optional width of the image to draw, used for scaling, if 0, the image is drawn horizontaly unscaled.
            \param h [in] Optional height of the image to draw, used for scaling, if 0, the image is drawn verticaly unscaled.
            \param algorithm [in] Optional enumeration value for the resize algorithm to be used, NEAREST is used by default.
            \sa Image::DrawSubImage
        */
        void DrawImage ( Image* image, int32_t x, int32_t y, int32_t w = 0, int32_t h = 0, ResizeAlgorithm algorithm = NEAREST );

        /*! \brief Draws a sub area of an image in screen space.

            This function supports image scaling but does not support patch 9 scaling, to draw patch 9 images, use Image::DrawImage instead.

            \note There is currently no way to draw a subimage of a post scaled patch 9 image, if you feed a patch 9 image to this function, it will draw a subimage of the original unscaled image.
            \param image [in] Pointer to the image object to draw.
            \param x [in] X screen coordinate to draw the image.
            \param y [in] Y screen coordinate to draw the image.
            \param subx [in] Starting X image coordinate of the sub area.
            \param suby [in] Starting Y image coordinate of the sub area.
            \param subw [in] Sub Area Width.
            \param subh [in] Sub area Height.
            \param w [in] Optional width of the image to draw, used for scaling, if 0, the image is drawn horizontaly unscaled.
            \param h [in] Optional height of the image to draw, used for scaling, if 0, the image is drawn verticaly unscaled.
            \param algorithm [in] Optional enumeration value for the resize algorithm to be used, NEAREST is used by default.
            \sa Image::DrawImage
        */
        void DrawSubImage ( Image* image, int32_t x, int32_t y, int32_t subx = 0, int32_t suby = 0, int32_t subw = 0, int32_t subh = 0, int32_t w = 0, int32_t h = 0, ResizeAlgorithm algorithm = NEAREST );

        /*! \brief Draw a string of text.
            \param color [in] Color to draw the string with.
            \param x [in] X coordinate to draw the string at.
            \param y [in] Y coordinate to draw the string at.
            \param text [in] Text to draw.
        */
        void DrawString ( Color color, int32_t x, int32_t y, const wchar_t* text );
        //@}
    protected:
        /// Current font used to Render Text.
        Font* font;
        /// Screen width.
        int32_t screen_w;
        /// Screen height.
        int32_t screen_h;
        /// Screen buffer.
        uint8_t* screen_bitmap;
        /// Widget linked list.
        Widget* widgets;
    };
}
#endif
