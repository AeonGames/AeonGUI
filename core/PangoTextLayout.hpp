/*
Copyright (C) 2025,2026 Rodrigo Jose Hernandez Cordoba

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef AEONGUI_PANGOTEXTLAYOUT_H
#define AEONGUI_PANGOTEXTLAYOUT_H
#include <cstdint>
#include <string>
#include <memory>
#include "aeongui/TextLayout.hpp"

struct _PangoLayout;
typedef struct _PangoLayout PangoLayout;
struct _PangoContext;
typedef struct _PangoContext PangoContext;
struct _PangoFontDescription;
typedef struct _PangoFontDescription PangoFontDescription;
typedef void* gpointer;

namespace AeonGUI
{
    /** @brief Pango-backed text layout implementation.
     *
     *  Uses PangoLayout and PangoFontDescription to measure and layout
     *  text for rendering by a Canvas.
     */
    class PangoTextLayout : public TextLayout
    {
    public:
        /** @brief Default constructor. Creates a layout with default font settings. */
        AEONGUI_DLL PangoTextLayout();
        /** @brief Destructor. Releases Pango resources. */
        AEONGUI_DLL ~PangoTextLayout() override;
        /** @brief Set the text content to lay out.
         *  @param aText UTF-8 text string.
         */
        AEONGUI_DLL void SetText ( const std::string& aText ) override;
        /** @brief Set the font family.
         *  @param aFamily Font family name (e.g. "sans-serif").
         */
        AEONGUI_DLL void SetFontFamily ( const std::string& aFamily ) override;
        /** @brief Set the font size.
         *  @param aSize Font size in CSS pixels.
         */
        AEONGUI_DLL void SetFontSize ( double aSize ) override;
        /** @brief Set the font weight.
         *  @param aWeight CSS font weight (400 = normal, 700 = bold).
         */
        AEONGUI_DLL void SetFontWeight ( int aWeight ) override;
        /** @brief Set the font style.
         *  @param aStyle 0 = normal, 1 = italic, 2 = oblique.
         */
        AEONGUI_DLL void SetFontStyle ( int aStyle ) override;
        /** @brief Get the logical width of the laid-out text.
         *  @return Width in CSS pixels.
         */
        AEONGUI_DLL double GetTextWidth() const override;
        /** @brief Get the logical height of the laid-out text.
         *  @return Height in CSS pixels.
         */
        AEONGUI_DLL double GetTextHeight() const override;
        /** @brief Get the baseline offset from the top of the layout.
         *  @return Baseline position in CSS pixels.
         */
        AEONGUI_DLL double GetBaseline() const override;
        /** @brief Get the x-offset of a character at the given byte index.
         *  @param aIndex UTF-8 byte index.
         *  @return X offset in CSS pixels.
         */
        AEONGUI_DLL double GetCharOffsetX ( int32_t aIndex ) const override;
        /** @brief Constrain the layout width to enable line wrapping.
         *  @param aWidth Wrap width in CSS pixels, or a negative value to disable wrapping.
         *
         *  Lines longer than @p aWidth are broken at word boundaries
         *  using Pango's default wrap mode.  This is a Pango-specific
         *  extension to the abstract TextLayout interface; the value
         *  must be set before SetText to take effect on subsequent
         *  measurements.
         */
        AEONGUI_DLL void SetWrapWidth ( double aWidth );
        /** @brief Access the underlying PangoLayout for advanced use.
         *  @return Pointer to the PangoLayout.
         */
        PangoLayout* GetPangoLayout() const;
    private:
        void UpdateFontDescription();
        PangoContext* mPangoContext{};
        PangoLayout* mLayout{};
        PangoFontDescription* mFontDescription{};
        std::string mFontFamily{"sans-serif"};
        double mFontSize{16.0};
        int mFontWeight{400};
        int mFontStyle{0};
    };
}

#endif