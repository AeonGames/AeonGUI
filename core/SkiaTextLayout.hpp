/*
Copyright (C) 2026 Rodrigo Jose Hernandez Cordoba

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
#ifndef AEONGUI_SKIATEXTLAYOUT_H
#define AEONGUI_SKIATEXTLAYOUT_H
#include <cstdint>
#include <string>
#include <vector>
#include "aeongui/TextLayout.hpp"

struct _PangoLayout;
typedef struct _PangoLayout PangoLayout;
struct _PangoContext;
typedef struct _PangoContext PangoContext;
struct _PangoFontDescription;
typedef struct _PangoFontDescription PangoFontDescription;

namespace AeonGUI
{
    /** @brief Skia-backend text layout implementation.
     *
     *  Uses PangoLayout backed by PangoFT2 for text measurement and layout.
     */
    class SkiaTextLayout : public TextLayout
    {
    public:
        AEONGUI_DLL SkiaTextLayout();
        AEONGUI_DLL ~SkiaTextLayout() override;
        AEONGUI_DLL void SetText ( const std::string& aText ) override;
        AEONGUI_DLL void SetFontFamily ( const std::string& aFamily ) override;
        AEONGUI_DLL void SetFontSize ( double aSize ) override;
        AEONGUI_DLL void SetFontWeight ( int aWeight ) override;
        AEONGUI_DLL void SetFontStyle ( int aStyle ) override;
        AEONGUI_DLL double GetTextWidth() const override;
        AEONGUI_DLL double GetTextHeight() const override;
        AEONGUI_DLL double GetBaseline() const override;
        AEONGUI_DLL double GetCharOffsetX ( long aIndex ) const override;
        /** @brief Get the underlying PangoLayout.
         *  @return Pointer to the PangoLayout.
         */
        PangoLayout* GetPangoLayout() const;
    private:
        void UpdateFontDescription();
        std::string mFontFamily{"sans-serif"};
        double mFontSize{16.0};
        int mFontWeight{400};
        int mFontStyle{0};
        PangoContext* mPangoContext{nullptr};
        PangoLayout* mLayout{nullptr};
        PangoFontDescription* mFontDescription{nullptr};
    };
}
#endif
