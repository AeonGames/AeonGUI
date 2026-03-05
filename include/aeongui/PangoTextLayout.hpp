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
    class PangoTextLayout : public TextLayout
    {
    public:
        DLL PangoTextLayout();
        DLL ~PangoTextLayout() override;
        DLL void SetText ( const std::string& aText ) override;
        DLL void SetFontFamily ( const std::string& aFamily ) override;
        DLL void SetFontSize ( double aSize ) override;
        DLL void SetFontWeight ( int aWeight ) override;
        DLL void SetFontStyle ( int aStyle ) override;
        DLL double GetTextWidth() const override;
        DLL double GetTextHeight() const override;
        DLL double GetBaseline() const override;
        DLL double GetCharOffsetX ( long aIndex ) const override;
        /// Access the underlying PangoLayout for advanced use.
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