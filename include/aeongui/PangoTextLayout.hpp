/*
Copyright (C) 2025 Rodrigo Jose Hernandez Cordoba

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
#include <memory>
#include "aeongui/TextLayout.hpp"
#include "aeongui/CairoCanvas.hpp"

struct _PangoLayout;
typedef struct _PangoLayout PangoLayout;
struct _cairo;
typedef struct _cairo cairo_t;
typedef void* gpointer;

namespace AeonGUI
{
    class PangoTextLayout : public TextLayout
    {
    public:
        DLL PangoTextLayout ( const CairoCanvas& aCanvas );
        DLL ~PangoTextLayout() override;
    private:
        std::unique_ptr<cairo_t, void ( * ) ( cairo_t * ) > mCairoContext;
        std::unique_ptr<PangoLayout, void ( * ) ( gpointer ) > mLayout;
    };
}

#endif