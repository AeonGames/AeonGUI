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
#ifndef AEONGUI_SVGLENGTHLIST_HPP
#define AEONGUI_SVGLENGTHLIST_HPP

#include <vector>
#include "SVGLength.hpp"
#include "aeongui/Platform.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        class DLL SVGLengthList
        {
        public:
            SVGLengthList();
            ~SVGLengthList();

            unsigned long length() const;
            unsigned long numberOfItems() const;

            void clear();
            SVGLength initialize ( const SVGLength& newItem );
            SVGLength getItem ( unsigned long index ) const;
            SVGLength insertItemBefore ( const SVGLength& newItem, unsigned long index );
            SVGLength replaceItem ( const SVGLength& newItem, unsigned long index );
            SVGLength removeItem ( unsigned long index );
            SVGLength appendItem ( const SVGLength& newItem );

        private:
            std::vector<SVGLength> mItems;
        };
    }
}

#endif