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
#ifndef AEONGUI_SVGNUMBERLIST_HPP
#define AEONGUI_SVGNUMBERLIST_HPP

#include <vector>
#include "aeongui/Platform.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        class DLL SVGNumberList
        {
        public:
            SVGNumberList();
            ~SVGNumberList();

            unsigned long length() const;
            unsigned long numberOfItems() const;

            void clear();
            float initialize ( float newItem );
            float getItem ( unsigned long index ) const;
            float insertItemBefore ( float newItem, unsigned long index );
            float replaceItem ( float newItem, unsigned long index );
            float removeItem ( unsigned long index );
            float appendItem ( float newItem );

        private:
            PRIVATE_TEMPLATE_MEMBERS_START
            std::vector<float> mItems;
            PRIVATE_TEMPLATE_MEMBERS_END
        };
    }
}

#endif
