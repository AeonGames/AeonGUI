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
#ifndef AEONGUI_SVGLENGTHLIST_HPP
#define AEONGUI_SVGLENGTHLIST_HPP

#include <vector>
#include "SVGLength.hpp"
#include "aeongui/Platform.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        /** @brief Ordered list of SVGLength values.
         *
         *  Provides indexed access and modification of SVG length items.
         *  @see https://www.w3.org/TR/SVG2/types.html#InterfaceSVGLengthList
         */
        class AEONGUI_DLL SVGLengthList
        {
        public:
            /** @brief Default constructor. */
            SVGLengthList();
            /** @brief Destructor. */
            ~SVGLengthList();

            /** @brief Get the number of items.
             *  @return The item count. */
            unsigned long length() const;
            /** @brief Get the number of items (alias).
             *  @return The item count. */
            unsigned long numberOfItems() const;

            /** @brief Remove all items. */
            void clear();
            /** @brief Clear and set a single item.
             *  @param newItem The item to initialize the list with.
             *  @return The initialized item.
             */
            SVGLength initialize ( const SVGLength& newItem );
            /** @brief Get the item at the given index.
             *  @param index Zero-based index.
             *  @return The item at the index.
             */
            SVGLength getItem ( unsigned long index ) const;
            /** @brief Insert an item before the given index.
             *  @param newItem The item to insert.
             *  @param index   Position to insert before.
             *  @return The inserted item.
             */
            SVGLength insertItemBefore ( const SVGLength& newItem, unsigned long index );
            /** @brief Replace the item at the given index.
             *  @param newItem The replacement item.
             *  @param index   Zero-based index.
             *  @return The new item.
             */
            SVGLength replaceItem ( const SVGLength& newItem, unsigned long index );
            /** @brief Remove the item at the given index.
             *  @param index Zero-based index.
             *  @return The removed item.
             */
            SVGLength removeItem ( unsigned long index );
            /** @brief Append an item to the end.
             *  @param newItem The item to append.
             *  @return The appended item.
             */
            SVGLength appendItem ( const SVGLength& newItem );

        private:
            PRIVATE_TEMPLATE_MEMBERS_START
            std::vector<SVGLength> mItems;
            PRIVATE_TEMPLATE_MEMBERS_END
        };
    }
}

#endif