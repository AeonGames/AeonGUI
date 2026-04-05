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
#ifndef AEONGUI_SVGNUMBERLIST_HPP
#define AEONGUI_SVGNUMBERLIST_HPP

#include <vector>
#include "aeongui/Platform.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        /** @brief Ordered list of floating-point numbers (SVG).
         *
         *  Provides indexed access and modification of numeric items.
         *  @see https://www.w3.org/TR/SVG2/types.html#InterfaceSVGNumberList
         */
        class AEONGUI_DLL SVGNumberList
        {
        public:
            /** @brief Default constructor. */
            SVGNumberList();
            /** @brief Destructor. */
            ~SVGNumberList();

            /** @brief Get the number of items.
             *  @return The item count. */
            unsigned long length() const;
            /** @brief Get the number of items (alias).
             *  @return The item count. */
            unsigned long numberOfItems() const;

            /** @brief Remove all items. */
            void clear();
            /** @brief Clear and set a single item.
             *  @param newItem The value to initialize the list with.
             *  @return The initialized value.
             */
            float initialize ( float newItem );
            /** @brief Get the item at the given index.
             *  @param index Zero-based index.
             *  @return The value at the index.
             */
            float getItem ( unsigned long index ) const;
            /** @brief Insert an item before the given index.
             *  @param newItem The value to insert.
             *  @param index   Position to insert before.
             *  @return The inserted value.
             */
            float insertItemBefore ( float newItem, unsigned long index );
            /** @brief Replace the item at the given index.
             *  @param newItem The replacement value.
             *  @param index   Zero-based index.
             *  @return The new value.
             */
            float replaceItem ( float newItem, unsigned long index );
            /** @brief Remove the item at the given index.
             *  @param index Zero-based index.
             *  @return The removed value.
             */
            float removeItem ( unsigned long index );
            /** @brief Append an item to the end.
             *  @param newItem The value to append.
             *  @return The appended value.
             */
            float appendItem ( float newItem );

        private:
            PRIVATE_TEMPLATE_MEMBERS_START
            std::vector<float> mItems;
            PRIVATE_TEMPLATE_MEMBERS_END
        };
    }
}

#endif
