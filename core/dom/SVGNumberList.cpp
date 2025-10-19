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
#include "aeongui/dom/SVGNumberList.hpp"
#include "aeongui/dom/DOMException.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        SVGNumberList::SVGNumberList() = default;
        SVGNumberList::~SVGNumberList() = default;

        unsigned long SVGNumberList::length() const
        {
            return static_cast<unsigned long> ( mItems.size() );
        }

        unsigned long SVGNumberList::numberOfItems() const
        {
            return length();
        }

        void SVGNumberList::clear()
        {
            mItems.clear();
        }

        float SVGNumberList::initialize ( float newItem )
        {
            mItems.clear();
            mItems.push_back ( newItem );
            return mItems.back();
        }

        float SVGNumberList::getItem ( unsigned long index ) const
        {
            if ( index >= mItems.size() )
            {
                throw DOMIndexSizeError ( "Index out of bounds" );
            }
            return mItems[index];
        }

        float SVGNumberList::insertItemBefore ( float newItem, unsigned long index )
        {
            if ( index > mItems.size() )
            {
                index = static_cast<unsigned long> ( mItems.size() );
            }
            mItems.insert ( mItems.begin() + index, newItem );
            return mItems[index];
        }

        float SVGNumberList::replaceItem ( float newItem, unsigned long index )
        {
            if ( index >= mItems.size() )
            {
                throw DOMIndexSizeError ( "Index out of bounds" );
            }
            mItems[index] = newItem;
            return mItems[index];
        }

        float SVGNumberList::removeItem ( unsigned long index )
        {
            if ( index >= mItems.size() )
            {
                throw DOMIndexSizeError ( "Index out of bounds" );
            }
            float item = mItems[index];
            mItems.erase ( mItems.begin() + index );
            return item;
        }

        float SVGNumberList::appendItem ( float newItem )
        {
            mItems.push_back ( newItem );
            return mItems.back();
        }
    }
}