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
#include "aeongui/dom/SVGLengthList.hpp"
#include "aeongui/dom/DOMException.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        SVGLengthList::SVGLengthList() = default;
        SVGLengthList::~SVGLengthList() = default;

        unsigned long SVGLengthList::length() const
        {
            return static_cast<unsigned long> ( mItems.size() );
        }

        unsigned long SVGLengthList::numberOfItems() const
        {
            return length();
        }

        void SVGLengthList::clear()
        {
            mItems.clear();
        }

        SVGLength SVGLengthList::initialize ( const SVGLength& newItem )
        {
            mItems.clear();
            mItems.push_back ( newItem );
            return mItems.back();
        }

        SVGLength SVGLengthList::getItem ( unsigned long index ) const
        {
            if ( index >= mItems.size() )
            {
                throw DOMIndexSizeError ( "Index out of bounds" );
            }
            return mItems[index];
        }

        SVGLength SVGLengthList::insertItemBefore ( const SVGLength& newItem, unsigned long index )
        {
            if ( index > mItems.size() )
            {
                index = static_cast<unsigned long> ( mItems.size() );
            }
            mItems.insert ( mItems.begin() + index, newItem );
            return mItems[index];
        }

        SVGLength SVGLengthList::replaceItem ( const SVGLength& newItem, unsigned long index )
        {
            if ( index >= mItems.size() )
            {
                throw DOMIndexSizeError ( "Index out of bounds" );
            }
            mItems[index] = newItem;
            return mItems[index];
        }

        SVGLength SVGLengthList::removeItem ( unsigned long index )
        {
            if ( index >= mItems.size() )
            {
                throw DOMIndexSizeError ( "Index out of bounds" );
            }
            SVGLength item = mItems[index];
            mItems.erase ( mItems.begin() + index );
            return item;
        }

        SVGLength SVGLengthList::appendItem ( const SVGLength& newItem )
        {
            mItems.push_back ( newItem );
            return mItems.back();
        }
    }
}