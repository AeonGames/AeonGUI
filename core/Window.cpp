/*
Copyright (C) 2019 Rodrigo Jose Hernandez Cordoba

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

#include <iostream>
#include <stdexcept>
#include <string>
#include "aeongui/Window.h"

namespace AeonGUI
{
    enum Element
    {
        unknown = 0,
        svg,
        g,
        path,
        rect,
        line,
        polyline,
        polygon,
        circle,
        ellipse
    };
    static Element GetElementNameEnum ( const std::string& aElementName )
    {
        if ( aElementName == "g" )
        {
            return g;
        }
        else if ( aElementName == "path" )
        {
            return path;
        }
        else if ( aElementName == "rect" )
        {
            return rect;
        }
        else if ( aElementName == "line" )
        {
            return line;
        }
        else if ( aElementName == "polyline" )
        {
            return polyline;
        }
        else if ( aElementName == "polygon" )
        {
            return polygon;
        }
        else if ( aElementName == "circle" )
        {
            return circle;
        }
        else if ( aElementName == "ellipse" )
        {
            return ellipse;
        }
        return unknown;
    }

    Window::Window ( const std::string aFilename, uint32_t aWidth, uint32_t aHeight ) :
        mDocument{xmlReadFile ( aFilename.c_str(), nullptr, 0 ) },
        mCanvas{aWidth, aHeight}
    {
        if ( mDocument == nullptr )
        {
            throw std::runtime_error ( "Failed to parse file." );
        }
    }

    void Window::ResizeViewport ( uint32_t aWidth, uint32_t aHeight )
    {
        mCanvas.ResizeViewport ( aWidth, aHeight );
    }

    const uint8_t* Window::GetPixels() const
    {
        return mCanvas.GetPixels();
    }

    size_t Window::GetWidth() const
    {
        return mCanvas.GetWidth();
    }
    size_t Window::GetHeight() const
    {
        return mCanvas.GetHeight();
    }
    size_t Window::GetStride() const
    {
        return mCanvas.GetStride();
    }

    static void Render ( Canvas* aCanvas, xmlNodePtr aNode )
    {
        xmlNode *cur_node = nullptr;
        for ( cur_node = aNode; cur_node; cur_node = cur_node->next )
        {
            if ( cur_node->type == XML_ELEMENT_NODE )
            {
                std::cout << "Element: " << cur_node->name << std::endl;
                switch ( GetElementNameEnum ( reinterpret_cast<const std::string::value_type*> ( cur_node->name ) ) )
                {
                case unknown:
                    std::cout << "Unknown Element: " << cur_node->name << std::endl;
                    break;
                case svg:
                    break;
                case g:
                    break;
                case path:
                    break;
                case rect:
                    break;
                case line:
                    break;
                case polyline:
                    break;
                case polygon:
                    break;
                case circle:
                    break;
                case ellipse:
                    break;
                }
            }
            Render ( aCanvas, cur_node->children );
        }
    }

    void Window::Render()
    {
        mCanvas.Clear();
        AeonGUI::Render ( &mCanvas, xmlDocGetRootElement ( mDocument ) );
    }
    Window::~Window()
    {
        if ( mDocument != nullptr )
        {
            xmlFreeDoc ( mDocument );
        }
    }
}
