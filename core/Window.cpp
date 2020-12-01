/*
Copyright (C) 2019,2020 Rodrigo Jose Hernandez Cordoba

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
#include <libxml/tree.h>
#include <libxml/parser.h>

#include "dom/Element.h"
#include "aeongui/Window.h"

namespace AeonGUI
{
    Window::Window () : mJavaScript{this} {}
    Window::Window ( const std::string aFilename, uint32_t aWidth, uint32_t aHeight ) :
        mJavaScript{this},
        mCanvas{aWidth, aHeight}
    {
        mJavaScript.SetLocation ( aFilename );
    }

    Window::~Window() = default;

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

    void Window::Draw()
    {
        mCanvas.Clear();
        // Only draw if document_element is valid.
        if ( Element* document_element = mJavaScript.GetDocumentElement() )
        {
            document_element->TraverseDepthFirstPreOrder (
                [this] ( const Node * aNode )
            {
                aNode->DrawStart ( mCanvas );
            },
            [this] ( const Node * aNode )
            {
                aNode->DrawFinish ( mCanvas );
            },
            [] ( const Node * aNode )
            {
                return aNode->IsDrawEnabled();
            } );
        }
    }
}
