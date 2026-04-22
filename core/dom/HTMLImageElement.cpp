/*
Copyright (C) 2026 Rodrigo Jose Hernandez Cordoba

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
#include <cctype>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include "aeongui/Canvas.hpp"
#include "aeongui/LogLevel.hpp"
#include "aeongui/dom/Document.hpp"
#include "aeongui/dom/HTMLImageElement.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        namespace
        {
            /// Mirrors SVGImageElement's URI-scheme detection — keeps
            /// HTML and SVG image loading in lockstep so authors get
            /// the same path-resolution rules in both contexts.
            bool HasUriScheme ( const DOMString& aValue )
            {
                if ( aValue.empty() ||
                     !std::isalpha ( static_cast<unsigned char> ( aValue[0] ) ) )
                {
                    return false;
                }
                for ( size_t i = 1; i < aValue.size(); ++i )
                {
                    const unsigned char c = static_cast<unsigned char> ( aValue[i] );
                    if ( c == ':' )
                    {
                        return true;
                    }
                    if ( !std::isalnum ( c ) && c != '+' && c != '-' && c != '.' )
                    {
                        return false;
                    }
                }
                return false;
            }

            std::filesystem::path DocumentPathFromUrl ( const DOMString& aDocumentUrl )
            {
                if ( aDocumentUrl.rfind ( "file://", 0 ) == 0 )
                {
                    DOMString path = aDocumentUrl.substr ( 7 );
                    // Strip the leading slash from "/C:/..." on Windows.
                    if ( path.size() > 2 && path[0] == '/' &&
                         std::isalpha ( static_cast<unsigned char> ( path[1] ) ) &&
                         path[2] == ':' )
                    {
                        path.erase ( 0, 1 );
                    }
                    return std::filesystem::path ( path );
                }
                return std::filesystem::path ( aDocumentUrl );
            }
        }

        HTMLImageElement::HTMLImageElement ( const DOMString& aTagName, AttributeMap&& aAttributes, Node* aParent )
            : HTMLElement { aTagName, std::move ( aAttributes ), aParent }
        {
            auto srcIt = mAttributes.find ( "src" );
            if ( srcIt != mAttributes.end() )
            {
                mSrc = srcIt->second;
            }
        }

        HTMLImageElement::~HTMLImageElement() = default;

        void HTMLImageElement::onAttributeChanged ( const DOMString& aName, const DOMString& aValue )
        {
            HTMLElement::onAttributeChanged ( aName, aValue );
            if ( aName == "src" )
            {
                mSrc = aValue;
                mRasterImage = RasterImage{};
                mImageLoadAttempted = false;
            }
        }

        bool HTMLImageElement::EnsureImageLoaded() const
        {
            if ( mRasterImage.IsLoaded() )
            {
                return true;
            }
            if ( mImageLoadAttempted || mSrc.empty() )
            {
                return false;
            }
            mImageLoadAttempted = true;

            DOMString resolved = mSrc;
            if ( !HasUriScheme ( resolved ) )
            {
                std::filesystem::path srcPath{resolved};
                if ( !srcPath.is_absolute() )
                {
                    if ( const auto * doc = ownerDocument() )
                    {
                        const std::filesystem::path docPath = DocumentPathFromUrl ( doc->url() );
                        if ( !docPath.empty() )
                        {
                            resolved = ( docPath.parent_path() / srcPath ).lexically_normal().string();
                        }
                    }
                }
                const size_t fragmentPos = resolved.find ( '#' );
                if ( fragmentPos != DOMString::npos )
                {
                    resolved.erase ( fragmentPos );
                }
            }

            try
            {
                mRasterImage.LoadFromFile ( resolved );
            }
            catch ( const std::exception& e )
            {
                std::cerr << LogLevel::Error
                          << "HTMLImageElement: failed to load '"
                          << mSrc << "' (resolved to '" << resolved
                          << "'): " << e.what() << std::endl;
                return false;
            }
            return true;
        }

        uint32_t HTMLImageElement::naturalWidth() const
        {
            EnsureImageLoaded();
            return mRasterImage.IsLoaded() ? mRasterImage.GetWidth() : 0u;
        }

        uint32_t HTMLImageElement::naturalHeight() const
        {
            EnsureImageLoaded();
            return mRasterImage.IsLoaded() ? mRasterImage.GetHeight() : 0u;
        }

        void HTMLImageElement::DrawStart ( Canvas& aCanvas ) const
        {
            // Paint background and borders via the base class first
            // so a missing image still leaves a visible box (matching
            // browsers, which paint the placeholder background).
            HTMLElement::DrawStart ( aCanvas );

            const auto& box = GetLayoutBox();
            if ( box.contentWidth <= 0.0 || box.contentHeight <= 0.0 )
            {
                return;
            }
            if ( !EnsureImageLoaded() )
            {
                return;
            }
            aCanvas.DrawImage (
                mRasterImage.GetPixels(),
                mRasterImage.GetWidth(),
                mRasterImage.GetHeight(),
                mRasterImage.GetStride(),
                box.contentX,
                box.contentY,
                box.contentWidth,
                box.contentHeight,
                1.0 );
        }
    }
}
