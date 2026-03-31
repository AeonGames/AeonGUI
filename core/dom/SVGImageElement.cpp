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
#include "aeongui/dom/SVGImageElement.hpp"
#include "aeongui/Canvas.hpp"
#include "aeongui/dom/Document.hpp"
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <filesystem>
#include <libcss/libcss.h>

namespace AeonGUI
{
    namespace DOM
    {
        static void ParseLengthAttribute ( const AttributeMap& aAttributes, const char* aKey, SVGAnimatedLength& aLength, float aDefault )
        {
            auto it = aAttributes.find ( aKey );
            if ( it == aAttributes.end() )
            {
                aLength.baseVal().newValueSpecifiedUnits ( SVGLengthType::NUMBER, aDefault );
                return;
            }

            try
            {
                aLength.baseVal().valueAsString ( it->second );
            }
            catch ( const std::exception& )
            {
                aLength.baseVal().newValueSpecifiedUnits ( SVGLengthType::NUMBER, aDefault );
            }
        }

        static bool HasUriScheme ( const DOMString& aValue )
        {
            // RFC 3986 scheme production: ALPHA *( ALPHA / DIGIT / "+" / "-" / "." ) ':'
            if ( aValue.empty() || !std::isalpha ( static_cast<unsigned char> ( aValue[0] ) ) )
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

        static std::filesystem::path DocumentPathFromUrl ( const DOMString& aDocumentUrl )
        {
            if ( aDocumentUrl.rfind ( "file://", 0 ) == 0 )
            {
                DOMString path = aDocumentUrl.substr ( 7 );
                if ( path.size() > 2 && path[0] == '/' && std::isalpha ( static_cast<unsigned char> ( path[1] ) ) && path[2] == ':' )
                {
                    path.erase ( 0, 1 );
                }
                return std::filesystem::path ( path );
            }
            return std::filesystem::path ( aDocumentUrl );
        }

        SVGImageElement::SVGImageElement ( const DOMString& aTagName, AttributeMap&& aAttributes, Node* aParent ) :
            SVGGraphicsElement ( aTagName, std::move ( aAttributes ), aParent )
        {
            ParseAttributes ( mAttributes );
        }

        SVGImageElement::~SVGImageElement() = default;

        void SVGImageElement::onAttributeChanged ( const DOMString& aName, const DOMString& aValue )
        {
            Element::onAttributeChanged ( aName, aValue );
            ParseAttributes ( mAttributes );
            if ( aName == "href" || aName == "xlink:href" )
            {
                mImageLoadAttempted = false;
                mRasterImage = RasterImage{};
            }
        }

        void SVGImageElement::ParseAttributes ( const AttributeMap& aAttributes )
        {
            ParseLengthAttribute ( aAttributes, "x", mX, 0.0f );
            ParseLengthAttribute ( aAttributes, "y", mY, 0.0f );
            ParseLengthAttribute ( aAttributes, "width", mWidth, 0.0f );
            ParseLengthAttribute ( aAttributes, "height", mHeight, 0.0f );

            auto hrefIt = aAttributes.find ( "href" );
            if ( hrefIt == aAttributes.end() )
            {
                hrefIt = aAttributes.find ( "xlink:href" );
            }
            if ( hrefIt != aAttributes.end() )
            {
                mHref.baseVal() = hrefIt->second;
                mHref.animVal() = hrefIt->second;
            }

            auto parIt = aAttributes.find ( "preserveAspectRatio" );
            if ( parIt != aAttributes.end() )
            {
                mPreserveAspectRatio.baseVal() = PreserveAspectRatio{parIt->second};
                mPreserveAspectRatio.animVal() = mPreserveAspectRatio.baseVal();
            }

            auto crossOriginIt = aAttributes.find ( "crossorigin" );
            if ( crossOriginIt != aAttributes.end() )
            {
                mCrossOrigin = crossOriginIt->second;
            }

            auto decodingIt = aAttributes.find ( "decoding" );
            if ( decodingIt != aAttributes.end() )
            {
                mDecoding = decodingIt->second;
            }
        }

        bool SVGImageElement::EnsureImageLoaded() const
        {
            if ( mRasterImage.IsLoaded() )
            {
                return true;
            }
            if ( mImageLoadAttempted )
            {
                return false;
            }

            mImageLoadAttempted = true;
            if ( mHref.baseVal().empty() )
            {
                return false;
            }

            DOMString resolvedPath = mHref.baseVal();
            if ( !HasUriScheme ( resolvedPath ) )
            {
                std::filesystem::path hrefPath{resolvedPath};
                if ( !hrefPath.is_absolute() )
                {
                    const auto* document = ownerDocument();
                    if ( document != nullptr )
                    {
                        const std::filesystem::path documentPath = DocumentPathFromUrl ( document->url() );
                        if ( !documentPath.empty() )
                        {
                            resolvedPath = ( documentPath.parent_path() / hrefPath ).lexically_normal().string();
                        }
                    }
                }

                // Fragment identifiers are not part of a filesystem path.
                const size_t fragmentPos = resolvedPath.find ( '#' );
                if ( fragmentPos != DOMString::npos )
                {
                    resolvedPath.erase ( fragmentPos );
                }
            }

            if ( resolvedPath.empty() )
            {
                std::fprintf ( stderr, "SVGImageElement: empty image path after href normalization from '%s'.\n", mHref.baseVal().c_str() );
                return false;
            }

            if ( !mRasterImage.LoadFromFile ( resolvedPath ) )
            {
                std::fprintf ( stderr, "SVGImageElement: failed to load image '%s' (resolved to '%s').\n", mHref.baseVal().c_str(), resolvedPath.c_str() );
                return false;
            }
            return true;
        }

        const SVGAnimatedLength& SVGImageElement::x() const
        {
            return mX;
        }

        const SVGAnimatedLength& SVGImageElement::y() const
        {
            return mY;
        }

        const SVGAnimatedLength& SVGImageElement::width() const
        {
            return mWidth;
        }

        const SVGAnimatedLength& SVGImageElement::height() const
        {
            return mHeight;
        }

        const SVGAnimatedString& SVGImageElement::href() const
        {
            return mHref;
        }

        const SVGAnimatedPreserveAspectRatio& SVGImageElement::preserveAspectRatio() const
        {
            return mPreserveAspectRatio;
        }

        const DOMString& SVGImageElement::crossOrigin() const
        {
            return mCrossOrigin;
        }

        const DOMString& SVGImageElement::decoding() const
        {
            return mDecoding;
        }

        void SVGImageElement::DrawStart ( Canvas& aCanvas ) const
        {
            SVGGraphicsElement::DrawStart ( aCanvas );
            if ( !EnsureImageLoaded() )
            {
                return;
            }

            double xPos = static_cast<double> ( mX.baseVal().value() );
            double yPos = static_cast<double> ( mY.baseVal().value() );
            double drawWidth = static_cast<double> ( mWidth.baseVal().value() );
            double drawHeight = static_cast<double> ( mHeight.baseVal().value() );
            if ( drawWidth <= 0.0 || drawHeight <= 0.0 )
            {
                return;
            }

            const double sourceWidth = static_cast<double> ( mRasterImage.GetWidth() );
            const double sourceHeight = static_cast<double> ( mRasterImage.GetHeight() );
            if ( sourceWidth <= 0.0 || sourceHeight <= 0.0 )
            {
                return;
            }

            if ( mPreserveAspectRatio.baseVal().GetAlign() != PreserveAspectRatio::Align::none )
            {
                double scaleX = drawWidth / sourceWidth;
                double scaleY = drawHeight / sourceHeight;
                double scale = scaleX;
                if ( mPreserveAspectRatio.baseVal().GetMeetOrSlice() == PreserveAspectRatio::MeetOrSlice::Meet )
                {
                    scale = std::min ( scaleX, scaleY );
                }
                else
                {
                    scale = std::max ( scaleX, scaleY );
                }

                const double contentWidth = sourceWidth * scale;
                const double contentHeight = sourceHeight * scale;
                const double remainingX = drawWidth - contentWidth;
                const double remainingY = drawHeight - contentHeight;

                switch ( mPreserveAspectRatio.baseVal().GetAlignX() )
                {
                case PreserveAspectRatio::MinMidMax::Mid:
                    xPos += remainingX * 0.5;
                    break;
                case PreserveAspectRatio::MinMidMax::Max:
                    xPos += remainingX;
                    break;
                case PreserveAspectRatio::MinMidMax::Min:
                default:
                    break;
                }

                switch ( mPreserveAspectRatio.baseVal().GetAlignY() )
                {
                case PreserveAspectRatio::MinMidMax::Mid:
                    yPos += remainingY * 0.5;
                    break;
                case PreserveAspectRatio::MinMidMax::Max:
                    yPos += remainingY;
                    break;
                case PreserveAspectRatio::MinMidMax::Min:
                default:
                    break;
                }

                drawWidth = contentWidth;
                drawHeight = contentHeight;
            }

            double opacity = 1.0;
            css_select_results* results{ GetComputedStyles() };
            if ( results && results->styles[CSS_PSEUDO_ELEMENT_NONE] )
            {
                css_fixed fixed{};
                css_computed_opacity ( results->styles[CSS_PSEUDO_ELEMENT_NONE], &fixed );
                opacity = std::clamp<double> ( FIXTOFLT ( fixed ), 0.0, 1.0 );
            }

            aCanvas.DrawImage ( mRasterImage.GetPixels(),
                                mRasterImage.GetWidth(),
                                mRasterImage.GetHeight(),
                                mRasterImage.GetStride(),
                                xPos,
                                yPos,
                                drawWidth,
                                drawHeight,
                                opacity );
        }
    }
}
