/*
Copyright (C) 2019,2020,2023-2025,2026 Rodrigo Jose Hernandez Cordoba

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
#include <regex>
#include "aeongui/Canvas.hpp"
#include "aeongui/Matrix2x3.hpp"
#include "aeongui/dom/SVGSVGElement.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        SVGSVGElement::SVGSVGElement ( const std::string& aTagName, AttributeMap&& aAttributes, Node* aParent ) :
            SVGGraphicsElement { aTagName, std::move ( aAttributes ), aParent }
        {
            ParseAttributes();
        }

        SVGSVGElement::~SVGSVGElement() = default;

        void SVGSVGElement::ParseAttributes()
        {
            mHasViewBox = false;
            if ( mAttributes.find ( "viewBox" ) != mAttributes.end() )
            {
                static const std::regex viewBoxRegex{R"((-?(?:[0-9]*\.[0-9]+(?:[eE][-+]?[0-9]+)?|[0-9]+))[[:space:]]+(-?(?:[0-9]*\.[0-9]+(?:[eE][-+]?[0-9]+)?|[0-9]+))[[:space:]]+(-?(?:[0-9]*\.[0-9]+(?:[eE][-+]?[0-9]+)?|[0-9]+))[[:space:]]+(-?(?:[0-9]*\.[0-9]+(?:[eE][-+]?[0-9]+)?|[0-9]+)))"};

                const std::string& viewBoxStr{mAttributes.at ( "viewBox" ) };
                std::smatch match{};
                if ( std::regex_match ( viewBoxStr, match, viewBoxRegex ) )
                {
                    mViewBox.width  = std::stod ( match[3] );
                    mViewBox.height = std::stod ( match[4] );
                    mViewBox.min_x      = std::stod ( match[1] );
                    mViewBox.min_y      = std::stod ( match[2] );
                    mHasViewBox = true;
                }
            }
            if ( mAttributes.find ( "width" ) != mAttributes.end() )
            {
                size_t pos{};
                const auto& val = mAttributes.at ( "width" );
                mWidthRaw = std::stod ( val, &pos );
                mWidthPct = ( pos < val.size() && val[pos] == '%' );
                mWidth = mWidthRaw;
            }
            if ( mAttributes.find ( "height" ) != mAttributes.end() )
            {
                const auto& val = mAttributes.at ( "height" );
                size_t pos{};
                mHeightRaw = std::stod ( val, &pos );
                mHeightPct = ( pos < val.size() && val[pos] == '%' );
                mHeight = mHeightRaw;
            }
            if ( mAttributes.find ( "preserveAspectRatio" ) != mAttributes.end() )
            {
                mPreserveAspectRatio = PreserveAspectRatio{mAttributes.at ( "preserveAspectRatio" ) };
            }
        }

        void SVGSVGElement::onAttributeChanged ( const DOMString& aName, const DOMString& aValue )
        {
            Element::onAttributeChanged ( aName, aValue );
            if ( aName == "width" || aName == "height" || aName == "viewBox" || aName == "preserveAspectRatio" )
            {
                ParseAttributes();
            }
        }

        void SVGSVGElement::DrawStart ( Canvas& aCanvas ) const
        {
            SVGGraphicsElement::DrawStart ( aCanvas );

            // Inline-in-HTML mode: when the HTML layout engine has
            // assigned a non-empty box on this <svg>, build the
            // viewport-to-device transform manually so the SVG paints
            // inside its laid-out rectangle.  We avoid Canvas::
            // SetViewBox here because it bakes in the canvas-wide
            // dimensions and would clobber any preceding translate.
            const bool inline_mode =
                mInlineLayoutBox.width  > 0.0 &&
                mInlineLayoutBox.height > 0.0;

            if ( inline_mode )
            {
                if ( mHasViewBox && mViewBox.width > 0.0 && mViewBox.height > 0.0 )
                {
                    double scale_x = mInlineLayoutBox.width  / mViewBox.width;
                    double scale_y = mInlineLayoutBox.height / mViewBox.height;
                    if ( mPreserveAspectRatio.GetAlign() != PreserveAspectRatio::Align::none )
                    {
                        if ( mPreserveAspectRatio.GetMeetOrSlice() ==
                             PreserveAspectRatio::MeetOrSlice::Meet )
                        {
                            scale_x = std::min ( scale_x, scale_y );
                            scale_y = scale_x;
                        }
                        else if ( mPreserveAspectRatio.GetMeetOrSlice() ==
                                  PreserveAspectRatio::MeetOrSlice::Slice )
                        {
                            scale_x = std::max ( scale_x, scale_y );
                            scale_y = scale_x;
                        }
                    }
                    const double tx = mInlineLayoutBox.x - mViewBox.min_x * scale_x;
                    const double ty = mInlineLayoutBox.y - mViewBox.min_y * scale_y;
                    aCanvas.SetTransform ( Matrix2x3 (
                                               scale_x, 0.0,
                                               0.0,     scale_y,
                                               tx,      ty ) );
                    aCanvas.PushViewport ( mViewBox.width, mViewBox.height );
                }
                else
                {
                    aCanvas.SetTransform ( Matrix2x3 (
                                               1.0, 0.0,
                                               0.0, 1.0,
                                               mInlineLayoutBox.x,
                                               mInlineLayoutBox.y ) );
                    aCanvas.PushViewport ( mInlineLayoutBox.width,
                                           mInlineLayoutBox.height );
                }
                return;
            }

            const double canvasW = static_cast<double> ( aCanvas.GetWidth() );
            const double canvasH = static_cast<double> ( aCanvas.GetHeight() );
            if ( mHasViewBox )
            {
                aCanvas.SetViewBox ( mViewBox, mPreserveAspectRatio );
                aCanvas.PushViewport ( mViewBox.width, mViewBox.height );
            }
            else
            {
                double vw = mWidthPct  ? ( mWidthRaw  * canvasW / 100.0 ) :
                            ( mWidth  > 0.0 ? mWidth  : canvasW );
                double vh = mHeightPct ? ( mHeightRaw * canvasH / 100.0 ) :
                            ( mHeight > 0.0 ? mHeight : canvasH );
                aCanvas.PushViewport ( vw, vh );
            }
        }

        void SVGSVGElement::DrawFinish ( Canvas& aCanvas ) const
        {
            aCanvas.PopViewport();
            SVGGraphicsElement::DrawFinish ( aCanvas );
        }
    }
}
