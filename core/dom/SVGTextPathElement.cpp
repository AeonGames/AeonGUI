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
#include "aeongui/dom/SVGTextPathElement.hpp"
#include "aeongui/dom/SVGGeometryElement.hpp"
#include "aeongui/dom/Document.hpp"
#include "aeongui/Canvas.hpp"
#include "aeongui/StyleSheet.hpp"
#include <libcss/libcss.h>
#include <string>

namespace AeonGUI
{
    namespace DOM
    {
        SVGTextPathElement::SVGTextPathElement ( const std::string& aTagName, AttributeMap&& aAttributes, Node* aParent ) :
            SVGTextContentElement { aTagName, std::move ( aAttributes ), aParent }
        {
            ParseAttributes();
        }

        SVGTextPathElement::~SVGTextPathElement() = default;

        void SVGTextPathElement::ParseAttributes()
        {
            // Parse href or xlink:href attribute.
            auto hrefIt = mAttributes.find ( "href" );
            if ( hrefIt == mAttributes.end() )
            {
                hrefIt = mAttributes.find ( "xlink:href" );
            }
            if ( hrefIt != mAttributes.end() )
            {
                mHref.baseVal() = hrefIt->second;
                mHref.animVal() = hrefIt->second;
            }

            // Parse startOffset attribute.
            auto offsetIt = mAttributes.find ( "startOffset" );
            if ( offsetIt != mAttributes.end() )
            {
                const std::string& val = offsetIt->second;
                if ( !val.empty() )
                {
                    // Check for percentage — will be resolved at draw time.
                    if ( val.back() == '%' )
                    {
                        // Store as negative to mark percentage; resolve at draw time.
                        double pct = std::stod ( val.substr ( 0, val.size() - 1 ) );
                        mStartOffset.baseVal().value ( static_cast<float> ( -pct ) );
                    }
                    else
                    {
                        mStartOffset.baseVal().value ( static_cast<float> ( std::stod ( val ) ) );
                    }
                }
            }

            // Parse side attribute (SVG2 §11.5.6.2).
            auto sideIt = mAttributes.find ( "side" );
            mSideRight = ( sideIt != mAttributes.end() && sideIt->second == "right" );

            // Parse text-anchor (SVG presentation attribute, not in libcss).
            mTextAnchor = 0;
            auto anchorIt = mAttributes.find ( "text-anchor" );
            if ( anchorIt != mAttributes.end() )
            {
                if ( anchorIt->second == "middle" )
                {
                    mTextAnchor = 1;
                }
                else if ( anchorIt->second == "end" )
                {
                    mTextAnchor = 2;
                }
            }
        }

        void SVGTextPathElement::onAttributeChanged ( const DOMString& aName, const DOMString& aValue )
        {
            Element::onAttributeChanged ( aName, aValue );
            if ( aName == "href" || aName == "xlink:href" || aName == "startOffset" ||
                 aName == "side" || aName == "text-anchor" )
            {
                ParseAttributes();
            }
        }

        const SVGAnimatedString& SVGTextPathElement::href() const
        {
            return mHref;
        }
        const SVGAnimatedLength& SVGTextPathElement::startOffset() const
        {
            return mStartOffset;
        }
        const SVGAnimatedEnumeration& SVGTextPathElement::method() const
        {
            return mMethod;
        }
        const SVGAnimatedEnumeration& SVGTextPathElement::spacing() const
        {
            return mSpacing;
        }

        void SVGTextPathElement::DrawStart ( Canvas& aCanvas ) const
        {
            SVGTextContentElement::DrawStart ( aCanvas );

            // Resolve the referenced path element by ID.
            const std::string& hrefVal = mHref.baseVal();
            if ( hrefVal.empty() )
            {
                return;
            }

            // Walk up to the Document root.
            const Node* node = this;
            while ( node->parentNode() )
            {
                node = node->parentNode();
            }
            const Document* document = static_cast<const Document*> ( node );

            // Use the DOM spec getElementById to find the referenced element.
            // Strip leading '#' from the href fragment.
            Element* foundElem = document->getElementById (
                                     hrefVal[0] == '#' ? hrefVal.substr ( 1 ) : hrefVal );
            const SVGGeometryElement* pathElement = foundElem
                                                    ? dynamic_cast<const SVGGeometryElement*> ( foundElem )
                                                    : nullptr;

            if ( !pathElement )
            {
                return;
            }

            // Get CSS styles — prefer our own, fall back to parent text element.
            css_select_results* results{ GetComputedStyles() };
            if ( !results || !results->styles[CSS_PSEUDO_ELEMENT_NONE] )
            {
                return;
            }
            css_computed_style* style = results->styles[CSS_PSEUDO_ELEMENT_NONE];

            ApplyCSSPaintProperties ( aCanvas, *this, style );

            // Font properties from CSS.
            std::string fontFamily = GetCSSFontFamily ( style );
            double fontSize = GetCSSFontSize ( style );
            int fontWeight = GetCSSFontWeight ( style );
            int fontStyle = GetCSSFontStyle ( style );

            // Compute startOffset.
            double startOffset = 0.0;
            double totalLength = pathElement->GetPath().GetTotalLength();
            float rawOffset = mStartOffset.baseVal().value();
            if ( rawOffset < 0.0f )
            {
                // Negative means percentage (our encoding).
                startOffset = totalLength * ( -rawOffset / 100.0 );
            }
            else
            {
                // Check for pathLength on the referenced element to scale the offset.
                auto plIt = pathElement->attributes().find ( "pathLength" );
                if ( plIt != pathElement->attributes().end() && !plIt->second.empty() )
                {
                    double authorLength = std::stod ( plIt->second );
                    if ( authorLength > 0.0 )
                    {
                        startOffset = static_cast<double> ( rawOffset ) * ( totalLength / authorLength );
                    }
                }
                else
                {
                    startOffset = static_cast<double> ( rawOffset );
                }
            }

            // text-anchor adjustment: shift startOffset by measured text width.
            // Also check the parent <text> element's text-anchor if not set on us.
            int anchor = mTextAnchor;
            if ( anchor == 0 && parentNode() )
            {
                const Element* parentElem = dynamic_cast<const Element*> ( parentNode() );
                if ( parentElem )
                {
                    auto pAnchorIt = parentElem->attributes().find ( "text-anchor" );
                    if ( pAnchorIt != parentElem->attributes().end() )
                    {
                        if ( pAnchorIt->second == "middle" )
                        {
                            anchor = 1;
                        }
                        else if ( pAnchorIt->second == "end" )
                        {
                            anchor = 2;
                        }
                    }
                }
            }

            // Collect text content from child text nodes.
            DOMString tc = textContent();

            if ( tc.empty() )
            {
                return;
            }

            if ( anchor != 0 )
            {
                double textWidth = aCanvas.MeasureText ( tc, fontFamily, fontSize, fontWeight, fontStyle );
                if ( anchor == 1 )
                {
                    startOffset -= textWidth * 0.5;
                }
                else if ( anchor == 2 )
                {
                    startOffset -= textWidth;
                }
            }

            bool isClosed = pathElement->GetPath().IsClosed();

            aCanvas.DrawTextOnPath ( tc, pathElement->GetPath(), startOffset,
                                     fontFamily, fontSize, fontWeight, fontStyle,
                                     mSideRight, isClosed );
        }
    }
}
