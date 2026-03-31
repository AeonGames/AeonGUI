/*
Copyright (C) 2023-2026 Rodrigo Jose Hernandez Cordoba

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
#include "aeongui/StyleSheet.hpp"
#include "aeongui/Canvas.hpp"
#include "aeongui/Color.hpp"
#include "aeongui/dom/Document.hpp"
#include <algorithm>
#include <iostream>
#include <string>
#include <libcss/libcss.h>

namespace AeonGUI
{
    void css_stylesheet_deleter::operator() ( css_stylesheet* p )
    {
        css_error code{ css_stylesheet_destroy ( p ) };
        if ( code != CSS_OK )
        {
            std::cerr << "css_stylesheet_destroy failed with code: " << code << std::endl;
        }
    }

    void css_select_ctx_deleter::operator() ( css_select_ctx* p )
    {
        css_error code{ css_select_ctx_destroy ( p ) };
        if ( code != CSS_OK )
        {
            std::cerr << "css_select_ctx_destroy failed with code: " << code << std::endl;
        }
    }

    void css_select_results_deleter::operator() ( css_select_results* p )
    {
        css_error code{ css_select_results_destroy ( p ) };
        if ( code != CSS_OK )
        {
            std::cerr << "css_select_results_destroy failed with code: " << code << std::endl;
        }
    }

    void css_computed_style_deleter::operator() ( css_computed_style* p )
    {
        css_error code{ css_computed_style_destroy ( p ) };
        if ( code != CSS_OK )
        {
            std::cerr << "css_computed_style_destroy failed with code: " << code << std::endl;
        }
    }

    std::string GetCSSFontFamily ( css_computed_style* aStyle )
    {
        lwc_string** names = nullptr;
        uint8_t family = css_computed_font_family ( aStyle, &names );
        if ( names && names[0] )
        {
            return std::string ( lwc_string_data ( names[0] ), lwc_string_length ( names[0] ) );
        }
        switch ( family )
        {
        case CSS_FONT_FAMILY_SERIF:
            return "serif";
        case CSS_FONT_FAMILY_MONOSPACE:
            return "monospace";
        case CSS_FONT_FAMILY_CURSIVE:
            return "cursive";
        case CSS_FONT_FAMILY_FANTASY:
            return "fantasy";
        case CSS_FONT_FAMILY_SANS_SERIF:
        default:
            return "sans-serif";
        }
    }

    double GetCSSFontSize ( css_computed_style* aStyle )
    {
        css_fixed length{};
        css_unit unit{};
        uint8_t sizeType = css_computed_font_size ( aStyle, &length, &unit );
        if ( sizeType == CSS_FONT_SIZE_DIMENSION )
        {
            return FIXTOFLT ( length );
        }
        return 16.0;
    }

    int GetCSSFontWeight ( css_computed_style* aStyle )
    {
        uint8_t w = css_computed_font_weight ( aStyle );
        switch ( w )
        {
        case CSS_FONT_WEIGHT_100:
            return 100;
        case CSS_FONT_WEIGHT_200:
            return 200;
        case CSS_FONT_WEIGHT_300:
            return 300;
        case CSS_FONT_WEIGHT_400:
        case CSS_FONT_WEIGHT_NORMAL:
            return 400;
        case CSS_FONT_WEIGHT_500:
            return 500;
        case CSS_FONT_WEIGHT_600:
            return 600;
        case CSS_FONT_WEIGHT_700:
        case CSS_FONT_WEIGHT_BOLD:
            return 700;
        case CSS_FONT_WEIGHT_800:
            return 800;
        case CSS_FONT_WEIGHT_900:
            return 900;
        default:
            return 400;
        }
    }

    int GetCSSFontStyle ( css_computed_style* aStyle )
    {
        uint8_t s = css_computed_font_style ( aStyle );
        switch ( s )
        {
        case CSS_FONT_STYLE_ITALIC:
            return 1;
        case CSS_FONT_STYLE_OBLIQUE:
            return 2;
        default:
            return 0;
        }
    }

    static bool ResolvePaintUri ( Canvas& aCanvas, const DOM::Element& aElement,
                                  lwc_string* aUri, bool aIsFill )
    {
        if ( !aUri )
        {
            return false;
        }
        std::string uri_str ( lwc_string_data ( aUri ), lwc_string_length ( aUri ) );
        // Strip leading '#' for fragment reference
        if ( uri_str.empty() || uri_str[0] != '#' )
        {
            return false;
        }
        std::string id = uri_str.substr ( 1 );
        DOM::Document* doc = aElement.ownerDocument();
        if ( !doc )
        {
            return false;
        }
        DOM::Element* paintServer = doc->getElementById ( id );
        if ( !paintServer || paintServer->tagName() != "linearGradient" )
        {
            return false;
        }
        const auto& gradAttrs = paintServer->attributes();
        auto getAttr = [&gradAttrs] ( const std::string & name, double fallback ) -> double
        {
            auto a = gradAttrs.find ( name );
            if ( a != gradAttrs.end() )
            {
                char* end{};
                double v = strtod ( a->second.c_str(), &end );
                if ( end != a->second.c_str() )
                {
                    return v;
                }
            }
            return fallback;
        };
        LinearGradient gradient;
        gradient.x1 = getAttr ( "x1", 0.0 );
        gradient.y1 = getAttr ( "y1", 0.0 );
        gradient.x2 = getAttr ( "x2", 1.0 );
        gradient.y2 = getAttr ( "y2", 0.0 );
        auto unitsIt = gradAttrs.find ( "gradientUnits" );
        gradient.objectBoundingBox = ( unitsIt == gradAttrs.end() || unitsIt->second == "objectBoundingBox" );

        for ( auto& child : paintServer->childNodes() )
        {
            if ( child->nodeType() != DOM::Node::ELEMENT_NODE )
            {
                continue;
            }
            DOM::Element* stopElem = static_cast<DOM::Element*> ( child.get() );
            if ( stopElem->tagName() != "stop" )
            {
                continue;
            }
            const auto& stopAttrs = stopElem->attributes();
            double offset = 0.0;
            auto offsetIt = stopAttrs.find ( "offset" );
            if ( offsetIt != stopAttrs.end() )
            {
                char* end{};
                double v = strtod ( offsetIt->second.c_str(), &end );
                if ( end != offsetIt->second.c_str() )
                {
                    if ( *end == '%' )
                    {
                        v /= 100.0;
                    }
                    offset = v;
                }
            }
            Color stopColor{0xffffffff};
            auto colorIt = stopAttrs.find ( "stop-color" );
            if ( colorIt != stopAttrs.end() )
            {
                uint32_t cv{};
                if ( Color::IsColor ( colorIt->second, &cv ) )
                {
                    stopColor = Color{cv};
                }
            }
            auto opacityIt = stopAttrs.find ( "stop-opacity" );
            if ( opacityIt != stopAttrs.end() )
            {
                char* end{};
                double v = strtod ( opacityIt->second.c_str(), &end );
                if ( end != opacityIt->second.c_str() )
                {
                    stopColor.a = static_cast<uint8_t> ( std::clamp ( v, 0.0, 1.0 ) * 255.0 );
                }
            }
            gradient.stops.push_back ( { offset, stopColor } );
        }
        if ( aIsFill )
        {
            aCanvas.SetFillColor ( gradient );
        }
        else
        {
            aCanvas.SetStrokeColor ( gradient );
        }
        return true;
    }

    void ApplyCSSPaintProperties ( Canvas& aCanvas, const DOM::Element& aElement, css_computed_style* aStyle )
    {
        css_color color{};
        lwc_string *uri{};
        css_fixed fixed{};
        css_unit unit{};
        uint8_t fill_type = css_computed_fill ( aStyle, &color, &uri );
        if ( fill_type == CSS_PAINT_URI )
        {
            if ( !ResolvePaintUri ( aCanvas, aElement, uri, true ) )
            {
                aCanvas.SetFillColor ( none{} );
            }
            css_computed_fill_opacity ( aStyle, &fixed );
            aCanvas.SetFillOpacity ( FIXTOFLT ( fixed ) );
        }
        else if ( fill_type != CSS_PAINT_NONE )
        {
            aCanvas.SetFillColor ( Color{color} );
            css_computed_fill_opacity ( aStyle, &fixed );
            aCanvas.SetFillOpacity ( FIXTOFLT ( fixed ) );
        }
        else
        {
            aCanvas.SetFillColor ( none{} );
        }
        uint8_t stroke_type = css_computed_stroke ( aStyle, &color, &uri );
        if ( stroke_type == CSS_PAINT_URI )
        {
            if ( !ResolvePaintUri ( aCanvas, aElement, uri, false ) )
            {
                aCanvas.SetStrokeColor ( none{} );
            }
            css_computed_stroke_opacity ( aStyle, &fixed );
            aCanvas.SetStrokeOpacity ( FIXTOFLT ( fixed ) );
            css_computed_stroke_width ( aStyle, &fixed, &unit );
            aCanvas.SetStrokeWidth ( FIXTOFLT ( fixed ) );
        }
        else if ( stroke_type != CSS_PAINT_NONE && stroke_type != CSS_PAINT_INHERIT )
        {
            aCanvas.SetStrokeColor ( Color{color} );
            css_computed_stroke_opacity ( aStyle, &fixed );
            aCanvas.SetStrokeOpacity ( FIXTOFLT ( fixed ) );
            css_computed_stroke_width ( aStyle, &fixed, &unit );
            aCanvas.SetStrokeWidth ( FIXTOFLT ( fixed ) );
        }
        else
        {
            aCanvas.SetStrokeColor ( none{} );
        }
        css_computed_opacity ( aStyle, &fixed );
        aCanvas.SetOpacity ( FIXTOFLT ( fixed ) );
    }
}
