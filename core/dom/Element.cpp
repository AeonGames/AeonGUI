/*
Copyright (C) 2010-2013,2019,2020,2023,2024 Rodrigo Jose Hernandez Cordoba

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
#include <string>
#include "Element.h"
#include "aeongui/Color.h"
#include "CSSSelectHandler.h"
#include <libcss/libcss.h>
namespace AeonGUI
{
    Element::Element ( const std::string& aTagName, const AttributeMap& aAttributes ) : mTagName{aTagName}, mAttributeMap{aAttributes}
    {
        auto style = mAttributeMap.find ( "style" );
        if ( style != mAttributeMap.end() )
        {
            css_error code{};
            css_stylesheet_params params{};
            params.params_version = CSS_STYLESHEET_PARAMS_VERSION_1;
            params.level = CSS_LEVEL_3;
            params.charset = "UTF-8";
            params.url = "";
            params.title = "Inline Style Sheet";
            params.allow_quirks = true;
            params.inline_style = true;
            params.resolve =
                [] ( void *pw,
                     const char *base, lwc_string * rel, lwc_string **abs ) -> css_error
            {
                ( void ) pw;
                ( void ) base;
                /* About as useless as possible */
                *abs = lwc_string_ref ( rel );
                return CSS_OK;
            };
            params.resolve_pw = nullptr;
            params.import = nullptr;
            params.import_pw = nullptr;
            params.color = nullptr;
            params.color_pw = nullptr;
            params.font = nullptr;
            params.font_pw = nullptr;
            {
                css_stylesheet* stylesheet{};
                code = css_stylesheet_create ( &params, &stylesheet );
                if ( code != CSS_OK )
                {
                    throw std::runtime_error ( css_error_to_string ( code ) );
                }
                mInlineStyleSheet.reset ( stylesheet );
            }

            // Parse inline style
            const std::string& css{ style->second };
            std::cerr << tagName() << std::endl;
            std::cerr << "css: " << css << std::endl << std::endl;
            code = css_stylesheet_append_data ( mInlineStyleSheet.get(), reinterpret_cast<const uint8_t*> ( css.data() ), css.size() );
            if ( code != CSS_OK  && code != CSS_NEEDDATA )
            {
                throw std::runtime_error ( css_error_to_string ( code ) );
            }
            code = css_stylesheet_data_done ( mInlineStyleSheet.get() );
            if ( code != CSS_OK )
            {
                throw std::runtime_error ( css_error_to_string ( code ) );
            }

            css_select_ctx* css_select_ctx{};
            css_error err{css_select_ctx_create ( &css_select_ctx ) };
            if ( err != CSS_OK )
            {
                throw std::runtime_error ( css_error_to_string ( code ) );
            }

            css_select_results* results {};
            err = css_select_style ( css_select_ctx, this, GetUnitLenCtx(), nullptr, mInlineStyleSheet.get(), GetSelectHandler(), nullptr, &results );
            if ( err != CSS_OK )
            {
                throw std::runtime_error ( css_error_to_string ( code ) );
            }
            mComputedStyles.reset ( results );
            css_select_ctx_destroy ( css_select_ctx );
        }
    }

    Element::~Element() = default;

    css_select_results* Element::GetParentComputedStyles() const
    {
        css_select_results* results {};
        Node* parent{ parentNode() };
        while ( parent != nullptr && results == nullptr )
        {
            if ( parent->nodeType() == ELEMENT_NODE && reinterpret_cast<Element * > ( parent )->mComputedStyles.get() != nullptr )
            {
                results = reinterpret_cast<Element*> ( parent )->mComputedStyles.get();
            }
            parent = parent->parentNode();
        }
        return results;
    }

    css_select_results* Element::GetComputedStyles() const
    {
        if ( mComputedStyles )
        {
            return mComputedStyles.get();
        }
        return GetParentComputedStyles();
    }

    void Element::OnAncestorChanged()
    {
        if ( !mComputedStyles )
        {
            return;
        }
        css_select_results* results {GetParentComputedStyles() };
        if ( results != nullptr )
        {
            css_computed_style* computed_style{};
            css_error err
            {
                css_computed_style_compose ( results->styles[CSS_PSEUDO_ELEMENT_NONE], mComputedStyles->styles[CSS_PSEUDO_ELEMENT_NONE], GetUnitLenCtx(), &computed_style )
            };
            if ( err != CSS_OK )
            {
                throw std::runtime_error ( css_error_to_string ( err ) );
            }
            css_computed_style_destroy ( mComputedStyles->styles[CSS_PSEUDO_ELEMENT_NONE] );
            mComputedStyles->styles[CSS_PSEUDO_ELEMENT_NONE] = computed_style;
        }
    }
#if 0
    AttributeType Element::GetAttribute ( const char* attrName, const AttributeType& aDefault ) const
    {
        auto i = mAttributeMap.find ( attrName );
        if ( i != mAttributeMap.end() )
        {
            return i->second;
        }
        return aDefault;
    }

    void Element::SetAttribute ( const char* attrName, const AttributeType& aValue )
    {
        mAttributeMap[attrName] = aValue;
    }

    AttributeType Element::GetInheritedAttribute ( const char* attrName, const AttributeType& aDefault ) const
    {
        AttributeType attr = GetAttribute ( attrName );
        Node* parent = parentNode();
        while ( std::holds_alternative<std::monostate> ( attr ) && parent != nullptr )
        {
            if ( parent->nodeType() == ELEMENT_NODE )
            {
                attr = reinterpret_cast<Element*> ( parent )->GetAttribute ( attrName );
            }
            parent = parent->parentNode();
        }
        return std::holds_alternative<std::monostate> ( attr ) ? aDefault : attr;
    }
#endif
    Node::NodeType Element::nodeType() const
    {
        return ELEMENT_NODE;
    }
    const std::string& Element::tagName() const
    {
        return mTagName;
    }
}
