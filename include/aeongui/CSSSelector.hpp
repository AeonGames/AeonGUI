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
#ifndef AEONGUI_CSS_SELECTOR_H
#define AEONGUI_CSS_SELECTOR_H

#include <string>
#include <vector>
#include "aeongui/Platform.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        class Element;
    }

    /// A single simple selector component (type, id, class, or attribute).
    struct SimpleSelector
    {
        /// Kind of simple selector.
        enum Type
        {
            Universal,   ///< *
            TypeSel,     ///< e.g. rect, text
            Id,          ///< \#myId
            Class,       ///< .myClass
            Attribute    ///< [attr] or [attr=value]
        };
        Type type{Universal};    ///< The selector kind.
        std::string name{};      ///< Selector name (tag name, id, class, or attribute name).
        std::string value{}; ///< For attribute selectors with a value.
        /// Attribute matching operator.
        enum AttrOp
        {
            Exists,      ///< [attr]
            Equals,      ///< [attr=value]
            Includes,    ///< [attr~=value]
            DashMatch,   ///< [attr|=value]
            Prefix,      ///< [attr^=value]
            Suffix,      ///< [attr$=value]
            Substring    ///< [attr*=value]
        };
        AttrOp attrOp{Exists}; ///< The attribute match operation.
    };

    /// A compound selector: a sequence of simple selectors that must all match the same element.
    struct CompoundSelector
    {
        std::vector<SimpleSelector> selectors{}; ///< Simple selectors in this compound.
    };

    /// Combinator between compound selectors.
    enum class Combinator
    {
        None,              ///< No combinator (rightmost compound).
        Descendant,        ///< space — any ancestor
        Child,             ///< > — direct parent
        AdjacentSibling,   ///< + — immediately preceding sibling
        GeneralSibling     ///< ~ — any preceding sibling
    };

    /// A complex selector: a chain of compound selectors separated by combinators.
    /// Stored left-to-right as written in CSS.
    struct ComplexSelector
    {
        /// A compound selector together with the combinator that precedes it.
        struct Part
        {
            CompoundSelector compound{};            ///< The compound selector for this part.
            Combinator combinator{Combinator::None}; ///< Combinator linking to the next part.
        };
        std::vector<Part> parts{}; ///< Parts in left-to-right order as written in CSS.
    };

    /// Parse a CSS selector string into a list of complex selectors (comma-separated).
    DLL std::vector<ComplexSelector> ParseSelector ( const std::string& aSelector );

    /// Check whether an Element matches a compound selector.
    DLL bool MatchesCompound ( const DOM::Element& aElement, const CompoundSelector& aCompound );

    /// Check whether an Element matches a complex selector (full chain with combinators).
    DLL bool MatchesSelector ( const DOM::Element& aElement, const ComplexSelector& aSelector );

    /// Check whether an Element matches any of the parsed selectors (comma-separated list).
    DLL bool MatchesAny ( const DOM::Element& aElement, const std::vector<ComplexSelector>& aSelectors );
}

#endif
