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
#include <iostream>
#include <stdexcept>
#include <libcss/libcss.h>
#include "aeongui/CSSSelector.hpp"
#include "aeongui/LogLevel.hpp"
#include "aeongui/dom/Element.hpp"

namespace AeonGUI
{
    namespace
    {
        class SelectorParser
        {
        public:
            explicit SelectorParser ( const std::string& aInput )
                : mInput{aInput}, mPos{0} {}

            std::vector<ComplexSelector> Parse()
            {
                std::vector<ComplexSelector> selectors;
                SkipWhitespace();
                if ( mPos >= mInput.size() )
                {
                    return selectors;
                }
                selectors.push_back ( ParseComplexSelector() );
                while ( mPos < mInput.size() )
                {
                    SkipWhitespace();
                    if ( mPos < mInput.size() && mInput[mPos] == ',' )
                    {
                        ++mPos;
                        SkipWhitespace();
                        selectors.push_back ( ParseComplexSelector() );
                    }
                    else
                    {
                        break;
                    }
                }
                return selectors;
            }

        private:
            const std::string& mInput;
            size_t mPos;

            void SkipWhitespace()
            {
                while ( mPos < mInput.size() && std::isspace ( static_cast<unsigned char> ( mInput[mPos] ) ) )
                {
                    ++mPos;
                }
            }

            bool IsIdentChar ( char c ) const
            {
                return std::isalnum ( static_cast<unsigned char> ( c ) ) || c == '-' || c == '_';
            }

            std::string ParseIdentifier()
            {
                size_t start = mPos;
                while ( mPos < mInput.size() && IsIdentChar ( mInput[mPos] ) )
                {
                    ++mPos;
                }
                if ( mPos == start )
                {
                    std::cerr << LogLevel::Error << "Expected identifier in CSS selector" << std::endl;
                    throw std::runtime_error ( "Expected identifier in CSS selector" );
                }
                return mInput.substr ( start, mPos - start );
            }

            SimpleSelector ParseSimpleSelector()
            {
                SimpleSelector sel{};
                if ( mPos >= mInput.size() )
                {
                    std::cerr << LogLevel::Error << "Unexpected end of selector" << std::endl;
                    throw std::runtime_error ( "Unexpected end of selector" );
                }

                char c = mInput[mPos];
                if ( c == '*' )
                {
                    sel.type = SimpleSelector::Universal;
                    ++mPos;
                }
                else if ( c == '#' )
                {
                    ++mPos;
                    sel.type = SimpleSelector::Id;
                    sel.name = ParseIdentifier();
                }
                else if ( c == '.' )
                {
                    ++mPos;
                    sel.type = SimpleSelector::Class;
                    sel.name = ParseIdentifier();
                }
                else if ( c == '[' )
                {
                    ++mPos;
                    sel.type = SimpleSelector::Attribute;
                    SkipWhitespace();
                    sel.name = ParseIdentifier();
                    SkipWhitespace();
                    if ( mPos < mInput.size() && mInput[mPos] == ']' )
                    {
                        sel.attrOp = SimpleSelector::Exists;
                        ++mPos;
                    }
                    else if ( mPos < mInput.size() )
                    {
                        // Parse operator
                        if ( mInput[mPos] == '=' )
                        {
                            sel.attrOp = SimpleSelector::Equals;
                            ++mPos;
                        }
                        else if ( mPos + 1 < mInput.size() && mInput[mPos + 1] == '=' )
                        {
                            switch ( mInput[mPos] )
                            {
                            case '~':
                                sel.attrOp = SimpleSelector::Includes;
                                break;
                            case '|':
                                sel.attrOp = SimpleSelector::DashMatch;
                                break;
                            case '^':
                                sel.attrOp = SimpleSelector::Prefix;
                                break;
                            case '$':
                                sel.attrOp = SimpleSelector::Suffix;
                                break;
                            case '*':
                                sel.attrOp = SimpleSelector::Substring;
                                break;
                            default:
                                std::cerr << LogLevel::Error << "Unknown attribute operator in CSS selector" << std::endl;
                                throw std::runtime_error ( "Unknown attribute operator in CSS selector" );
                            }
                            mPos += 2;
                        }
                        else
                        {
                            std::cerr << LogLevel::Error << "Invalid attribute selector syntax" << std::endl;
                            throw std::runtime_error ( "Invalid attribute selector syntax" );
                        }
                        SkipWhitespace();
                        sel.value = ParseAttributeValue();
                        SkipWhitespace();
                        if ( mPos >= mInput.size() || mInput[mPos] != ']' )
                        {
                            std::cerr << LogLevel::Error << "Expected ']' in attribute selector" << std::endl;
                            throw std::runtime_error ( "Expected ']' in attribute selector" );
                        }
                        ++mPos;
                    }
                }
                else if ( IsIdentChar ( c ) )
                {
                    sel.type = SimpleSelector::TypeSel;
                    sel.name = ParseIdentifier();
                }
                else
                {
                    std::cerr << LogLevel::Error << "Unexpected character in CSS selector" << std::endl;
                    throw std::runtime_error ( "Unexpected character in CSS selector" );
                }
                return sel;
            }

            std::string ParseAttributeValue()
            {
                if ( mPos >= mInput.size() )
                {
                    std::cerr << LogLevel::Error << "Unexpected end of attribute value" << std::endl;
                    throw std::runtime_error ( "Unexpected end of attribute value" );
                }
                if ( mInput[mPos] == '"' || mInput[mPos] == '\'' )
                {
                    char quote = mInput[mPos];
                    ++mPos;
                    size_t start = mPos;
                    while ( mPos < mInput.size() && mInput[mPos] != quote )
                    {
                        ++mPos;
                    }
                    if ( mPos >= mInput.size() )
                    {
                        std::cerr << LogLevel::Error << "Unterminated string in attribute selector" << std::endl;
                        throw std::runtime_error ( "Unterminated string in attribute selector" );
                    }
                    std::string val = mInput.substr ( start, mPos - start );
                    ++mPos; // skip closing quote
                    return val;
                }
                // Unquoted value (just an identifier)
                return ParseIdentifier();
            }

            CompoundSelector ParseCompoundSelector()
            {
                CompoundSelector compound{};
                // A compound selector starts with a type/universal selector or a simple selector starting with #, ., [
                while ( mPos < mInput.size() )
                {
                    char c = mInput[mPos];
                    if ( c == '#' || c == '.' || c == '[' )
                    {
                        compound.selectors.push_back ( ParseSimpleSelector() );
                    }
                    else if ( c == '*' || IsIdentChar ( c ) )
                    {
                        if ( compound.selectors.empty() ||
                             ( compound.selectors.back().type != SimpleSelector::TypeSel &&
                               compound.selectors.back().type != SimpleSelector::Universal ) )
                        {
                            compound.selectors.push_back ( ParseSimpleSelector() );
                        }
                        else
                        {
                            break;
                        }
                    }
                    else
                    {
                        break;
                    }
                }
                return compound;
            }

            ComplexSelector ParseComplexSelector()
            {
                ComplexSelector complex{};
                ComplexSelector::Part part{};
                part.compound = ParseCompoundSelector();
                part.combinator = Combinator::None;
                complex.parts.push_back ( std::move ( part ) );

                while ( mPos < mInput.size() )
                {
                    size_t beforeWs = mPos;
                    SkipWhitespace();
                    if ( mPos >= mInput.size() || mInput[mPos] == ',' )
                    {
                        break;
                    }
                    Combinator comb = Combinator::Descendant; // default: whitespace = descendant
                    char c = mInput[mPos];
                    if ( c == '>' )
                    {
                        comb = Combinator::Child;
                        ++mPos;
                        SkipWhitespace();
                    }
                    else if ( c == '+' )
                    {
                        comb = Combinator::AdjacentSibling;
                        ++mPos;
                        SkipWhitespace();
                    }
                    else if ( c == '~' )
                    {
                        comb = Combinator::GeneralSibling;
                        ++mPos;
                        SkipWhitespace();
                    }
                    else if ( mPos == beforeWs )
                    {
                        // No whitespace and no combinator → end of this complex selector
                        break;
                    }
                    // else: whitespace was consumed, descendant combinator

                    ComplexSelector::Part nextPart{};
                    nextPart.compound = ParseCompoundSelector();
                    nextPart.combinator = comb;
                    complex.parts.push_back ( std::move ( nextPart ) );
                }
                return complex;
            }
        };

        const DOM::Element* PreviousSiblingElement ( const DOM::Element& aElement )
        {
            const DOM::Node* parent = aElement.parentNode();
            if ( !parent )
            {
                return nullptr;
            }
            const auto& children = parent->childNodes();
            const DOM::Element* prev = nullptr;
            for ( const auto& child : children )
            {
                if ( child.get() == &aElement )
                {
                    return prev;
                }
                if ( child->nodeType() == DOM::Node::ELEMENT_NODE )
                {
                    prev = static_cast<const DOM::Element*> ( child.get() );
                }
            }
            return nullptr;
        }
    }

    std::vector<ComplexSelector> ParseSelector ( const std::string& aSelector )
    {
        SelectorParser parser{aSelector};
        return parser.Parse();
    }

    bool MatchesCompound ( const DOM::Element& aElement, const CompoundSelector& aCompound )
    {
        for ( const auto& simple : aCompound.selectors )
        {
            switch ( simple.type )
            {
            case SimpleSelector::Universal:
                break;
            case SimpleSelector::TypeSel:
                if ( aElement.tagName() != simple.name )
                {
                    return false;
                }
                break;
            case SimpleSelector::Id:
                if ( aElement.id() != simple.name )
                {
                    return false;
                }
                break;
            case SimpleSelector::Class:
            {
                bool found = false;
                for ( const auto * cls : aElement.classes() )
                {
                    const char* data = lwc_string_data ( cls );
                    size_t len = lwc_string_length ( cls );
                    if ( std::string ( data, len ) == simple.name )
                    {
                        found = true;
                        break;
                    }
                }
                if ( !found )
                {
                    return false;
                }
                break;
            }
            case SimpleSelector::Attribute:
            {
                const DOM::DOMString* val = aElement.getAttribute ( simple.name );
                switch ( simple.attrOp )
                {
                case SimpleSelector::Exists:
                    if ( !val )
                    {
                        return false;
                    }
                    break;
                case SimpleSelector::Equals:
                    if ( !val || *val != simple.value )
                    {
                        return false;
                    }
                    break;
                case SimpleSelector::Includes:
                {
                    if ( !val )
                    {
                        return false;
                    }
                    bool found = false;
                    size_t start = 0;
                    while ( start < val->size() )
                    {
                        size_t end = val->find ( ' ', start );
                        if ( end == std::string::npos )
                        {
                            end = val->size();
                        }
                        if ( val->substr ( start, end - start ) == simple.value )
                        {
                            found = true;
                            break;
                        }
                        start = end + 1;
                    }
                    if ( !found )
                    {
                        return false;
                    }
                    break;
                }
                case SimpleSelector::DashMatch:
                    if ( !val || ( *val != simple.value && val->substr ( 0, simple.value.size() + 1 ) != simple.value + "-" ) )
                    {
                        return false;
                    }
                    break;
                case SimpleSelector::Prefix:
                    if ( !val || val->substr ( 0, simple.value.size() ) != simple.value )
                    {
                        return false;
                    }
                    break;
                case SimpleSelector::Suffix:
                    if ( !val || val->size() < simple.value.size() || val->substr ( val->size() - simple.value.size() ) != simple.value )
                    {
                        return false;
                    }
                    break;
                case SimpleSelector::Substring:
                    if ( !val || val->find ( simple.value ) == std::string::npos )
                    {
                        return false;
                    }
                    break;
                }
                break;
            }
            }
        }
        return true;
    }

    bool MatchesSelector ( const DOM::Element& aElement, const ComplexSelector& aSelector )
    {
        if ( aSelector.parts.empty() )
        {
            return false;
        }

        // Match right-to-left: the last part must match the element.
        size_t idx = aSelector.parts.size() - 1;
        if ( !MatchesCompound ( aElement, aSelector.parts[idx].compound ) )
        {
            return false;
        }

        if ( idx == 0 )
        {
            return true;
        }

        const DOM::Element* current = &aElement;
        --idx;

        while ( true )
        {
            const auto& part = aSelector.parts[idx];
            // part.combinator is the combinator between parts[idx] and parts[idx+1]
            // It describes how parts[idx+1] relates to parts[idx].
            // But we stored it as the combinator ON the right part. Let me re-check...
            // Actually, parts[idx+1].combinator is the combinator between parts[idx] and parts[idx+1].
            Combinator comb = aSelector.parts[idx + 1].combinator;
            switch ( comb )
            {
            case Combinator::Descendant:
            {
                bool matched = false;
                DOM::Node* ancestor = current->parentNode();
                while ( ancestor )
                {
                    if ( ancestor->nodeType() == DOM::Node::ELEMENT_NODE )
                    {
                        const auto* elem = static_cast<const DOM::Element*> ( ancestor );
                        if ( MatchesCompound ( *elem, part.compound ) )
                        {
                            current = elem;
                            matched = true;
                            break;
                        }
                    }
                    ancestor = ancestor->parentNode();
                }
                if ( !matched )
                {
                    return false;
                }
                break;
            }
            case Combinator::Child:
            {
                DOM::Node* parent = current->parentNode();
                if ( !parent || parent->nodeType() != DOM::Node::ELEMENT_NODE )
                {
                    return false;
                }
                const auto* elem = static_cast<const DOM::Element*> ( parent );
                if ( !MatchesCompound ( *elem, part.compound ) )
                {
                    return false;
                }
                current = elem;
                break;
            }
            case Combinator::AdjacentSibling:
            {
                const DOM::Element* prev = PreviousSiblingElement ( *current );
                if ( !prev || !MatchesCompound ( *prev, part.compound ) )
                {
                    return false;
                }
                current = prev;
                break;
            }
            case Combinator::GeneralSibling:
            {
                DOM::Node* parent = current->parentNode();
                if ( !parent )
                {
                    return false;
                }
                bool matched = false;
                const auto& children = parent->childNodes();
                for ( const auto& child : children )
                {
                    if ( child.get() == current )
                    {
                        break;
                    }
                    if ( child->nodeType() == DOM::Node::ELEMENT_NODE )
                    {
                        const auto* elem = static_cast<const DOM::Element*> ( child.get() );
                        if ( MatchesCompound ( *elem, part.compound ) )
                        {
                            current = elem;
                            matched = true;
                            // Don't break — keep looking for the closest match
                        }
                    }
                }
                if ( !matched )
                {
                    return false;
                }
                break;
            }
            case Combinator::None:
                // Shouldn't happen for non-rightmost parts, but treat as no match.
                return false;
            }

            if ( idx == 0 )
            {
                return true;
            }
            --idx;
        }
    }

    bool MatchesAny ( const DOM::Element& aElement, const std::vector<ComplexSelector>& aSelectors )
    {
        for ( const auto& sel : aSelectors )
        {
            if ( MatchesSelector ( aElement, sel ) )
            {
                return true;
            }
        }
        return false;
    }
}
