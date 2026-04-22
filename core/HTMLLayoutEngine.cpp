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

#include "aeongui/HTMLLayoutEngine.hpp"
#include "aeongui/dom/HTMLElement.hpp"
#include "aeongui/dom/Node.hpp"
#include "aeongui/StyleSheet.hpp"

#include <yoga/Yoga.h>
#include <libcss/libcss.h>

namespace AeonGUI
{
    namespace
    {
        /// Convert a libcss css_fixed (Q16.16) to float pixels.
        inline float FixedToPx ( css_fixed aValue )
        {
            return static_cast<float> ( FIXTOFLT ( aValue ) );
        }

        /// Map a libcss flex-direction enum to a Yoga value.  Inherit/unknown
        /// fall back to column to mirror Yoga's own default.
        YGFlexDirection MapFlexDirection ( uint8_t aValue )
        {
            switch ( aValue )
            {
            case CSS_FLEX_DIRECTION_ROW:
                return YGFlexDirectionRow;
            case CSS_FLEX_DIRECTION_ROW_REVERSE:
                return YGFlexDirectionRowReverse;
            case CSS_FLEX_DIRECTION_COLUMN_REVERSE:
                return YGFlexDirectionColumnReverse;
            case CSS_FLEX_DIRECTION_COLUMN:
            case CSS_FLEX_DIRECTION_INHERIT:
            default:
                return YGFlexDirectionColumn;
            }
        }

        /// Apply a width/height accessor result.  CSS_*_AUTO is
        /// intentionally a no-op: Yoga's "no value set" default already
        /// behaves like CSS auto for flex children (stretch on the
        /// cross axis, fit content on the main axis), whereas calling
        /// the explicit *Auto setter would force a different mode.
        ///
        /// @param aSetter         Yoga absolute setter, e.g. YGNodeStyleSetWidth.
        /// @param aPercentSetter  Yoga percent setter, e.g. YGNodeStyleSetWidthPercent.
        template<typename SetFn, typename PctFn>
        void ApplyDimension (
            YGNodeRef aNode,
            uint8_t aResult, uint8_t aSetSentinel,
            css_fixed aLength, css_unit aUnit,
            SetFn aSetter, PctFn aPercentSetter )
        {
            if ( aResult != aSetSentinel )
            {
                return;
            }
            if ( aUnit == CSS_UNIT_PCT )
            {
                aPercentSetter ( aNode, FixedToPx ( aLength ) );
            }
            else if ( aUnit == CSS_UNIT_PX )
            {
                aSetter ( aNode, FixedToPx ( aLength ) );
            }
            // Other units (em, rem, pt, ...) intentionally ignored in
            // this slice — they need a length-resolution context that
            // we will plumb through in the next milestone.
        }

        /// Apply width to a Yoga node from a libcss computed style.
        void ApplyWidth ( YGNodeRef aNode, const css_computed_style* aStyle )
        {
            css_fixed length{};
            css_unit unit{};
            uint8_t result = css_computed_width ( aStyle, &length, &unit );
            ApplyDimension (
                aNode, result, CSS_WIDTH_SET, length, unit,
                YGNodeStyleSetWidth, YGNodeStyleSetWidthPercent );
        }

        /// Apply height to a Yoga node from a libcss computed style.
        void ApplyHeight ( YGNodeRef aNode, const css_computed_style* aStyle )
        {
            css_fixed length{};
            css_unit unit{};
            uint8_t result = css_computed_height ( aStyle, &length, &unit );
            ApplyDimension (
                aNode, result, CSS_WIDTH_SET, length, unit,
                YGNodeStyleSetHeight, YGNodeStyleSetHeightPercent );
        }

        /// Apply one margin edge.
        void ApplyMargin ( YGNodeRef aNode, YGEdge aEdge,
                           uint8_t ( *aGetter ) ( const css_computed_style*, css_fixed*, css_unit* ),
                           const css_computed_style* aStyle )
        {
            css_fixed length{};
            css_unit unit{};
            uint8_t result = aGetter ( aStyle, &length, &unit );
            if ( result == CSS_MARGIN_SET )
            {
                if ( unit == CSS_UNIT_PCT )
                {
                    YGNodeStyleSetMarginPercent ( aNode, aEdge, FixedToPx ( length ) );
                }
                else if ( unit == CSS_UNIT_PX )
                {
                    YGNodeStyleSetMargin ( aNode, aEdge, FixedToPx ( length ) );
                }
            }
            else if ( result == CSS_MARGIN_AUTO )
            {
                YGNodeStyleSetMarginAuto ( aNode, aEdge );
            }
        }

        /// Apply one padding edge.  Padding has no AUTO sentinel.
        void ApplyPadding ( YGNodeRef aNode, YGEdge aEdge,
                            uint8_t ( *aGetter ) ( const css_computed_style*, css_fixed*, css_unit* ),
                            const css_computed_style* aStyle )
        {
            css_fixed length{};
            css_unit unit{};
            uint8_t result = aGetter ( aStyle, &length, &unit );
            if ( result == CSS_PADDING_SET )
            {
                if ( unit == CSS_UNIT_PCT )
                {
                    YGNodeStyleSetPaddingPercent ( aNode, aEdge, FixedToPx ( length ) );
                }
                else if ( unit == CSS_UNIT_PX )
                {
                    YGNodeStyleSetPadding ( aNode, aEdge, FixedToPx ( length ) );
                }
            }
        }

        /// Resolve a CSS border-width to pixels for layout purposes.
        /// Styles NONE and HIDDEN force a zero-width edge per CSS 2.1
        /// regardless of the declared width.  Keyword widths use the
        /// conventional 1/3/5 px mapping; explicit lengths only honor
        /// CSS_UNIT_PX in this slice (em/rem need a length context).
        float BorderEdgePx ( uint8_t aStyleResult, uint8_t aWidthResult,
                             css_fixed aLength, css_unit aUnit )
        {
            if ( aStyleResult == CSS_BORDER_STYLE_NONE ||
                 aStyleResult == CSS_BORDER_STYLE_HIDDEN )
            {
                return 0.0f;
            }
            switch ( aWidthResult )
            {
            case CSS_BORDER_WIDTH_THIN:
                return 1.0f;
            case CSS_BORDER_WIDTH_MEDIUM:
                return 3.0f;
            case CSS_BORDER_WIDTH_THICK:
                return 5.0f;
            case CSS_BORDER_WIDTH_WIDTH:
                if ( aUnit == CSS_UNIT_PX )
                {
                    return FixedToPx ( aLength );
                }
                return 0.0f;
            default:
                return 0.0f;
            }
        }

        /// Apply one border edge to a Yoga node so the border reserves
        /// space inside the border box (Yoga shrinks the content area
        /// by the border on each side).
        void ApplyBorder ( YGNodeRef aNode, YGEdge aEdge,
                           uint8_t ( *aWidthGetter ) ( const css_computed_style*, css_fixed*, css_unit* ),
                           uint8_t ( *aStyleGetter ) ( const css_computed_style* ),
                           const css_computed_style* aStyle )
        {
            css_fixed length{};
            css_unit unit{};
            uint8_t width_result = aWidthGetter ( aStyle, &length, &unit );
            uint8_t style_result = aStyleGetter ( aStyle );
            float px = BorderEdgePx ( style_result, width_result, length, unit );
            if ( px > 0.0f )
            {
                YGNodeStyleSetBorder ( aNode, aEdge, px );
            }
        }

        /// Apply the subset of CSS properties this slice understands to
        /// a Yoga node.  Anything we don't read keeps Yoga's default.
        ///
        /// Note: libcss has no HTML UA stylesheet loaded, so `display`
        /// for an unstyled element comes back as the CSS-spec default
        /// `inline`.  Because HTML's actual semantic for most elements
        /// is `block`, we treat anything that is not explicitly `flex`,
        /// `inline-flex`, or `none` as a block-formatted box (Yoga
        /// flex column).  flex-direction is only honored for actual
        /// flex containers — otherwise we'd pick up the CSS default
        /// `row` and stack block children horizontally.
        void ApplyComputedStyle ( YGNodeRef aNode, const css_computed_style* aStyle, bool aIsRoot )
        {
            uint8_t display = css_computed_display ( aStyle, aIsRoot );
            if ( display == CSS_DISPLAY_NONE )
            {
                YGNodeStyleSetDisplay ( aNode, YGDisplayNone );
                return;
            }
            YGNodeStyleSetDisplay ( aNode, YGDisplayFlex );

            const bool is_flex_container =
                ( display == CSS_DISPLAY_FLEX || display == CSS_DISPLAY_INLINE_FLEX );
            YGNodeStyleSetFlexDirection ( aNode,
                                          is_flex_container
                                          ? MapFlexDirection ( css_computed_flex_direction ( aStyle ) )
                                          : YGFlexDirectionColumn );

            ApplyWidth  ( aNode, aStyle );
            ApplyHeight ( aNode, aStyle );

            ApplyMargin ( aNode, YGEdgeTop,    css_computed_margin_top,    aStyle );
            ApplyMargin ( aNode, YGEdgeRight,  css_computed_margin_right,  aStyle );
            ApplyMargin ( aNode, YGEdgeBottom, css_computed_margin_bottom, aStyle );
            ApplyMargin ( aNode, YGEdgeLeft,   css_computed_margin_left,   aStyle );

            ApplyPadding ( aNode, YGEdgeTop,    css_computed_padding_top,    aStyle );
            ApplyPadding ( aNode, YGEdgeRight,  css_computed_padding_right,  aStyle );
            ApplyPadding ( aNode, YGEdgeBottom, css_computed_padding_bottom, aStyle );
            ApplyPadding ( aNode, YGEdgeLeft,   css_computed_padding_left,   aStyle );

            ApplyBorder ( aNode, YGEdgeTop,    css_computed_border_top_width,
                          css_computed_border_top_style,    aStyle );
            ApplyBorder ( aNode, YGEdgeRight,  css_computed_border_right_width,
                          css_computed_border_right_style,  aStyle );
            ApplyBorder ( aNode, YGEdgeBottom, css_computed_border_bottom_width,
                          css_computed_border_bottom_style, aStyle );
            ApplyBorder ( aNode, YGEdgeLeft,   css_computed_border_left_width,
                          css_computed_border_left_style,   aStyle );
        }

        /// Recursively build a Yoga subtree mirroring HTMLElement
        /// children of @p aElement.  Non-HTML DOM children (text, inline
        /// SVG, ...) are skipped in this slice — they will become opaque
        /// leaf boxes in a later milestone via measure callbacks.
        YGNodeRef BuildYogaSubtree ( YGConfigRef aConfig, DOM::HTMLElement* aElement, bool aIsRoot )
        {
            YGNodeRef node = YGNodeNewWithConfig ( aConfig );

            css_select_results* results = aElement->GetComputedStyles();
            if ( results && results->styles[CSS_PSEUDO_ELEMENT_NONE] )
            {
                ApplyComputedStyle ( node, results->styles[CSS_PSEUDO_ELEMENT_NONE], aIsRoot );
            }

            uint32_t insertion_index = 0;
            for ( const auto& child : aElement->childNodes() )
            {
                DOM::HTMLElement* html_child = dynamic_cast<DOM::HTMLElement*> ( child.get() );
                if ( !html_child )
                {
                    continue;
                }
                YGNodeRef child_node = BuildYogaSubtree ( aConfig, html_child, false );
                YGNodeInsertChild ( node, child_node, insertion_index++ );
            }

            return node;
        }

        /// Recursively read computed positions out of the Yoga tree and
        /// write them onto each HTMLElement in document coordinates.
        void WriteLayoutBack ( DOM::HTMLElement* aElement, YGNodeRef aNode,
                               float aParentX, float aParentY )
        {
            const float x = aParentX + YGNodeLayoutGetLeft ( aNode );
            const float y = aParentY + YGNodeLayoutGetTop ( aNode );
            const float w = YGNodeLayoutGetWidth ( aNode );
            const float h = YGNodeLayoutGetHeight ( aNode );

            aElement->SetLayoutBox ( DOM::HTMLElement::LayoutBox{ x, y, w, h } );

            uint32_t yoga_index = 0;
            for ( const auto& child : aElement->childNodes() )
            {
                DOM::HTMLElement* html_child = dynamic_cast<DOM::HTMLElement*> ( child.get() );
                if ( !html_child )
                {
                    continue;
                }
                YGNodeRef child_node = YGNodeGetChild ( aNode, yoga_index++ );
                WriteLayoutBack ( html_child, child_node, x, y );
            }
        }
    }

    struct HTMLLayoutEngine::Impl
    {
        YGConfigRef config{};
    };

    HTMLLayoutEngine::HTMLLayoutEngine()
        : mImpl{ new Impl{} }
    {
        mImpl->config = YGConfigNew();
    }

    HTMLLayoutEngine::~HTMLLayoutEngine()
    {
        if ( mImpl )
        {
            if ( mImpl->config )
            {
                YGConfigFree ( mImpl->config );
            }
            delete mImpl;
        }
    }

    void HTMLLayoutEngine::Layout ( DOM::HTMLElement* aRoot, float aWidth, float aHeight )
    {
        if ( !aRoot )
        {
            return;
        }

        YGNodeRef root_node = BuildYogaSubtree ( mImpl->config, aRoot, true );

        // Per the HTML support plan's hard constraint, the root
        // containing block IS the window: there is no viewport vs
        // document distinction.  If the root element does not specify
        // an explicit width/height in CSS, fall back to the available
        // size so that descendants with percentage sizes and stretch
        // alignment have a defined containing block.  An explicit
        // width/height on the root is still respected — useful for
        // tests laying out a single subtree.
        css_select_results* root_results = aRoot->GetComputedStyles();
        const css_computed_style* root_style =
            ( root_results && root_results->styles[CSS_PSEUDO_ELEMENT_NONE] )
            ? root_results->styles[CSS_PSEUDO_ELEMENT_NONE] : nullptr;
        if ( !root_style )
        {
            YGNodeStyleSetWidth  ( root_node, aWidth );
            YGNodeStyleSetHeight ( root_node, aHeight );
        }
        else
        {
            css_fixed length{};
            css_unit unit{};
            if ( css_computed_width  ( root_style, &length, &unit ) != CSS_WIDTH_SET )
            {
                YGNodeStyleSetWidth ( root_node, aWidth );
            }
            if ( css_computed_height ( root_style, &length, &unit ) != CSS_WIDTH_SET )
            {
                YGNodeStyleSetHeight ( root_node, aHeight );
            }
        }

        YGNodeCalculateLayout ( root_node, aWidth, aHeight, YGDirectionLTR );
        WriteLayoutBack ( aRoot, root_node, 0.0f, 0.0f );
        YGNodeFreeRecursive ( root_node );
    }
}
