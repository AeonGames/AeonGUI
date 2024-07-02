/*
Copyright (C) 2024 Rodrigo Jose Hernandez Cordoba

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
#include "CSSSelectHandler.h"
#include "Element.h"
namespace AeonGUI
{
    static css_error node_name ( void *pw, void *node,
                                 css_qname *qname );
    static css_error node_classes ( void *pw, void *node,
                                    lwc_string ***classes, uint32_t *n_classes );
    static css_error node_id ( void *pw, void *node,
                               lwc_string **id );
    static css_error named_ancestor_node ( void *pw, void *node,
                                           const css_qname *qname,
                                           void **ancestor );
    static css_error named_parent_node ( void *pw, void *node,
                                         const css_qname *qname,
                                         void **parent );
    static css_error named_sibling_node ( void *pw, void *node,
                                          const css_qname *qname,
                                          void **sibling );
    static css_error named_generic_sibling_node ( void *pw, void *node,
            const css_qname *qname,
            void **sibling );
    static css_error parent_node ( void *pw, void *node, void **parent );
    static css_error sibling_node ( void *pw, void *node, void **sibling );
    static css_error node_has_name ( void *pw, void *node,
                                     const css_qname *qname,
                                     bool *match );
    static css_error node_has_class ( void *pw, void *node,
                                      lwc_string *name,
                                      bool *match );
    static css_error node_has_id ( void *pw, void *node,
                                   lwc_string *name,
                                   bool *match );
    static css_error node_has_attribute ( void *pw, void *node,
                                          const css_qname *qname,
                                          bool *match );
    static css_error node_has_attribute_equal ( void *pw, void *node,
            const css_qname *qname,
            lwc_string *value,
            bool *match );
    static css_error node_has_attribute_dashmatch ( void *pw, void *node,
            const css_qname *qname,
            lwc_string *value,
            bool *match );
    static css_error node_has_attribute_includes ( void *pw, void *node,
            const css_qname *qname,
            lwc_string *value,
            bool *match );
    static css_error node_has_attribute_prefix ( void *pw, void *node,
            const css_qname *qname,
            lwc_string *value,
            bool *match );
    static css_error node_has_attribute_suffix ( void *pw, void *node,
            const css_qname *qname,
            lwc_string *value,
            bool *match );
    static css_error node_has_attribute_substring ( void *pw, void *node,
            const css_qname *qname,
            lwc_string *value,
            bool *match );
    static css_error node_is_root ( void *pw, void *node, bool *match );
    static css_error node_count_siblings ( void *pw, void *node,
                                           bool same_name, bool after, int32_t *count );
    static css_error node_is_empty ( void *pw, void *node, bool *match );
    static css_error node_is_link ( void *pw, void *node, bool *match );
    static css_error node_is_visited ( void *pw, void *node, bool *match );
    static css_error node_is_hover ( void *pw, void *node, bool *match );
    static css_error node_is_active ( void *pw, void *node, bool *match );
    static css_error node_is_focus ( void *pw, void *node, bool *match );
    static css_error node_is_enabled ( void *pw, void *node, bool *match );
    static css_error node_is_disabled ( void *pw, void *node, bool *match );
    static css_error node_is_checked ( void *pw, void *node, bool *match );
    static css_error node_is_target ( void *pw, void *node, bool *match );
    static css_error node_is_lang ( void *pw, void *node,
                                    lwc_string *lang, bool *match );
    static css_error node_presentational_hint ( void *pw, void *node,
            uint32_t *nhints, css_hint **hints );
    static css_error ua_default_for_property ( void *pw, uint32_t property,
            css_hint *hint );
    static css_error set_libcss_node_data ( void *pw, void *n,
                                            void *libcss_node_data );
    static css_error get_libcss_node_data ( void *pw, void *n,
                                            void **libcss_node_data );

    static css_unit_ctx unit_len_ctx =
    {
        .viewport_width    = 800 * ( 1 << CSS_RADIX_POINT ),
        .viewport_height   = 600 * ( 1 << CSS_RADIX_POINT ),
        .font_size_default =  16 * ( 1 << CSS_RADIX_POINT ),
        .font_size_minimum =   6 * ( 1 << CSS_RADIX_POINT ),
        .device_dpi        =  96 * ( 1 << CSS_RADIX_POINT ),
        .root_style        = nullptr, /* We don't have a root node yet. */
        .pw                = nullptr, /* We're not implementing measure callback */
        .measure           = nullptr, /* We're not implementing measure callback */
    };

    static css_select_handler select_handler =
    {
        CSS_SELECT_HANDLER_VERSION_1,

        node_name,
        node_classes,
        node_id,
        named_ancestor_node,
        named_parent_node,
        named_sibling_node,
        named_generic_sibling_node,
        parent_node,
        sibling_node,
        node_has_name,
        node_has_class,
        node_has_id,
        node_has_attribute,
        node_has_attribute_equal,
        node_has_attribute_dashmatch,
        node_has_attribute_includes,
        node_has_attribute_prefix,
        node_has_attribute_suffix,
        node_has_attribute_substring,
        node_is_root,
        node_count_siblings,
        node_is_empty,
        node_is_link,
        node_is_visited,
        node_is_hover,
        node_is_active,
        node_is_focus,
        node_is_enabled,
        node_is_disabled,
        node_is_checked,
        node_is_target,
        node_is_lang,
        node_presentational_hint,
        ua_default_for_property,
        set_libcss_node_data,
        get_libcss_node_data,
    };

    css_select_handler* GetSelectHandler()
    {
        return &select_handler;
    }

    css_unit_ctx* GetUnitLenCtx()
    {
        return &unit_len_ctx;
    }

    css_error node_name ( void *pw, void *node, css_qname *qname )
    {
        Element *element {reinterpret_cast<Element*> ( node ) };
        ( void ) ( pw );
        qname->name = lwc_string_ref ( element->tagName() );
        return CSS_OK;
    }

    css_error node_classes ( void *pw, void *node,
                             lwc_string ***classes, uint32_t *n_classes )
    {
        Element *element {reinterpret_cast<Element*> ( node ) };
        ( void ) ( pw );
        *classes = element->classes().data();
        *n_classes = static_cast<uint32_t> ( element->classes().size() );
        return CSS_OK;
    }

    css_error node_id ( void *pw, void *node, lwc_string **id )
    {
        if ( !id )
        {
            return  CSS_BADPARM;
        }
        Element *element {reinterpret_cast<Element*> ( node ) };
        ( void ) ( pw );
        *id = element->id() ? lwc_string_ref ( element->id() ) : nullptr;
        return CSS_OK;
    }

    css_error named_ancestor_node ( void *pw, void *n,
                                    const css_qname *qname,
                                    void **ancestor )
    {
        ( void ) ( pw );
        ( void ) ( n );
        ( void ) ( qname );
        *ancestor = nullptr;
        return CSS_OK;
    }

    css_error named_parent_node ( void *pw, void *n,
                                  const css_qname *qname,
                                  void **parent )
    {
        ( void ) ( pw );
        ( void ) ( n );
        ( void ) ( qname );
        *parent = nullptr;
        return CSS_OK;
    }

    css_error named_generic_sibling_node ( void *pw, void *n,
                                           const css_qname *qname,
                                           void **sibling )
    {
        ( void ) ( pw );
        ( void ) ( n );
        ( void ) ( qname );
        *sibling = nullptr;
        return CSS_OK;
    }

    css_error named_sibling_node ( void *pw, void *n,
                                   const css_qname *qname,
                                   void **sibling )
    {
        ( void ) ( pw );
        ( void ) ( n );
        ( void ) ( qname );
        *sibling = nullptr;
        return CSS_OK;
    }

    css_error parent_node ( void *pw, void *node, void **parent )
    {
        if ( !parent )
        {
            return  CSS_BADPARM;
        }
        Element *element {reinterpret_cast<Element*> ( node ) };
        ( void ) ( pw );
        *parent = element->parentNode();
        return CSS_OK;
    }

    css_error sibling_node ( void *pw, void *n, void **sibling )
    {
        ( void ) ( pw );
        ( void ) ( n );
        *sibling = nullptr;
        return CSS_OK;
    }

    css_error node_has_name ( void *pw, void *node,
                              const css_qname *qname,
                              bool *match )
    {
        ( void ) ( pw );
        Element *element {reinterpret_cast<Element*> ( node ) };
        lwc_string_caseless_isequal ( element->tagName(), qname->name, match );
        return CSS_OK;
    }

    css_error node_has_class ( void *pw, void *n,
                               lwc_string *name,
                               bool *match )
    {
        ( void ) ( pw );
        ( void ) ( n );
        ( void ) ( name );
        *match = false;
        return CSS_OK;
    }

    css_error node_has_id ( void *pw, void *node,
                            lwc_string *id,
                            bool *match )
    {
        ( void ) ( pw );
        Element *element {reinterpret_cast<Element*> ( node ) };
        lwc_string_caseless_isequal ( element->id(), id, match );
        return CSS_OK;
    }

    css_error node_has_attribute ( void *pw, void *n,
                                   const css_qname *qname,
                                   bool *match )
    {
        ( void ) ( pw );
        ( void ) ( n );
        ( void ) ( qname );
        *match = false;
        return CSS_OK;
    }

    css_error node_has_attribute_equal ( void *pw, void *n,
                                         const css_qname *qname,
                                         lwc_string *value,
                                         bool *match )
    {
        ( void ) ( pw );
        ( void ) ( n );
        ( void ) ( qname );
        ( void ) ( value );
        *match = false;
        return CSS_OK;
    }

    css_error node_has_attribute_dashmatch ( void *pw, void *n,
            const css_qname *qname,
            lwc_string *value,
            bool *match )
    {
        ( void ) ( pw );
        ( void ) ( n );
        ( void ) ( qname );
        ( void ) ( value );
        *match = false;
        return CSS_OK;
    }

    css_error node_has_attribute_includes ( void *pw, void *n,
                                            const css_qname *qname,
                                            lwc_string *value,
                                            bool *match )
    {
        ( void ) ( pw );
        ( void ) ( n );
        ( void ) ( qname );
        ( void ) ( value );
        *match = false;
        return CSS_OK;
    }

    css_error node_has_attribute_prefix ( void *pw, void *n,
                                          const css_qname *qname,
                                          lwc_string *value,
                                          bool *match )
    {
        ( void ) ( pw );
        ( void ) ( n );
        ( void ) ( qname );
        ( void ) ( value );
        *match = false;
        return CSS_OK;
    }

    css_error node_has_attribute_suffix ( void *pw, void *n,
                                          const css_qname *qname,
                                          lwc_string *value,
                                          bool *match )
    {
        ( void ) ( pw );
        ( void ) ( n );
        ( void ) ( qname );
        ( void ) ( value );
        *match = false;
        return CSS_OK;
    }

    css_error node_has_attribute_substring ( void *pw, void *n,
            const css_qname *qname,
            lwc_string *value,
            bool *match )
    {
        ( void ) ( pw );
        ( void ) ( n );
        ( void ) ( qname );
        ( void ) ( value );
        *match = false;
        return CSS_OK;
    }

    css_error node_is_first_child ( void *pw, void *n, bool *match )
    {
        ( void ) ( pw );
        ( void ) ( n );
        *match = false;
        return CSS_OK;
    }

    css_error node_is_root ( void *pw, void *n, bool *match )
    {
        ( void ) ( pw );
        ( void ) ( n );
        *match = false;
        return CSS_OK;
    }

    css_error node_count_siblings ( void *pw, void *n,
                                    bool same_name, bool after, int32_t *count )
    {
        ( void ) ( pw );
        ( void ) ( n );
        ( void ) ( same_name );
        ( void ) ( after );
        *count = 1;
        return CSS_OK;
    }

    css_error node_is_empty ( void *pw, void *n, bool *match )
    {
        ( void ) ( pw );
        ( void ) ( n );
        *match = false;
        return CSS_OK;
    }

    css_error node_is_link ( void *pw, void *n, bool *match )
    {
        ( void ) ( pw );
        ( void ) ( n );
        *match = false;
        return CSS_OK;
    }

    css_error node_is_visited ( void *pw, void *n, bool *match )
    {
        ( void ) ( pw );
        ( void ) ( n );
        *match = false;
        return CSS_OK;
    }

    css_error node_is_hover ( void *pw, void *n, bool *match )
    {
        ( void ) ( pw );
        ( void ) ( n );
        *match = false;
        return CSS_OK;
    }

    css_error node_is_active ( void *pw, void *n, bool *match )
    {
        ( void ) ( pw );
        ( void ) ( n );
        *match = false;
        return CSS_OK;
    }

    css_error node_is_focus ( void *pw, void *n, bool *match )
    {
        ( void ) ( pw );
        ( void ) ( n );
        *match = false;
        return CSS_OK;
    }

    css_error node_is_enabled ( void *pw, void *n, bool *match )
    {
        ( void ) ( pw );
        ( void ) ( n );
        *match = false;
        return CSS_OK;
    }

    css_error node_is_disabled ( void *pw, void *n, bool *match )
    {
        ( void ) ( pw );
        ( void ) ( n );
        *match = false;
        return CSS_OK;
    }

    css_error node_is_checked ( void *pw, void *n, bool *match )
    {
        ( void ) ( pw );
        ( void ) ( n );
        *match = false;
        return CSS_OK;
    }

    css_error node_is_target ( void *pw, void *n, bool *match )
    {
        ( void ) ( pw );
        ( void ) ( n );
        *match = false;
        return CSS_OK;
    }


    css_error node_is_lang ( void *pw, void *n,
                             lwc_string *lang,
                             bool *match )
    {
        ( void ) ( pw );
        ( void ) ( n );
        ( void ) ( lang );
        *match = false;
        return CSS_OK;
    }

    css_error node_presentational_hint ( void *pw, void *node,
                                         uint32_t *nhints, css_hint **hints )
    {
        ( void ) ( pw );
        ( void ) ( node );
        *nhints = 0;
        *hints = nullptr;
        return CSS_OK;
    }

    css_error ua_default_for_property ( void *pw, uint32_t property, css_hint *hint )
    {
        ( void ) ( pw );

        if ( property == CSS_PROP_COLOR )
        {
            hint->data.color = 0x00000000;
            hint->status = CSS_COLOR_COLOR;
        }
        else if ( property == CSS_PROP_FONT_FAMILY )
        {
            hint->data.strings = nullptr;
            hint->status = CSS_FONT_FAMILY_SANS_SERIF;
        }
        else if ( property == CSS_PROP_QUOTES )
        {
            /* Not exactly useful :) */
            hint->data.strings = nullptr;
            hint->status = CSS_QUOTES_NONE;
        }
        else if ( property == CSS_PROP_VOICE_FAMILY )
        {
            /** \todo Fix this when we have voice-family done */
            hint->data.strings = nullptr;
            hint->status = 0;
        }
        else
        {
            return CSS_INVALID;
        }

        return CSS_OK;
    }

    static css_error set_libcss_node_data ( void *pw, void *n,
                                            void *libcss_node_data )
    {
        ( void ) ( pw );
        ( void ) ( n );

        /* Since we're not storing it, ensure node data gets deleted */
        css_libcss_node_data_handler ( &select_handler, CSS_NODE_DELETED,
                                       pw, n, nullptr, libcss_node_data );

        return CSS_OK;
    }

    static css_error get_libcss_node_data ( void *pw, void *n,
                                            void **libcss_node_data )
    {
        ( void ) ( pw );
        ( void ) ( n );
        *libcss_node_data = nullptr;

        return CSS_OK;
    }

}