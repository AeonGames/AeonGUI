#ifndef AEONGUI_DOCUMENTCONTAINER_H
#define AEONGUI_DOCUMENTCONTAINER_H
#include "Platform.h"
#include "litehtml.h"

namespace AeonGUI
{
    class DocumentContainer : public litehtml::document_container
    {
    public:
        DLL DocumentContainer();
        DLL ~DocumentContainer();
        ///@name litehtml::document_container Overrides
        ///@{
        litehtml::uint_ptr  create_font ( const litehtml::tchar_t* faceName, int size, int weight, litehtml::font_style italic, unsigned int decoration, litehtml::font_metrics* fm ) override final;
        void                delete_font ( litehtml::uint_ptr hFont ) override final;
        int                 text_width ( const litehtml::tchar_t* text, litehtml::uint_ptr hFont ) override final;
        void                draw_text ( litehtml::uint_ptr hdc, const litehtml::tchar_t* text, litehtml::uint_ptr hFont, litehtml::web_color color, const litehtml::position& pos ) override final;
        int                 pt_to_px ( int pt ) override final;
        int                 get_default_font_size() const override final;
        const litehtml::tchar_t*    get_default_font_name() const override final;
        void                draw_list_marker ( litehtml::uint_ptr hdc, const litehtml::list_marker& marker ) override final;
        void                load_image ( const litehtml::tchar_t* src, const litehtml::tchar_t* baseurl, bool redraw_on_ready ) override final;
        void                get_image_size ( const litehtml::tchar_t* src, const litehtml::tchar_t* baseurl, litehtml::size& sz ) override final;
        void                draw_background ( litehtml::uint_ptr hdc, const litehtml::background_paint& bg ) override final;
        void                draw_borders ( litehtml::uint_ptr hdc, const litehtml::borders& borders, const litehtml::position& draw_pos, bool root ) override final;

        void                set_caption ( const litehtml::tchar_t* caption ) override final;
        void                set_base_url ( const litehtml::tchar_t* base_url ) override final;
        void                link ( const std::shared_ptr<litehtml::document>& doc, const litehtml::element::ptr& el ) override final;
        void                on_anchor_click ( const litehtml::tchar_t* url, const litehtml::element::ptr& el ) override final;
        void                set_cursor ( const litehtml::tchar_t* cursor ) override final;
        void                transform_text ( litehtml::tstring& text, litehtml::text_transform tt ) override final;
        void                import_css ( litehtml::tstring& text, const litehtml::tstring& url, litehtml::tstring& baseurl ) override final;
        void                set_clip ( const litehtml::position& pos, const litehtml::border_radiuses& bdr_radius, bool valid_x, bool valid_y ) override final;
        void                del_clip() override final;
        void                get_client_rect ( litehtml::position& client ) const override final;
        std::shared_ptr<litehtml::element>  create_element ( const litehtml::tchar_t *tag_name,
                const litehtml::string_map &attributes,
                const std::shared_ptr<litehtml::document> &doc ) override final;

        void                get_media_features ( litehtml::media_features& media ) const override final;
        void                get_language ( litehtml::tstring& language, litehtml::tstring & culture ) const override final;
        ///@}
    private:
    };
}
#endif
