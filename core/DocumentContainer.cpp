#include "DocumentContainer.h"
#include "Log.h"

namespace AeonGUI
{
    DocumentContainer::DocumentContainer() : mFreeType ( nullptr )
    {
        FT_Error ft_error;
        if ( ( ft_error = FT_Init_FreeType ( &mFreeType ) ) != 0 )
        {
            AEONGUI_LOG_ERROR ( "FT_Init_FreeType returned error code 0x%02x", ft_error );
        }
    }

    DocumentContainer::~DocumentContainer()
    {
        FT_Error ft_error;
        if ( ( ft_error = FT_Done_FreeType ( mFreeType ) ) != 0 )
        {
            AEONGUI_LOG_ERROR ( "FT_Done_FreeType returned error code 0x%02x", ft_error );
        }
    }

    litehtml::uint_ptr DocumentContainer::create_font ( const litehtml::tchar_t * faceName, int size, int weight, litehtml::font_style italic, unsigned int decoration, litehtml::font_metrics * fm )
    {
        return litehtml::uint_ptr();
    }

    void DocumentContainer::delete_font ( litehtml::uint_ptr hFont )
    {
    }

    int DocumentContainer::text_width ( const litehtml::tchar_t * text, litehtml::uint_ptr hFont )
    {
        return 0;
    }

    void DocumentContainer::draw_text ( litehtml::uint_ptr hdc, const litehtml::tchar_t * text, litehtml::uint_ptr hFont, litehtml::web_color color, const litehtml::position & pos )
    {
    }

    int DocumentContainer::pt_to_px ( int pt )
    {
        return 0;
    }

    int DocumentContainer::get_default_font_size() const
    {
        return 0;
    }

    const litehtml::tchar_t * DocumentContainer::get_default_font_name() const
    {
        return nullptr;
    }

    void DocumentContainer::draw_list_marker ( litehtml::uint_ptr hdc, const litehtml::list_marker & marker )
    {
    }

    void DocumentContainer::load_image ( const litehtml::tchar_t * src, const litehtml::tchar_t * baseurl, bool redraw_on_ready )
    {
    }

    void DocumentContainer::get_image_size ( const litehtml::tchar_t * src, const litehtml::tchar_t * baseurl, litehtml::size & sz )
    {
    }

    void DocumentContainer::draw_background ( litehtml::uint_ptr hdc, const litehtml::background_paint & bg )
    {
    }

    void DocumentContainer::draw_borders ( litehtml::uint_ptr hdc, const litehtml::borders & borders, const litehtml::position & draw_pos, bool root )
    {
    }

    void DocumentContainer::set_caption ( const litehtml::tchar_t * caption )
    {
    }

    void DocumentContainer::set_base_url ( const litehtml::tchar_t * base_url )
    {
    }

    void DocumentContainer::link ( const std::shared_ptr<litehtml::document>& doc, const litehtml::element::ptr & el )
    {
    }

    void DocumentContainer::on_anchor_click ( const litehtml::tchar_t * url, const litehtml::element::ptr & el )
    {
    }

    void DocumentContainer::set_cursor ( const litehtml::tchar_t * cursor )
    {
    }

    void DocumentContainer::transform_text ( litehtml::tstring & text, litehtml::text_transform tt )
    {
    }

    void DocumentContainer::import_css ( litehtml::tstring & text, const litehtml::tstring & url, litehtml::tstring & baseurl )
    {
    }

    void DocumentContainer::set_clip ( const litehtml::position & pos, const litehtml::border_radiuses & bdr_radius, bool valid_x, bool valid_y )
    {
    }

    void DocumentContainer::del_clip()
    {
    }

    void DocumentContainer::get_client_rect ( litehtml::position & client ) const
    {
    }

    std::shared_ptr<litehtml::element> DocumentContainer::create_element ( const litehtml::tchar_t * tag_name, const litehtml::string_map & attributes, const std::shared_ptr<litehtml::document>& doc )
    {
        return std::shared_ptr<litehtml::element>();
    }

    void DocumentContainer::get_media_features ( litehtml::media_features & media ) const
    {
    }

    void DocumentContainer::get_language ( litehtml::tstring & language, litehtml::tstring & culture ) const
    {
    }
}
