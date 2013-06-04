/******************************************************************************
Copyright 2010-2012 Rodrigo Hernandez Cordoba

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
******************************************************************************/
#include <ft2build.h>
#include FT_FREETYPE_H
#include FT_BDF_H
#include <cstdio>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <iomanip>
#include <cmath>
#include <cassert>
#include <vector>
#include <climits>
#include "pcx.h"
#include "fontstructs.h"

/*! \file
    \brief Creates raster fonts out of TrueType font files.
    \author Rodrigo Hernandez
*/
/// Program Class.
class Font2C
{
public:
    /*! \brief Default constructor.

        Stores argument data;
        \param _argc [in] Argument count.
        \param _argv [in] Argument array;
    */
    Font2C ( int32_t _argc, char **_argv ) :
        argc (_argc),
        argv (_argv),
		bytesperline(24),
	    freetype2(NULL),
	    face(NULL),
		fontsize(12),
		glyphs_per_row(0),
		glyphs_per_column(0),
		pixels(NULL),
		poweroftwo(false),
		buffersize(0)
    {
        header.id[0] = 'A';
        header.id[1] = 'E';
        header.id[2] = 'O';
        header.id[3] = 'N';
        header.id[4] = 'F';
        header.id[5] = 'N';
        header.id[6] = 'T';
        header.id[7] = 0;
        header.version[0] = 0;
        header.version[1] = 1;
        header.glyphcount = 0;
		fontfile.clear();
    }
    ~Font2C()
    {
        if ( pixels != NULL )
        {
            delete[] pixels;
            pixels = NULL;
        }
    }
    /*! \brief Print usage message.
        \param executable [in] Executable file name.
    */
    inline void Usage ( char* executable )
    {
        std::cout << "font2c version 0.2" << std::endl;
        std::cout << "(c) 2010-2011 Rodrigo Hernandez" << std::endl;
        std::cout << "Usage:" << std::endl;
        std::cout << executable << " [-s|--fontsize <size>] [-b|--bytesperline <count>] [-o|--output <filename>] [-p|--poweroftwo <filename>] <TrueType font file>" << std::endl;
        std::cout.flush();
    }
    /*! \brief Checks for proper arguments.

        Calls Font2C::Usage if arguments are incorrect.
        \return true if arguments are correct, false if not.
        \sa Font2C::Usage
    */
    inline bool ProcessArgs()
    {
        char drive[FILENAME_MAX] = "";
        char directory[FILENAME_MAX] = "";
        char filename[FILENAME_MAX] = "";
#if WIN32
        char extension[FILENAME_MAX] = "";
#endif
        for ( int32_t i = 1; i < argc; ++i )
        {
            if ( argv[i][0] == '-' )
            {
                if ( argv[i][1] == '-' )
                {
                    if ( strcmp ( &argv[i][2], "bytesperline" ) == 0 )
                    {
                        i++;
                        bytesperline = atol ( argv[i] );
                    }
                    else if ( strcmp ( &argv[i][2], "fontsize" ) == 0 )
                    {
                        i++;
                        fontsize = ( FT_UInt ) atol ( argv[i] );
                    }
                    else if ( strcmp ( &argv[i][2], "output" ) == 0 )
                    {
                        i++;
                        output = argv[i];
                    }
                    else if ( strcmp ( &argv[i][2], "poweroftwo" ) == 0 )
                    {
                        poweroftwo = true;
                    }
                }
                else
                {
                    switch ( argv[i][1] )
                    {
                    case 'b':
                        i++;
                        bytesperline = atol ( argv[i] );
                        break;
                    case 's':
                        i++;
                        fontsize = ( FT_UInt ) atol ( argv[i] );
                        break;
                    case 'o':
                        i++;
                        output = argv[i];
                        break;
                    case 'p':
                        poweroftwo = true;
                        break;
                    }
                }
            }
            else
            {
                fontfile = argv[i];
            }
        }
        if ( fontfile.empty() )
        {
            return false;
        }
#ifdef _WIN32
        _splitpath ( fontfile.c_str(), drive, directory, filename, extension );
#else
        sprintf ( filename, "%s", basename ( fontfile.c_str() ) );
        //sprintf(directory,"%s",dirname(fontfile.c_str()));
        filename[strlen ( filename ) - 4] = 0;
#endif
        if ( output.empty() )
        {
            std::ostringstream stream;
            stream << drive << directory << filename;
            output = stream.str();
        }
        fontfilename = filename;
        return true;
    }
    inline bool Run()
    {
        if ( !ProcessArgs() )
        {
            Usage ( argv[0] );
            return false;
        }
        std::cout << "Generating C files for " << fontfile << "." << std::endl;
        if ( FT_Init_FreeType ( &freetype2 ) )
        {
            std::cerr << "Could not initialize FreeType." << std::endl;
            return false;
        }
        if ( !LoadFont ( fontfile ) )
        {
            std::cerr << "Could not load font, make sure the path is correct and the font type is supported by FreeType." << std::endl;
            FT_Done_FreeType ( freetype2 );
            return false;
        }
        CreateRasterFont();
        CalculateBufferSize();
        WriteFNT();
        WriteImage();
        WriteH();
        WriteC();
        std::ifstream file;
        std::ofstream hfile;
        std::ofstream cfile;
        UnloadFont();
        FT_Done_FreeType ( freetype2 );
        return true;
    }
private:
    static bool compare_glyphs ( const FNTGlyph& a, const FNTGlyph& b )
    {
        return a.charcode < b.charcode;
    }
    inline int32_t ceil266 ( FT_Pos x )
    {
#if 0
        return ( ( ( x + 63 ) & -64 ) / 64 );
#else
        return x >> 6;
#endif
    }
    inline bool LoadFont ( std::string& fontfilename )
    {
        uint32_t area;
        uint32_t side;
        if ( FT_New_Face ( freetype2, fontfilename.c_str(), 0, &face ) )
        {
            return false;
        }
        if ( FT_Set_Pixel_Sizes ( face, 0, fontsize ) )
        {
            FT_Done_Face ( face );
            return false;
        }
        // From this point on, this function does a lot of stuff it shouldn't
        // Store Nominal Sizes
        header.nominal_width  = face->size->metrics.x_ppem;
        header.nominal_height = face->size->metrics.y_ppem;
        header.ascender       = static_cast<int16_t> ( ceil266 ( face->size->metrics.ascender ) );
        header.descender      = static_cast<int16_t> ( ceil266 ( face->size->metrics.descender ) );
        header.max_advance    = static_cast<int16_t> ( ceil266 ( face->size->metrics.max_advance ) );
        header.height         = static_cast<uint16_t> ( ceil266 ( face->size->metrics.height ) );
#if 0
        // iterate to find real glyph count because face->num_glyphs is unreliable (usually a higher number)
        header.glyphcount = 0;
        charcode = FT_Get_First_Char ( face, &gindex );
        if ( gindex == 0 )
        {
            FT_Done_Face ( face );
            return false;
        }
        while ( gindex != 0 )
        {
            header.glyphcount++;
            charcode = FT_Get_Next_Char ( face, charcode, &gindex );
        }
#endif
        area = ( face->size->metrics.x_ppem * face->size->metrics.y_ppem * face->num_glyphs );
        side = ( uint32_t ) ( ceil ( sqrt ( double ( area ) ) ) );
        // Calculate image width for a square image
        header.map_width  = side + ( face->size->metrics.x_ppem - ( side % face->size->metrics.x_ppem ) );
        if ( poweroftwo )
        {
            // Recalculate width based on previous width
            header.map_width = 1 << ( int32_t ) ceil ( ( log ( ( double ) header.map_width ) / log ( 2.0f ) ) + 0.5f );
            glyphs_per_row = header.map_width / face->size->metrics.x_ppem;
            // Calculate height based on glyphs per row
            header.map_height = ( int32_t ) ceil ( ( float ( face->num_glyphs ) / float ( glyphs_per_row ) ) * face->size->metrics.y_ppem );
            header.map_height = 1 << ( int32_t ) ceil ( ( log ( ( double ) header.map_height ) / log ( 2.0f ) ) + 0.5f );
        }
        else
        {
            glyphs_per_row = header.map_width / face->size->metrics.x_ppem;
            header.map_height = side + ( face->size->metrics.y_ppem - ( side % face->size->metrics.y_ppem ) );
        }
        std::cout << "Glyph count: "   << face->num_glyphs << std::endl;
        std::cout << "Charmap count: " << face->num_charmaps << std::endl;
        std::cout << "Face count: "    << face->num_faces << std::endl;
        ///\todo Add the following 3 fields to the font file
        std::cout << "Ascender: "      << ceil266 ( face->size->metrics.ascender ) << std::endl;
        std::cout << "Descender: "     << ceil266 ( face->size->metrics.descender ) << std::endl;
        std::cout << "Height: "        << ceil266 ( face->size->metrics.height ) << std::endl;
        std::cout << "Max Advance: "   << ceil266 ( face->size->metrics.max_advance ) << std::endl;
#if 0
        std::cout << "Bounding Box: "  << ceil266 ( face->bbox.xMin ) << ","
                  << ceil266 ( face->bbox.yMin ) << ","
                  << ceil266 ( face->bbox.xMax ) << ","
                  << ceil266 ( face->bbox.yMax ) << std::endl;
#endif
        std::cout << "Fixed size count: " << face->num_fixed_sizes << std::endl;
        for ( int32_t i = 0; i < face->num_fixed_sizes; ++i )
        {
            std::cout << "Size: " << face->available_sizes[i].width << ", " << face->available_sizes[i].height << std::endl;
        }
        return true;
    }
    inline void UnloadFont()
    {
        FT_Done_Face ( face );
    }
    void CreateRasterFont()
    {
        FNTGlyph glyph;
        FT_UInt   gindex;
        uint8_t* pixel;
        uint8_t* pos;
        //FT_Long glyphx = 0;
        //FT_Long glyphy = 0;
        unsigned short glyphx = 0;
        unsigned short glyphy = 0;
        unsigned short glyphs_taken;
        if ( pixels != NULL )
        {
            delete[] pixels;
        }
        pixels = new uint8_t[header.map_width * header.map_height];
        // clear to black
        memset ( reinterpret_cast<void*>(pixels), 0, static_cast<unsigned long int>(sizeof ( uint8_t ) *header.map_width * header.map_height ));
        pos = pixels;
        glyph.charcode = wchar_t ( FT_Get_First_Char ( face, &gindex ) );
        int32_t printableglyphs = 0;
        while ( gindex != 0 )
        {
            FT_Load_Glyph ( face, gindex, FT_LOAD_RENDER );
            // Advance should be set even if no bitmap exists
            glyph.advance[0] = ceil266 ( face->glyph->advance.x );
            glyph.advance[1] = ceil266 ( face->glyph->advance.y );
            if ( face->glyph->format == FT_GLYPH_FORMAT_BITMAP )
            {
                if ( face->glyph->bitmap.buffer != NULL )
                {
                    // bitmap width may be larger than metrics.x_ppem
                    glyphs_taken = face->glyph->bitmap.width / face->size->metrics.x_ppem;
                    if ( ( face->glyph->bitmap.width % face->size->metrics.x_ppem ) > 0 )
                    {
                        glyphs_taken++;
                    }
                    // Grab top and left offsets
                    glyph.top = face->glyph->bitmap_top;
                    glyph.left = face->glyph->bitmap_left;
                    // Save dimensions
                    glyph.width = face->glyph->bitmap.width;
                    glyph.height = face->glyph->bitmap.rows;
                    // Calculate glyph min/max
                    glyph.min[0] = glyphx * face->size->metrics.x_ppem;
                    glyph.min[1] = glyphy * face->size->metrics.y_ppem;
                    glyph.max[0] = glyph.min[0] + ( face->glyph->bitmap.width );
                    glyph.max[1] = glyph.min[1] + ( face->glyph->bitmap.rows );
                    // Normalize glyph min/max
                    glyph.normalized_min[0] = float ( glyph.min[0] ) / float ( header.map_width );
                    glyph.normalized_min[1] = float ( glyph.min[1] ) / float ( header.map_height );
                    glyph.normalized_max[0] = float ( glyph.max[0] ) / float ( header.map_width );
                    glyph.normalized_max[1] = float ( glyph.max[1] ) / float ( header.map_height );
                    pos = pixels +
                          ( glyph.min[1] * header.map_width ) +
                          ( glyph.min[0] );
                    printableglyphs++;
                    glyphx += glyphs_taken;
                    pixel = face->glyph->bitmap.buffer;
                    for ( int32_t py = 0; py < face->glyph->bitmap.rows; ++py )
                    {
                        memcpy ( reinterpret_cast<void*>(pos + ( py * header.map_width )), pixel, sizeof ( uint8_t ) *face->glyph->bitmap.width );
                        pixel += face->glyph->bitmap.pitch;
                    }
                    if ( glyphx == glyphs_per_row )
                    {
                        glyphx = 0;
                        glyphy++;
                    }
                }
                else
                {
                    std::cout << "NO BITMAP BUFFER " << int32_t ( glyph.charcode ) << std::endl;
                    // Zero out all bitmap related values
                    glyph.width = 0;
                    glyph.height = 0;
                    glyph.min[0] = 0;
                    glyph.min[1] = 0;
                    glyph.max[0] = 0;
                    glyph.max[1] = 0;
                    glyph.normalized_min[0] = 0.0f;
                    glyph.normalized_min[1] = 0.0f;
                    glyph.normalized_max[0] = 0.0f;
                    glyph.normalized_max[1] = 0.0f;
                    glyph.top = 0;
                    glyph.left = 0;
                }
            }
            else
            {
                std::cout << "NO BITMAP GLYPH (this should never happen!)" << std::endl;
                assert ( face->glyph->format == FT_GLYPH_FORMAT_BITMAP );
            }
            glyphs.push_back ( glyph );
            glyph.charcode = wchar_t ( FT_Get_Next_Char ( face, static_cast<unsigned long int>(glyph.charcode), &gindex ) );
        }
        // Sorting by charcode is probably unnecessary, but better safe than sorry
        std::sort ( glyphs.begin(), glyphs.end(), compare_glyphs );
        // Set Glyph count
        header.glyphcount = static_cast<uint32_t>(glyphs.size());
#if 0
        for ( std::vector<FNTGlyph>::iterator i = glyphs.begin(); i != glyphs.end(); ++i )
        {
            std::cout << "Charcode " << i->charcode << std::endl;
        }
#endif
        std::cout << "Printable Glyphs " << printableglyphs << std::endl;
        std::cout << "REAL Glyph count: " << header.glyphcount << std::endl;
    }
    bool WriteFNT()
    {
        std::ofstream fnt;
        std::string fntfilename ( output + ".fnt" );
        fnt.open ( fntfilename.c_str(), std::ios_base::out | std::ios_base::binary );
        if ( !fnt.is_open() )
        {
            std::cerr << "Problem opening " << fntfilename << " for writting." << std::endl;
            return false;
        }
        fnt.write ( ( const char* ) &header, sizeof ( FNTHeader ) );
        for ( std::vector<FNTGlyph>::iterator i = glyphs.begin(); i != glyphs.end(); ++i )
        {
            fnt.write ( ( const char* ) & ( *i ), sizeof ( FNTGlyph ) );
        }
        fnt.write ( ( const char* ) pixels, (long int) (sizeof ( char ) * ( header.map_width * header.map_height )) );
        fnt.close();
        return true;
    }
    bool WriteImage()
    {
        std::string imagename = output + ".pcx";
        Pcx pcx;
        pcx.Encode ( header.map_width, header.map_height, pixels, header.map_width * header.map_height );
        pcx.Save ( imagename.c_str() );
        return true;
    }
    bool WriteH()
    {
        assert ( buffersize != 0 );
        std::string hfilename ( output + ".h" );
        std::string upper;
        std::ofstream hfile;
        hfile.open ( hfilename.c_str() );
        if ( !hfile.is_open() )
        {
            std::cout << "Problem opening " << hfilename << " for writting." << std::endl;
            return false;
        }
        upper.resize ( fontfilename.size() );
        std::transform ( fontfilename.begin(), fontfilename.end(), upper.begin(), toupper );
        std::cout << "Writting output to " << hfilename << " please wait." << std::endl;
        hfile << "#ifndef " << upper << "_H" << std::endl;
        hfile << "#define " << upper << "_H" << std::endl;
        hfile << "#ifdef __cplusplus" << std::endl;
        hfile << "#include <cstdint>" << std::endl;
        hfile << "extern \"C\" {" << std::endl;
        hfile << "#else" << std::endl;
        hfile << "#include <stdint.h>" << std::endl;
        hfile << "#endif" << std::endl;
        hfile << "extern struct {" << std::endl;
        hfile << "uint32_t  size;" << std::endl;
        hfile << "uint8_t data" << "[" << buffersize << "];" << std::endl;
        hfile << "} " << fontfilename << ";" << std::endl;
        hfile << "#ifdef __cplusplus" << std::endl;
        hfile << "}" << std::endl;
        hfile << "#endif" << std::endl;
        hfile << "#endif" << std::endl << std::endl;
        hfile.close();
        return true;
    };
    bool WriteC()
    {
        assert ( buffersize != 0 );
        assert ( pixels != NULL );
        uint8_t* pixel = pixels;
        uint8_t* byte = NULL;
        uint32_t i;
        std::string cfilename ( output + ".c" );
        std::ofstream cfile;
        cfile.open ( cfilename.c_str() );
        if ( !cfile.is_open() )
        {
            std::cout << "Problem opening " << cfilename << " for writting." << std::endl;
            return false;
        }
        std::cout << "Writting output to " << cfilename << " please wait." << std::endl;
        //------------------------------------------------------------------------------
        cfile << "#include <stdint.h>" << std::endl;
        cfile << "struct {" << std::endl;
        cfile << "uint32_t  size;" << std::endl;
        cfile << "uint8_t data" << "[" << buffersize << "];" << std::endl;
        cfile << "} " << fontfilename << "={" << std::endl;
        cfile << buffersize << "," << std::endl;
#if 0
        cfile << std::endl << "\"";
        byte = ( uint8_t* ) &header;
        for ( uint32_t i = 0; i < sizeof ( FNTHeader ); ++i )
        {
            cfile << "\\x" << std::setfill ( '0' ) << std::setw ( 2 ) << std::hex << ( int32_t ) byte[i];
        }
        cfile << "\"" << std::endl;
        for ( std::vector<FNTGlyph>::iterator i = glyphs.begin(); i != glyphs.end(); ++i )
        {
            cfile << "\"";
            byte = ( uint8_t* ) & ( *i );
            for ( uint32_t j = 0; j < sizeof ( FNTGlyph ); ++j )
            {
                cfile << "\\x" << std::setfill ( '0' ) << std::setw ( 2 ) << std::hex << ( int32_t ) byte[j];
            }
            cfile << "\"" << std::endl;
        }
        //------------------------------------------------------------------------------
        for ( uint32_t y = 0; y < header.map_height; ++y )
        {
            cfile << "\"";
            for ( uint32_t x = 0; x < header.map_width; ++x )
            {
                cfile << "\\x" << std::setfill ( '0' ) << std::setw ( 2 ) << std::hex << ( int32_t ) *pixel;
                pixel++;
            }
            cfile << "\"" << std::endl;
        }
        cfile << ";" << std::endl;
#else
        cfile << "{" << std::endl;
        byte = ( uint8_t* ) &header;
        for ( i = 0; i < sizeof ( FNTHeader ); ++i )
        {
            //cfile << "0x" << std::setfill('0') << std::setw(1) << std::hex << (int32_t) byte[i] << ",";
            cfile << std::dec << ( int32_t ) byte[i] << ",";
        }
        cfile << std::endl;
        for ( std::vector<FNTGlyph>::iterator i = glyphs.begin(); i != glyphs.end(); ++i )
        {
#if 0
            std::wcout << "Charcode: " << i->charcode << std::endl;
            std::cout << "Width      " << i->width << std::endl;
            std::cout << "Height:    " << i->height << std::endl;
            std::cout << "Minimum X: " << i->min[0] << std::endl;
            std::cout << "Minimum Y: " << i->min[1] << std::endl;
            std::cout << "Maximum X: " << i->max[0] << std::endl;
            std::cout << "Maximum Y: " << i->max[1] << std::endl;
            std::cout << "Normalized Minimum X: " << i->normalized_min[0] << std::endl;
            std::cout << "Normalized Minimum Y: " << i->normalized_min[1] << std::endl;
            std::cout << "Normalized Maximum X: " << i->normalized_max[0] << std::endl;
            std::cout << "Normalized Maximum Y: " << i->normalized_max[1] << std::endl;
            std::cout << "Top: " << i->top << std::endl;
            std::cout << "Left: " << i->left << std::endl;
            std::cout << "Horizontal Advance: " << i->advance[0] << std::endl;
            std::cout << "Vertical Advance: " << i->advance[1] << std::endl;
#endif
            byte = ( uint8_t* ) & ( *i );
            for ( uint32_t j = 0; j < sizeof ( FNTGlyph ); ++j )
            {
                //cfile << "0x" << std::setfill('0') << std::setw(1) << std::hex << (int32_t) byte[j] << ",";
                cfile << std::dec << ( int32_t ) byte[j] << ",";
            }
            cfile << std::endl;
        }
        //------------------------------------------------------------------------------
        for ( uint32_t y = 0; y < header.map_height; ++y )
        {
            for ( uint32_t x = 0; x < header.map_width; ++x )
            {
                //cfile << "0x" << std::setfill('0') << std::setw(1) << std::hex << (int32_t) *pixel << ",";
                cfile << std::dec << ( int32_t ) *pixel << ",";
                pixel++;
            }
            cfile << std::endl;
        }
        cfile << "}};" << std::endl << std::endl;
#endif
        cfile.close();
        return true;
    };
    inline void CalculateBufferSize()
    {
        buffersize = sizeof ( FNTHeader ) + ( sizeof ( FNTGlyph ) * glyphs.size() ) + ( header.map_width * header.map_height );
    }
    /// Program argument count.
    int32_t argc;
    /// Program argument array.
    char **argv;
    int32_t bytesperline;
    std::string fontfile;
    std::string fontfilename;
    std::string output;
    FT_Library freetype2;
    FT_Face face;
    FT_UInt fontsize;
    uint32_t glyphs_per_row;
    uint32_t glyphs_per_column;
    uint8_t* pixels;
    bool poweroftwo;
    size_t buffersize;
    FNTHeader header;
    std::vector<FNTGlyph> glyphs;
};
/*! \brief Program entry point.
    \param argc [in] Argument count.
    \param argv [in] Argumens.
    \return Zero on success, minus one on failure.
*/
int32_t main ( int32_t argc, char **argv )
{
    std::cout << "sizeof(FNTHeader) = " << sizeof ( FNTHeader ) << std::endl;
    std::cout << "sizeof(FNTGlyph) = " << sizeof ( FNTGlyph ) << std::endl;
    Font2C bin2hex ( argc, argv );
    if ( !bin2hex.Run() )
    {
        return -1;
    }
    return 0;
}
