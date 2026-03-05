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

#include "aeongui/RasterImage.hpp"

#include <array>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>

#if defined ( USE_PNG ) && USE_PNG
#include <png.h>
#endif

#if defined ( USE_JPEG ) && USE_JPEG
#include <jpeglib.h>
#include <setjmp.h>
#endif

namespace AeonGUI
{
    namespace
    {
        static uint16_t ReadUint16LE ( const uint8_t* aData )
        {
            return static_cast<uint16_t> ( aData[0] ) | ( static_cast<uint16_t> ( aData[1] ) << 8u );
        }

        RasterImage::EncodedFormat GuessFormatFromMagic ( const uint8_t* aData, size_t aSize )
        {
            if ( aData == nullptr || aSize < 3 )
            {
                return RasterImage::EncodedFormat::Unknown;
            }

            static constexpr std::array<uint8_t, 8> pngMagic{0x89u, 0x50u, 0x4Eu, 0x47u, 0x0Du, 0x0Au, 0x1Au, 0x0Au};
            if ( aSize >= pngMagic.size() && std::memcmp ( aData, pngMagic.data(), pngMagic.size() ) == 0 )
            {
                return RasterImage::EncodedFormat::PNG;
            }
            if ( aData[0] == 0xFFu && aData[1] == 0xD8u && aData[2] == 0xFFu )
            {
                return RasterImage::EncodedFormat::JPEG;
            }
            if ( aSize >= 128u && aData[0] == 0x0Au && aData[2] == 0x01u && aData[3] == 0x08u )
            {
                return RasterImage::EncodedFormat::PCX;
            }
            return RasterImage::EncodedFormat::Unknown;
        }

#if defined ( USE_PNG ) && USE_PNG
        bool DecodePngToRgba ( const uint8_t* aData, size_t aSize, std::vector<uint8_t>& aPixels, uint32_t& aWidth, uint32_t& aHeight )
        {
            png_image image{};
            image.version = PNG_IMAGE_VERSION;
            if ( !png_image_begin_read_from_memory ( &image, aData, aSize ) )
            {
                return false;
            }

            image.format = PNG_FORMAT_RGBA;
            std::vector<uint8_t> rgba ( PNG_IMAGE_SIZE ( image ) );
            if ( !png_image_finish_read ( &image, nullptr, rgba.data(), 0, nullptr ) )
            {
                png_image_free ( &image );
                return false;
            }
            png_image_free ( &image );

            aWidth = image.width;
            aHeight = image.height;
            aPixels.swap ( rgba );
            return true;
        }
#endif

#if defined ( USE_JPEG ) && USE_JPEG
        struct JpegErrorManager
        {
            jpeg_error_mgr pub;
            jmp_buf jump;
        };

        extern "C" void JpegErrorExit ( j_common_ptr aCommonPtr )
        {
            JpegErrorManager* error = reinterpret_cast<JpegErrorManager*> ( aCommonPtr->err );
            longjmp ( error->jump, 1 );
        }

        bool DecodeJpegToRgba ( const uint8_t* aData, size_t aSize, std::vector<uint8_t>& aPixels, uint32_t& aWidth, uint32_t& aHeight )
        {
            jpeg_decompress_struct info{};
            JpegErrorManager error{};
            info.err = jpeg_std_error ( &error.pub );
            error.pub.error_exit = JpegErrorExit;
            if ( setjmp ( error.jump ) )
            {
                jpeg_destroy_decompress ( &info );
                return false;
            }

            jpeg_create_decompress ( &info );
            jpeg_mem_src ( &info, aData, static_cast<unsigned long> ( aSize ) );
            jpeg_read_header ( &info, TRUE );
            jpeg_start_decompress ( &info );

            aWidth = info.output_width;
            aHeight = info.output_height;
            const size_t rowStride = static_cast<size_t> ( aWidth ) * info.output_components;
            std::vector<uint8_t> row ( rowStride );
            std::vector<uint8_t> rgba ( static_cast<size_t> ( aWidth ) * static_cast<size_t> ( aHeight ) * 4u );

            while ( info.output_scanline < info.output_height )
            {
                uint8_t* rowPtr = row.data();
                jpeg_read_scanlines ( &info, &rowPtr, 1 );

                const uint32_t y = info.output_scanline - 1;
                for ( uint32_t x = 0; x < aWidth; ++x )
                {
                    const size_t src = static_cast<size_t> ( x ) * info.output_components;
                    const size_t dst = ( static_cast<size_t> ( y ) * static_cast<size_t> ( aWidth ) + static_cast<size_t> ( x ) ) * 4u;
                    if ( info.output_components == 3 )
                    {
                        rgba[dst + 0] = row[src + 0];
                        rgba[dst + 1] = row[src + 1];
                        rgba[dst + 2] = row[src + 2];
                    }
                    else if ( info.output_components == 1 )
                    {
                        rgba[dst + 0] = row[src + 0];
                        rgba[dst + 1] = row[src + 0];
                        rgba[dst + 2] = row[src + 0];
                    }
                    else
                    {
                        jpeg_finish_decompress ( &info );
                        jpeg_destroy_decompress ( &info );
                        return false;
                    }
                    rgba[dst + 3] = 255u;
                }
            }

            jpeg_finish_decompress ( &info );
            jpeg_destroy_decompress ( &info );
            aPixels.swap ( rgba );
            return true;
        }
#endif

        bool DecodePcxToRgba ( const uint8_t* aData, size_t aSize, std::vector<uint8_t>& aPixels, uint32_t& aWidth, uint32_t& aHeight )
        {
            if ( aData == nullptr || aSize < 128u )
            {
                return false;
            }
            if ( aData[0] != 0x0Au || aData[1] != 5u || aData[2] != 1u || aData[3] != 8u )
            {
                return false;
            }

            const uint16_t xStart = ReadUint16LE ( aData + 4u );
            const uint16_t yStart = ReadUint16LE ( aData + 6u );
            const uint16_t xEnd = ReadUint16LE ( aData + 8u );
            const uint16_t yEnd = ReadUint16LE ( aData + 10u );
            if ( xEnd < xStart || yEnd < yStart )
            {
                return false;
            }

            const uint8_t numBitPlanes = aData[65];
            const uint16_t bytesPerLine = ReadUint16LE ( aData + 66u );
            if ( ( numBitPlanes != 1u && numBitPlanes != 3u && numBitPlanes != 4u ) || bytesPerLine == 0u )
            {
                return false;
            }

            aWidth = static_cast<uint32_t> ( xEnd - xStart ) + 1u;
            aHeight = static_cast<uint32_t> ( yEnd - yStart ) + 1u;
            const uint32_t scanlineLength = static_cast<uint32_t> ( numBitPlanes ) * static_cast<uint32_t> ( bytesPerLine );

            std::vector<uint8_t> decoded ( static_cast<size_t> ( scanlineLength ) * static_cast<size_t> ( aHeight ), 0u );
            size_t offset = 128u;
            for ( uint32_t row = 0; row < aHeight; ++row )
            {
                for ( uint8_t plane = 0; plane < numBitPlanes; ++plane )
                {
                    uint8_t* pixel = decoded.data() + ( static_cast<size_t> ( scanlineLength ) * static_cast<size_t> ( row ) ) + plane;
                    uint16_t written = 0u;
                    while ( written < bytesPerLine )
                    {
                        if ( offset >= aSize )
                        {
                            return false;
                        }

                        uint8_t value = 0u;
                        uint8_t count = 1u;
                        const uint8_t byte = aData[offset++];
                        if ( ( byte & 0xC0u ) == 0xC0u )
                        {
                            count = byte & 0x3Fu;
                            if ( count == 0u || offset >= aSize )
                            {
                                return false;
                            }
                            value = aData[offset++];
                        }
                        else
                        {
                            value = byte;
                        }

                        for ( uint8_t i = 0u; i < count && written < bytesPerLine; ++i, ++written )
                        {
                            *pixel = value;
                            pixel += numBitPlanes;
                        }
                    }
                }
            }

            std::vector<uint8_t> rgba ( static_cast<size_t> ( aWidth ) * static_cast<size_t> ( aHeight ) * 4u, 255u );
            for ( uint32_t y = 0; y < aHeight; ++y )
            {
                const uint8_t* row = decoded.data() + ( static_cast<size_t> ( y ) * static_cast<size_t> ( scanlineLength ) );
                for ( uint32_t x = 0; x < aWidth; ++x )
                {
                    const size_t src = static_cast<size_t> ( x ) * static_cast<size_t> ( numBitPlanes );
                    const size_t dst = ( static_cast<size_t> ( y ) * static_cast<size_t> ( aWidth ) + static_cast<size_t> ( x ) ) * 4u;
                    if ( numBitPlanes == 1u )
                    {
                        rgba[dst + 0u] = row[src + 0u];
                        rgba[dst + 1u] = row[src + 0u];
                        rgba[dst + 2u] = row[src + 0u];
                    }
                    else
                    {
                        rgba[dst + 0u] = row[src + 0u];
                        rgba[dst + 1u] = row[src + 1u];
                        rgba[dst + 2u] = row[src + 2u];
                        if ( numBitPlanes == 4u )
                        {
                            rgba[dst + 3u] = row[src + 3u];
                        }
                    }
                }
            }

            aPixels.swap ( rgba );
            return true;
        }
    }

    RasterImage::RasterImage() :
        mEncodedFormat ( EncodedFormat::Unknown ),
        mPixelFormat ( PixelFormat::Unknown ),
        mWidth ( 0 ),
        mHeight ( 0 )
    {
    }

    bool RasterImage::LoadFromFile ( const std::string& aPath )
    {
        std::ifstream file ( aPath, std::ios_base::binary | std::ios_base::in );
        if ( !file.is_open() )
        {
            return false;
        }

        file.seekg ( 0, std::ios_base::end );
        const std::streampos end = file.tellg();
        file.seekg ( 0, std::ios_base::beg );
        if ( end <= 0 )
        {
            return false;
        }

        std::vector<uint8_t> bytes ( static_cast<size_t> ( end ) );
        file.read ( reinterpret_cast<char*> ( bytes.data() ), static_cast<std::streamsize> ( bytes.size() ) );
        if ( !file )
        {
            return false;
        }

        return LoadFromMemory ( bytes.data(), bytes.size() );
    }

    bool RasterImage::LoadFromMemory ( const void* aData, size_t aSize )
    {
        if ( aData == nullptr || aSize < 3 )
        {
            return false;
        }

        const uint8_t* bytes = static_cast<const uint8_t*> ( aData );
        EncodedFormat format = GuessFormatFromMagic ( bytes, aSize );

        std::vector<uint8_t> rgba;
        uint32_t width = 0;
        uint32_t height = 0;
        bool loaded = false;

        switch ( format )
        {
        case EncodedFormat::PNG:
#if defined ( USE_PNG ) && USE_PNG
            loaded = DecodePngToRgba ( bytes, aSize, rgba, width, height );
#endif
            break;
        case EncodedFormat::JPEG:
#if defined ( USE_JPEG ) && USE_JPEG
            loaded = DecodeJpegToRgba ( bytes, aSize, rgba, width, height );
#endif
            break;
        case EncodedFormat::PCX:
            loaded = DecodePcxToRgba ( bytes, aSize, rgba, width, height );
            break;
        case EncodedFormat::Unknown:
            break;
        }

        if ( !loaded )
        {
            return false;
        }

        if ( IsLoaded() )
        {
            std::fprintf ( stderr,
                           "RasterImage: replacing image %ux%u (%zu bytes) with %ux%u (%zu bytes).\n",
                           mWidth,
                           mHeight,
                           mPixelData.size(),
                           width,
                           height,
                           rgba.size() );
        }

        mEncodedFormat = format;
        mPixelFormat = PixelFormat::RGBA8;
        mWidth = width;
        mHeight = height;
        mPixelData.swap ( rgba );
        return true;
    }

    void RasterImage::Clear()
    {
        mEncodedFormat = EncodedFormat::Unknown;
        mPixelFormat = PixelFormat::Unknown;
        mWidth = 0;
        mHeight = 0;
        mPixelData.clear();
    }

    bool RasterImage::IsLoaded() const
    {
        return mPixelFormat != PixelFormat::Unknown && mWidth > 0 && mHeight > 0 && !mPixelData.empty();
    }

    RasterImage::EncodedFormat RasterImage::GetEncodedFormat() const
    {
        return mEncodedFormat;
    }

    RasterImage::PixelFormat RasterImage::GetPixelFormat() const
    {
        return mPixelFormat;
    }

    uint32_t RasterImage::GetWidth() const
    {
        return mWidth;
    }

    uint32_t RasterImage::GetHeight() const
    {
        return mHeight;
    }

    size_t RasterImage::GetStride() const
    {
        return static_cast<size_t> ( mWidth ) * 4u;
    }

    const uint8_t* RasterImage::GetPixels() const
    {
        return mPixelData.empty() ? nullptr : mPixelData.data();
    }

    const std::vector<uint8_t>& RasterImage::GetPixelData() const
    {
        return mPixelData;
    }
}
