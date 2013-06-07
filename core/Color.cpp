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
#include "Color.h"
#include <algorithm>
namespace AeonGUI
{
    void Color::Blend ( Color src )
    {
        if ( ( src.a == 255 ) )
        {
            /*  Full source opacity
                do a simple replacement*/
            bgra = src.bgra;
        }
		/*	If the source alpha is 0 
			the destination color is unchanged */
        else if(src.a > 0)
        {
            float sfactor = ( static_cast<float> ( src.a ) / 255.0f );
            float dfactor = 1.0f - sfactor;
            r = static_cast<uint8_t> ( std::min ( 255.0f, ( static_cast<float> ( src.r ) * sfactor + static_cast<float> ( r ) * dfactor ) ) );
            g = static_cast<uint8_t> ( std::min ( 255.0f, ( static_cast<float> ( src.g ) * sfactor + static_cast<float> ( g ) * dfactor ) ) );
            b = static_cast<uint8_t> ( std::min ( 255.0f, ( static_cast<float> ( src.b ) * sfactor + static_cast<float> ( b ) * dfactor ) ) );
            // Just acumulate alpha, if something looks odd try multipling the second addend by dfactor.
            a = static_cast<uint8_t> ( std::min ( 255.0f, ( static_cast<float> ( src.a ) + static_cast<float> ( a ) ) ) );
        }
    }
}
