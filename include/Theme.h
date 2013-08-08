#ifndef AEONGAMES_THEME_H
#define AEONGAMES_THEME_H
/******************************************************************************
Copyright 2013 Rodrigo Hernandez Cordoba

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
#include "Image.h"
namespace AeonGUI
{
    class Theme
    {
    public:
        Theme();
        ~Theme();
        Image* GetCaptionedWindowImage();
        void SetCaptionedWindowImage ( Image* image );
    private:
        Image* cwf_image; ///< The Image used for captioned window frames.
    };
}
#endif