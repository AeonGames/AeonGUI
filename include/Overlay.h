#ifndef AEONGUI_OVERLAY_H
#define AEONGUI_OVERLAY_H
/******************************************************************************
Copyright 2015 Rodrigo Hernandez Cordoba

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
#include "Platform.h"
/**@todo Perhaps change the overlay class so it uses the pimpl pattern
    and we can do witout the libxml dependency at this level.*/
#include <libxml/tree.h>
#include <string>
namespace AeonGUI
{
    /*! \brief Top window with borders and frame. */
    class Overlay
    {
    public:
        DLL Overlay();
        DLL ~Overlay();
        DLL bool ReadFile ( const std::string& aFileName );
    private:
        xmlDocPtr mDocument;
    };
}
#endif
