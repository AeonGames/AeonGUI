#ifndef AEONGAMES_AEONGUI_H
#define AEONGAMES_AEONGUI_H
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
#include "Platform.h"
namespace AeonGUI
{
    /*! \brief Initializes extensions and global resources required by the library.
        \return true if initialization succeded, false if not.
        \sa Finalize
    */
    bool DLL Initialize();
    /*! \brief Finalizes any global resources allocated by Initialize.
        \sa Initialize
    */
    void DLL Finalize();
}
#endif