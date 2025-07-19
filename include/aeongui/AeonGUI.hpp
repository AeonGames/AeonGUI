/*
Copyright (C) 2013,2019,2020,2023,2025 Rodrigo Jose Hernandez Cordoba

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

#ifndef AEONGAMES_AEONGUI_H
#define AEONGAMES_AEONGUI_H
#include "aeongui/Platform.hpp"

namespace AeonGUI
{
    /*! \brief Initializes extensions and global resources required by the library.
        \return true if initialization succeded, false if not.
        \sa Finalize
    */
    DLL bool Initialize ( int argc, char *argv[] );
    /*! \brief Finalizes any global resources allocated by Initialize.
        \sa Initialize
    */
    DLL void Finalize();
}
#endif