/*
Copyright (C) 2019,2020 Rodrigo Jose Hernandez Cordoba

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
#ifndef AEONGUI_DOCUMENT_H
#define AEONGUI_DOCUMENT_H
#include <cstdint>
#include <vector>
#include <memory>
#include <algorithm>
#include "aeongui/Platform.h"
#include "aeongui/Canvas.h"
#include "aeongui/JavaScript.h"
#include "aeongui/Duktape.h"
#include "dom/Node.h"

namespace AeonGUI
{
    class Document
    {
    public:
        DLL Document();
        DLL Document ( const std::string& aFilename );
        DLL ~Document();
        DLL void Draw ( Canvas& aCanvas ) const;
        /**DOM Properties and Methods @{*/
        DLL Node* documentElement();
        /**@}*/
    private:
        Duktape mJavaScript{};
        std::unique_ptr<Node> mDocumentElement{};
    };
}
#endif
