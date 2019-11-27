/*
Copyright (C) 2019 Rodrigo Jose Hernandez Cordoba

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
#include "aeongui/Element.h"

extern "C"
{
    typedef struct _xmlDoc xmlDoc;
    typedef xmlDoc *xmlDocPtr;
}

namespace AeonGUI
{
    class Document
    {
    public:
        DLL Document();
        DLL Document ( const std::string& aFilename );
        DLL ~Document();
        DLL void Draw ( Canvas& aCanvas ) const;
        DLL void Load ( JavaScript& aJavaScript );
        DLL void Unload ( JavaScript& aJavaScript );
        DLL Element* AddElement ( std::unique_ptr<Element> aElement );
        DLL std::unique_ptr<Element> RemoveElement ( const Element* aElement );
        DLL void TraverseDepthFirstPreOrder ( const std::function<void ( Element& ) >& aAction );
        DLL void TraverseDepthFirstPreOrder ( const std::function<void ( const Element& ) >& aAction ) const;
        DLL void TraverseDepthFirstPostOrder ( const std::function<void ( Element& ) >& aAction );
        DLL void TraverseDepthFirstPostOrder ( const std::function<void ( const Element& ) >& aAction ) const;
        DLL void TraverseDepthFirstPreOrder ( const std::function<void ( Element& ) >& aPreamble, const std::function<void ( Element& ) >& aPostamble );
        DLL void TraverseDepthFirstPreOrder ( const std::function<void ( const Element& ) >& aPreamble, const std::function<void ( const Element& ) >& aPostamble ) const;
    private:
        xmlDocPtr mDocument{};
        std::vector<std::unique_ptr<Element>> mChildren{};
    };
}
#endif
