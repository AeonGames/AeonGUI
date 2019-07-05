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
#ifndef AEONGUI_ELEMENT_H
#define AEONGUI_ELEMENT_H
#include <cstdint>
#include <vector>
#include <memory>
#include <functional>
#include "aeongui/Platform.h"
#include "aeongui/Transform.h"
#include "aeongui/AABB.h"

extern "C"
{
    typedef struct _xmlElement xmlElement;
    typedef xmlElement *xmlElementPtr;
}

namespace AeonGUI
{
    class Canvas;
    class Element
    {
    public:
        DLL Element ( xmlElementPtr aXmlElementPtr );
        DLL Element* AddElement ( std::unique_ptr<Element> aElement );
        DLL std::unique_ptr<Element> RemoveElement ( const Element* aElement );
        DLL void TraverseDepthFirstPreOrder ( const std::function<void ( Element& ) >& aAction );
        DLL void TraverseDepthFirstPreOrder ( const std::function<void ( const Element& ) >& aAction ) const;
        DLL void TraverseDepthFirstPostOrder ( const std::function<void ( Element& ) >& aAction );
        DLL void TraverseDepthFirstPostOrder ( const std::function<void ( const Element& ) >& aAction ) const;
        DLL const char* GetTagName() const;
        DLL bool HasAttr ( const char* aAttrName ) const;
        DLL const char* GetAttr ( const char* aAttrName ) const;
        DLL virtual void Draw ( Canvas& aCanvas ) const;
        DLL virtual ~Element();
    protected:
        xmlElementPtr mXmlElementPtr{};
    private:
        Element* mParent{};
        std::vector<std::unique_ptr<Element>> mChildren{};
        mutable std::vector<std::unique_ptr<Element>>::size_type mIterator{ 0 };
    };
}
#endif
