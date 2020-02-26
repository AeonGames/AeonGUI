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
#ifndef AEONGUI_ELEMENT_H
#define AEONGUI_ELEMENT_H
#include <cstdint>
#include <vector>
#include <memory>
#include <functional>
#include <variant>
#include "aeongui/Platform.h"
#include "aeongui/Transform.h"
#include "aeongui/AABB.h"
#include "aeongui/AttributeMap.h"

extern "C"
{
    typedef struct _xmlElement xmlElement;
    typedef xmlElement *xmlElementPtr;
}

namespace AeonGUI
{
    class Canvas;
    class JavaScript;
    class Document;
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
        DLL void TraverseDepthFirstPreOrder ( const std::function<void ( Element& ) >& aPreamble, const std::function<void ( Element& ) >& aPostamble );
        DLL void TraverseDepthFirstPreOrder ( const std::function<void ( const Element& ) >& aPreamble, const std::function<void ( const Element& ) >& aPostamble ) const;
        DLL void TraverseDepthFirstPreOrder ( const std::function<void ( Element& ) >& aPreamble, const std::function<void ( Element& ) >& aPostamble, const std::function<bool ( Element& ) >& aUnaryPredicate );
        DLL void TraverseDepthFirstPreOrder ( const std::function<void ( const Element& ) >& aPreamble, const std::function<void ( const Element& ) >& aPostamble, const std::function<bool ( const Element& ) >& aUnaryPredicate ) const;

        DLL const char* GetTagName() const;
        DLL bool HasAttr ( const char* aAttrName ) const;
        DLL const char* GetAttr ( const char* aAttrName ) const;
        DLL double GetAttrAsDouble ( const char* aAttrName, double aDefault = 0 ) const;
        DLL const char* GetContent () const;
        DLL AttributeType GetAttribute ( const char* attrName, const AttributeType& aDefault = {} ) const;
        DLL AttributeType GetInheritedAttribute ( const char* attrName, const AttributeType& aDefault = {} ) const;
        DLL virtual void DrawStart ( Canvas& aCanvas ) const;
        DLL virtual void DrawFinish ( Canvas& aCanvas ) const;
        DLL virtual void Load ( JavaScript& aJavaScript );
        DLL virtual void Unload ( JavaScript& aJavaScript );
        /** Returns whether this node and all descendants should be skipped
         *  in a drawing operation.
         *  @return true by default override to disable drawing.
        */
        DLL virtual bool IsDrawEnabled() const;
        DLL virtual ~Element();
    protected:
        xmlElementPtr mXmlElementPtr{};
        AttributeMap mAttributeMap{};
    private:
        Element* mParent{};
        std::vector<std::unique_ptr<Element>> mChildren{};
        mutable std::vector<std::unique_ptr<Element>>::size_type mIterator{ 0 };
    };
}
#endif
