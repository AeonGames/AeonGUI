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
#include <cstdint>
#include <vector>
#include <memory>
#include <functional>
#include "aeongui/Platform.h"
#include "aeongui/Rect.h"
namespace AeonGUI
{
    class Widget
    {
    public:
        DLL Widget ( const Rect& aRect );
        DLL const Rect& GetRect() const;

        DLL void TraverseDepthFirstPreOrder ( const std::function<void ( Widget& ) >& aAction );
        DLL void TraverseDepthFirstPreOrder ( const std::function<void ( const Widget& ) >& aAction ) const;
        DLL void TraverseDepthFirstPostOrder ( const std::function<void ( Widget& ) >& aAction );
        DLL void TraverseDepthFirstPostOrder ( const std::function<void ( const Widget& ) >& aAction ) const;
    private:
        Rect mRect{};
        Widget* mParent{};
        mutable std::vector<std::unique_ptr<Widget>>::size_type mIterator{ 0 };
        std::vector<std::unique_ptr<Widget>> mChildren{};
    };
}
