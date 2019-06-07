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
#include "aeongui/Transform.h"
#include "aeongui/AABB.h"
namespace AeonGUI
{
    class Widget
    {
    public:
        DLL Widget ( const Transform& aTransform, const AABB& aAABB );
        DLL const Transform& GetLocalTransform() const;
        DLL void SetTransform ( const Transform& aTransform );
        DLL const Transform GetGlobalTransform() const;
        DLL const AABB& GetAABB() const;
        DLL void Draw ( void* aDrawingContext ) const; ///< This should be virtual, but keeping as regular in the meantime.

        DLL Widget* AddWidget ( std::unique_ptr<Widget> aWidget );
        DLL std::unique_ptr<Widget> RemoveWidget ( const Widget* aWidget );

        DLL void TraverseDepthFirstPreOrder ( const std::function<void ( Widget& ) >& aAction );
        DLL void TraverseDepthFirstPreOrder ( const std::function<void ( const Widget& ) >& aAction ) const;
        DLL void TraverseDepthFirstPostOrder ( const std::function<void ( Widget& ) >& aAction );
        DLL void TraverseDepthFirstPostOrder ( const std::function<void ( const Widget& ) >& aAction ) const;
    private:
        Transform mTransform{};
        AABB mAABB{};
        Widget* mParent{};
        mutable std::vector<std::unique_ptr<Widget>>::size_type mIterator{ 0 };
        std::vector<std::unique_ptr<Widget>> mChildren{};
    };
}
