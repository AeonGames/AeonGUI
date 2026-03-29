/*
Copyright (C) 2026 Rodrigo Jose Hernandez Cordoba

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
#ifndef AEONGUI_SVGANIMATIONELEMENT_H
#define AEONGUI_SVGANIMATIONELEMENT_H

#include "SVGElement.hpp"
#include "EventListener.hpp"
#include <vector>
#include <string>
#include <limits>

namespace AeonGUI
{
    class Canvas;
    namespace DOM
    {
        class SVGAnimationElement : public SVGElement, public EventListener
        {
        public:
            SVGAnimationElement ( const DOMString& aTagName, AttributeMap&& aAttributes, Node* aParent );
            ~SVGAnimationElement() override;
            void handleEvent ( Event& event ) override;
            DLL bool IsDrawEnabled() const override final;
            DLL void Update ( double aDocumentTime ) override;
            virtual void ApplyToCanvas ( Canvas& aCanvas ) const = 0;
            bool IsActive() const;
            const std::string& GetAttributeName() const
            {
                return mAttributeName;
            }
        protected:
            static double ParseDuration ( const std::string& aDuration );
            static std::vector<std::string> SplitValues ( const std::string& aValues );
            std::string mAttributeName;
            double mDuration{0.0};
            double mBeginTime{0.0};
            double mRepeatCount{1.0};
            bool mFreezeOnEnd{false};
            bool mIsActive{false};
            double mProgress{0.0};
            std::string mBeginEventType;
            double mLastDocumentTime{0.0};
        };
    }
}
#endif
