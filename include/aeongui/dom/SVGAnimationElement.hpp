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
        /** @brief Abstract base class for SMIL animation elements.
         *
         *  Provides timing, activation, event-based begin, and progress
         *  computation shared by all concrete SMIL animation elements.
         *  Inherits EventListener so that event-based begin values
         *  (e.g. "click", "mouseenter") can trigger the animation.
         *  @see https://www.w3.org/TR/SVG11/animate.html
         */
        class SVGAnimationElement : public SVGElement, public EventListener
        {
        public:
            /** @brief Construct an SVGAnimationElement.
             *  @param aTagName    Tag name of the element.
             *  @param aAttributes Parsed attribute map (moved in).
             *  @param aParent     Parent node in the DOM tree.
             */
            SVGAnimationElement ( const DOMString& aTagName, AttributeMap&& aAttributes, Node* aParent );
            /** @brief Destructor. */
            ~SVGAnimationElement() override;
            /** @brief Handle a DOM event (implements EventListener).
             *  @param event The event to handle.
             */
            void handleEvent ( Event& event ) override;
            /** @brief Animation elements are never drawn directly.
             *  @return Always false.
             */
            AEONGUI_DLL bool IsDrawEnabled() const override final;
            /** @copydoc Node::Update */
            AEONGUI_DLL void Update ( double aDocumentTime ) override;
            /** @brief Apply this animation's effect to the canvas.
             *  @param aCanvas The target canvas.
             */
            virtual void ApplyToCanvas ( Canvas& aCanvas ) const = 0;
            /** @brief Check whether the animation is currently active.
             *  @return true if the animation is within its active interval.
             */
            bool IsActive() const;
            /** @brief Get the target attribute name.
             *  @return Const reference to the attributeName string.
             */
            const std::string& GetAttributeName() const
            {
                return mAttributeName;
            }
        protected:
            /** @brief Parse a SMIL duration or clock value string.
             *  @param aDuration The duration string (e.g. "2s", "500ms").
             *  @return Duration in seconds, or 0 for event names.
             */
            static double ParseDuration ( const std::string& aDuration );
            /** @brief Split a semicolon-separated value list.
             *  @param aValues The values string.
             *  @return Vector of individual value strings.
             */
            static std::vector<std::string> SplitValues ( const std::string& aValues );
            std::string mAttributeName;  ///< Target attribute name from the attributeName attribute.
            double mDuration{0.0};       ///< Animation duration in seconds.
            double mBeginTime{0.0};      ///< Begin time in document seconds.
            double mRepeatCount{1.0};    ///< Number of times to repeat (indefinite = infinity).
            bool mFreezeOnEnd{false};    ///< Whether fill="freeze" is set.
            bool mIsActive{false};       ///< Current activation state.
            double mProgress{0.0};       ///< Normalised progress within one cycle [0,1].
            std::string mBeginEventType; ///< Event type for event-based begin, or empty.
            double mLastDocumentTime{0.0}; ///< Last document time seen by Update.
        };
    }
}
#endif
