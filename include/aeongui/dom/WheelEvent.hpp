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
#ifndef AEONGUI_DOM_WHEELEVENT_H
#define AEONGUI_DOM_WHEELEVENT_H
#include "aeongui/dom/MouseEvent.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        /** @brief Initialization dictionary for WheelEvent construction.
         *  @see https://www.w3.org/TR/uievents/#idl-wheeleventinit
         */
        struct WheelEventInit : MouseEventInit
        {
            double deltaX{0.0};          ///< Scroll amount for the X axis.
            double deltaY{0.0};          ///< Scroll amount for the Y axis.
            double deltaZ{0.0};          ///< Scroll amount for the Z axis.
            unsigned long deltaMode{0};  ///< Unit indicator for delta values.
        };

        /** @brief Wheel event providing scroll-related information.
         *
         *  Indicates that the user has rotated a wheel, scroll hinge, or
         *  similar device on a pointing device.
         *  @see https://www.w3.org/TR/uievents/#idl-wheelevent
         */
        class WheelEvent : public MouseEvent
        {
        public:
            static constexpr unsigned long DOM_DELTA_PIXEL = 0x00; ///< Delta values are in pixels.
            static constexpr unsigned long DOM_DELTA_LINE  = 0x01; ///< Delta values are in lines.
            static constexpr unsigned long DOM_DELTA_PAGE  = 0x02; ///< Delta values are in pages.

            /** @brief Construct a WheelEvent.
             *  @param type The event type name (typically "wheel").
             *  @param eventInitDict Optional initialization dictionary.
             */
            WheelEvent ( const DOMString& type, const WheelEventInit& eventInitDict = {} );

            /** @brief Get the X-axis scroll amount.
             *  @return Delta X value. */
            double deltaX() const;
            /** @brief Get the Y-axis scroll amount.
             *  @return Delta Y value. */
            double deltaY() const;
            /** @brief Get the Z-axis scroll amount.
             *  @return Delta Z value. */
            double deltaZ() const;
            /** @brief Get the unit indicator for delta values.
             *  @return One of DOM_DELTA_PIXEL, DOM_DELTA_LINE, DOM_DELTA_PAGE. */
            unsigned long deltaMode() const;

        private:
            double m_deltaX{0.0};
            double m_deltaY{0.0};
            double m_deltaZ{0.0};
            unsigned long m_deltaMode{0};
        };
    }
}
#endif
