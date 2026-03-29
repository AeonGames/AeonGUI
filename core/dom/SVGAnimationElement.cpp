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
#include <cmath>
#include <limits>
#include <sstream>
#include "aeongui/dom/SVGAnimationElement.hpp"
#include "aeongui/dom/Event.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        SVGAnimationElement::SVGAnimationElement ( const DOMString& aTagName, AttributeMap&& aAttributes, Node* aParent )
            : SVGElement { aTagName, std::move ( aAttributes ), aParent }
        {
            if ( mAttributes.find ( "attributeName" ) != mAttributes.end() )
            {
                mAttributeName = mAttributes.at ( "attributeName" );
            }
            if ( mAttributes.find ( "dur" ) != mAttributes.end() )
            {
                mDuration = ParseDuration ( mAttributes.at ( "dur" ) );
            }
            if ( mAttributes.find ( "begin" ) != mAttributes.end() )
            {
                const auto& beginStr = mAttributes.at ( "begin" );
                if ( !beginStr.empty() && !std::isdigit ( static_cast<unsigned char> ( beginStr[0] ) ) &&
                     beginStr[0] != '-' && beginStr[0] != '+' && beginStr[0] != '.' &&
                     beginStr != "indefinite" )
                {
                    // Event-based begin (e.g. "click", "mouseover")
                    mBeginEventType = beginStr;
                    mBeginTime = std::numeric_limits<double>::infinity();
                    if ( aParent )
                    {
                        static_cast<EventTarget*> ( aParent )->addEventListener ( mBeginEventType, this );
                    }
                }
                else
                {
                    mBeginTime = ParseDuration ( beginStr );
                }
            }
            if ( mAttributes.find ( "repeatCount" ) != mAttributes.end() )
            {
                const auto& rc = mAttributes.at ( "repeatCount" );
                if ( rc == "indefinite" )
                {
                    mRepeatCount = -1.0;
                }
                else
                {
                    mRepeatCount = std::stod ( rc );
                }
            }
            if ( mAttributes.find ( "fill" ) != mAttributes.end() )
            {
                mFreezeOnEnd = ( mAttributes.at ( "fill" ) == "freeze" );
            }
        }

        SVGAnimationElement::~SVGAnimationElement()
        {
            if ( !mBeginEventType.empty() && parentNode() )
            {
                static_cast<EventTarget*> ( parentNode() )->removeEventListener ( mBeginEventType, this );
            }
        }

        bool SVGAnimationElement::IsDrawEnabled() const
        {
            return false;
        }

        bool SVGAnimationElement::IsActive() const
        {
            return mIsActive;
        }

        void SVGAnimationElement::handleEvent ( Event& event )
        {
            if ( event.type() == mBeginEventType )
            {
                mBeginTime = mLastDocumentTime;
            }
        }

        void SVGAnimationElement::Update ( double aDocumentTime )
        {
            mLastDocumentTime = aDocumentTime;
            if ( mDuration <= 0.0 )
            {
                mIsActive = false;
                return;
            }
            double elapsed = aDocumentTime - mBeginTime;
            if ( elapsed < 0.0 )
            {
                mIsActive = false;
                return;
            }
            double totalDuration = ( mRepeatCount < 0 ) ?
                                   std::numeric_limits<double>::infinity() :
                                   mDuration * mRepeatCount;
            if ( elapsed >= totalDuration )
            {
                mIsActive = mFreezeOnEnd;
                mProgress = 1.0;
                return;
            }
            mIsActive = true;
            mProgress = std::fmod ( elapsed, mDuration ) / mDuration;
        }

        double SVGAnimationElement::ParseDuration ( const std::string& aDuration )
        {
            if ( aDuration.empty() || aDuration == "indefinite" )
            {
                return 0.0;
            }
            // Event-based values (e.g. "click") are not time durations
            if ( !aDuration.empty() && !std::isdigit ( static_cast<unsigned char> ( aDuration[0] ) ) &&
                 aDuration[0] != '-' && aDuration[0] != '+' && aDuration[0] != '.' )
            {
                return 0.0;
            }
            double value = 0.0;
            size_t pos = 0;
            value = std::stod ( aDuration, &pos );
            std::string unit = aDuration.substr ( pos );
            if ( unit == "ms" )
            {
                return value / 1000.0;
            }
            if ( unit == "min" )
            {
                return value * 60.0;
            }
            if ( unit == "h" )
            {
                return value * 3600.0;
            }
            // Default: seconds (with or without "s" suffix)
            return value;
        }

        std::vector<std::string> SVGAnimationElement::SplitValues ( const std::string& aValues )
        {
            std::vector<std::string> result;
            std::istringstream stream ( aValues );
            std::string token;
            while ( std::getline ( stream, token, ';' ) )
            {
                // Trim whitespace
                size_t start = token.find_first_not_of ( " \t\n\r" );
                size_t end = token.find_last_not_of ( " \t\n\r" );
                if ( start != std::string::npos )
                {
                    result.push_back ( token.substr ( start, end - start + 1 ) );
                }
            }
            return result;
        }
    }
}
