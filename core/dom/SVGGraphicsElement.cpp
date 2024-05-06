/*
Copyright (C) 2020,2024 Rodrigo Jose Hernandez Cordoba

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
#include "SVGGraphicsElement.h"
#include "CSSSelectHandler.h"
#include <libcss/libcss.h>

namespace AeonGUI
{
    namespace DOM
    {
        SVGGraphicsElement::SVGGraphicsElement ( const std::string& aTagName, const AttributeMap& aAttributes ) : SVGElement { aTagName, aAttributes } {}
        SVGGraphicsElement::~SVGGraphicsElement() = default;

        void SVGGraphicsElement::DrawStart ( Canvas& aCanvas ) const
        {
            css_select_results* results{ GetComputedStyles() };
            if ( results && results->styles[CSS_PSEUDO_ELEMENT_NONE] )
            {
                css_matrix transform{};
                if ( css_computed_transform ( results->styles[CSS_PSEUDO_ELEMENT_NONE], &transform ) != CSS_TRANSFORM_NONE )
                {
                    Matrix2x3 matrix
                    {
                        FIXTOFLT ( transform.m[0] ), FIXTOFLT ( transform.m[1] ),
                        FIXTOFLT ( transform.m[2] ), FIXTOFLT ( transform.m[3] ),
                        FIXTOFLT ( transform.m[4] ), FIXTOFLT ( transform.m[5] )
                    };
                    aCanvas.SetTransform ( matrix );
                }
            }
        }
    }
}
