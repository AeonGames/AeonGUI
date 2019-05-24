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
#include "aeongui/Matrix3x3.h"
#include "aeongui/Transform.h"
namespace AeonGUI
{
    Matrix3x3::Matrix3x3()
    {
        mMatrix3x3[0] = mMatrix3x3[4] = mMatrix3x3[8] = 1.0f;
        mMatrix3x3[1] = mMatrix3x3[2] = mMatrix3x3[3] = mMatrix3x3[5] = mMatrix3x3[6] = mMatrix3x3[7] = 0.0f;
    }
    Matrix3x3::Matrix3x3 ( const Transform& aTransform ) : Matrix3x3{aTransform.GetMatrix() }
    {}
}
