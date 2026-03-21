/*
Copyright (C) 2019,2025,2026 Rodrigo Jose Hernandez Cordoba

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
#ifndef AEONGUI_VECTOR2_H
#define AEONGUI_VECTOR2_H
#include <cstddef>
#include <cstdint>
#include "aeongui/Platform.hpp"
namespace AeonGUI
{
    class Matrix2x3;
    /** @brief 2D vector of doubles.
     *
     *  Provides basic 2D vector arithmetic including addition, subtraction,
     *  scalar and component-wise multiplication, division, and
     *  transformation by a Matrix2x3.
     */
    class Vector2
    {
    public:
        /** @brief Default constructor. Initializes to (0, 0). */
        DLL Vector2();
        /** @brief Construct from X and Y components.
         *  @param aX The X component.
         *  @param aY The Y component.
         */
        DLL Vector2 ( double aX, double aY );
        /** @brief Get the X component.
         *  @return The X value.
         */
        DLL double GetX() const;
        /** @brief Get the Y component.
         *  @return The Y value.
         */
        DLL double GetY() const;
        /** @brief Set the X component.
         *  @param aX The new X value.
         */
        DLL void SetX ( double aX );
        /** @brief Set the Y component.
         *  @param aY The new Y value.
         */
        DLL void SetY ( double aY );
        /** @brief Get the length (magnitude) of the vector.
         *  @return The length.
         */
        DLL double Length() const;
        /** @brief Access a component by index (const).
         *  @param aIndex 0 for X, 1 for Y.
         *  @return Const reference to the component.
         */
        DLL const double& operator[] ( std::size_t aIndex ) const;
        /** @brief Access a component by index.
         *  @param aIndex 0 for X, 1 for Y.
         *  @return Reference to the component.
         */
        DLL double& operator[] ( std::size_t aIndex );
        /** @brief Add another vector in-place.
         *  @param aRight The vector to add.
         *  @return Reference to this.
         */
        DLL Vector2& operator+= ( const Vector2& aRight );
        /** @brief Subtract another vector in-place.
         *  @param aRight The vector to subtract.
         *  @return Reference to this.
         */
        DLL Vector2& operator-= ( const Vector2& aRight );
        /** @brief Transform by a Matrix2x3 in-place.
         *  @param aRight The matrix to multiply by.
         *  @return Reference to this.
         */
        DLL Vector2& operator*= ( const Matrix2x3& aRight );
        /** @brief Component-wise multiply in-place.
         *  @param aRight The vector to multiply by.
         *  @return Reference to this.
         */
        DLL Vector2& operator*= ( const Vector2& aRight );
        /** @brief Scalar multiply in-place.
         *  @param aRight The scalar to multiply by.
         *  @return Reference to this.
         */
        DLL Vector2& operator*= ( double aRight );
        /** @brief Scalar divide in-place.
         *  @param aRight The scalar to divide by.
         *  @return Reference to this.
         */
        DLL Vector2& operator/= ( double aRight );
    private:
        double mVector2[2];
    };
    /** @brief Add two vectors. */
    DLL Vector2 operator+ ( const Vector2& aLeft, const Vector2& aRight );
    /** @brief Subtract two vectors. */
    DLL Vector2 operator- ( const Vector2& aLeft, const Vector2& aRight );
    /** @brief Transform a vector by a Matrix2x3. */
    DLL Vector2 operator* ( const Vector2& aLeft, const Matrix2x3& aRight );
    /** @brief Component-wise multiply two vectors. */
    DLL Vector2 operator* ( const Vector2& aLeft, const Vector2& aRight );
    /** @brief Divide a vector by a scalar. */
    DLL Vector2 operator/ ( const Vector2& aLeft, double aRight );
    /** @brief Multiply a vector by a scalar. */
    DLL Vector2 operator* ( const Vector2& aLeft, double aRight );
    /** @brief Component-wise absolute value. */
    DLL Vector2 Abs ( const Vector2& aVector2 );
    /** @brief Dot product of two vectors. */
    DLL double Dot ( const Vector2& aLeft, const Vector2& aRight );
    /** @brief Euclidean distance between two points. */
    DLL double Distance ( const Vector2& aLeft, const Vector2& aRight );
}
#endif
