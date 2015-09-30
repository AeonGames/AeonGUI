#ifndef AEONGUI_SURFACE_H
#define AEONGUI_SURFACE_H
/******************************************************************************
Copyright 2015 Rodrigo Hernandez Cordoba

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
******************************************************************************/

#include <cstdint>
namespace AeonGUI
{
    /*! \brief Surface Interface. */
    class Surface
    {
    public:
        /// Virtual destructor.
        virtual ~Surface() {};
        ///@name Getters
        ///@{
        /// Get the surface width in pixels.
        virtual uint32_t Width() const = 0;
        /// Get the surface height in pixels.
        virtual uint32_t Height() const = 0;
        /** Maps the surface buffer into user memory.
         * Depending on the implementation this may be
         * a simple getter or an actual memory mapping to
         * from specialized hardware.
         * On thread safe applications, a mutex may be locked
         * by this function call.
         * @return A pointer to the mapped memory.
         * @note the mapped memory is assumed to be read/write enabled.
         * */
        virtual void* MapMemory() = 0;
        /** Unmaps the surface buffer from user memory
         * and commits any changes made to it.
         * */
        virtual void UnmapMemory() = 0;
        ///@}
        ///@name Setters
        ///@{
        /** Changes the surface size.
         * @param aWidth The new surface width, must be > 0.
         * @param aHeight The new surface height, must be > 0.
         * */
        virtual void ReSize ( uint32_t aWidth, uint32_t aHeight ) = 0;
        ///@}
    };
}
#endif

