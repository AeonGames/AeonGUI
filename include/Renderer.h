#ifndef AEONGUI_RENDERER_H
#define AEONGUI_RENDERER_H
/******************************************************************************
Copyright 2010-2013,2015 Rodrigo Hernandez Cordoba, AeonGames

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
#include "Platform.h"

namespace AeonGUI
{
    /*! \brief Renderer interface class.
        Implement this interface to support different drawing APIs.*/
    class Renderer
    {
    public:
        /// Virtual destructor so all derived destructors are called accordingly.
        virtual ~Renderer() {};
        /*! \brief Initialization Function.
            \return true on success, false on failure.*/
        virtual bool Initialize() = 0;
        /*! \brief Finalization Function. */
        virtual void Finalize() = 0;
        /*! \brief Used to do any post-rendering cleanup. */
        virtual void Render() = 0;
        ///@name Rendering Surface Interface
        ///@{
        ///@name Getters
        ///@{
        /// Get the surface width in pixels.
        virtual uint32_t SurfaceWidth() const = 0;
        /// Get the surface height in pixels.
        virtual uint32_t SurfaceHeight() const = 0;
        /** Maps the surface buffer into user memory.
        * Depending on the implementation this may be
        * a simple getter or an actual memory mapping to
        * from specialized hardware.
        * On thread safe applications, a mutex may be locked
        * by this function call.
        * @return A pointer to the mapped memory.
        * @note the mapped memory is assumed to be read/write enabled.
        * */
        virtual uint8_t* MapMemory() = 0;
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
        ///@}
    };
}
#endif
