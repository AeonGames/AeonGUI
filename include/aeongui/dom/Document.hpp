/*
Copyright (C) 2019,2020,2023,2025,2026 Rodrigo Jose Hernandez Cordoba

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
#ifndef AEONGUI_DOCUMENT_H
#define AEONGUI_DOCUMENT_H
#include <cstdint>
#include <vector>
#include <memory>
#include <functional>
#include <algorithm>
#include "aeongui/Platform.hpp"
#include "aeongui/Canvas.hpp"
#include "aeongui/StyleSheet.hpp"
#include "aeongui/dom/Node.hpp"
#include "aeongui/dom/DOMString.hpp"
#include "aeongui/dom/USVString.hpp"
#include "aeongui/dom/Element.hpp"
namespace AeonGUI
{
    namespace DOM
    {
        /** @brief Represents a DOM Document.
         *
         *  The Document is the root of the DOM tree. It owns the parsed
         *  SVG node hierarchy and an associated CSS stylesheet.
         */
        class Document : public Node
        {
        public:
            /** @brief Default constructor. Creates an empty document. */
            AEONGUI_DLL Document();
            /** @brief Load a document from a file.
             *  @param aFilename Path or URL of the SVG/XML file to load.
             */
            AEONGUI_DLL void Load ( const USVString& aFilename );
            /** @brief Destructor. Unloads the document. */
            AEONGUI_DLL ~Document();
            /** @brief Draw the document onto a canvas.
             *  @param aCanvas The rendering surface.
             */
            AEONGUI_DLL void Draw ( Canvas& aCanvas ) const;
            /** @brief Draw the document onto a canvas with a per-node callback.
             *
             *  The callback is invoked for each node before DrawStart,
             *  with the canvas already saved.  This is used to assign
             *  pick IDs before geometry elements render their paths.
             *  @param aCanvas  The rendering surface.
             *  @param aPreDraw Callback invoked for each node before DrawStart.
             */
            AEONGUI_DLL void Draw ( Canvas& aCanvas, const std::function<void ( const Node& ) >& aPreDraw ) const;
            /** @brief Advance animation time and update all animations.
             *  @param aDeltaTime Time elapsed since last update, in seconds.
             */
            AEONGUI_DLL void AdvanceTime ( double aDeltaTime );
            /** @brief Get the document URL.
             *  @return The URL from which the document was loaded.
             */
            AEONGUI_DLL const USVString& url() const;
            /**DOM Properties and Methods @{*/
            /** @brief Get the node type (always DOCUMENT_NODE).
             *  @return NodeType::DOCUMENT_NODE. */
            AEONGUI_DLL NodeType nodeType() const final;
            /** @brief Find an element by its ID attribute.
             *  @param aElementId The ID to search for.
             *  @return Pointer to the matching Element, or nullptr.
             *  @see https://dom.spec.whatwg.org/#dom-document-getelementbyid
             */
            AEONGUI_DLL Element* getElementById ( const DOMString& aElementId ) const;
            /** @brief Get the document-level CSS stylesheet.
             *  @return Raw pointer to the stylesheet, or nullptr.
             */
            AEONGUI_DLL css_stylesheet* GetStyleSheet() const;
            /** @brief Mark the entire document as needing a full redraw.
             *
             *  Used for global changes like viewport resize or document load.
             */
            void MarkDirty()
            {
                mFullDirty = true;
            }
            /** @brief Mark a specific element as needing redraw.
             *
             *  Called automatically when element styles or attributes change.
             *  The Window uses element-level dirty tracking to compute
             *  the minimal dirty rectangle for partial redraws.
             *  @param aElement The element whose appearance changed.
             */
            void MarkElementDirty ( Element* aElement )
            {
                mDirtyElements.push_back ( aElement );
            }
            /** @brief Check whether the document needs redrawing.
             *  @return true if the document has been modified since the last draw.
             */
            bool IsDirty() const
            {
                return mFullDirty || !mDirtyElements.empty();
            }
            /** @brief Check whether a full (non-partial) redraw is needed.
             *  @return true if the entire scene must be redrawn.
             */
            bool IsFullDirty() const
            {
                return mFullDirty;
            }
            /** @brief Get the list of elements dirtied since the last draw.
             *  @return Reference to the dirty elements vector.
             */
            const std::vector<Element*>& GetDirtyElements() const
            {
                return mDirtyElements;
            }
            /** @brief Clear the dirty flag after a redraw. */
            void ClearDirty()
            {
                mFullDirty = false;
                mDirtyElements.clear();
            }
            /**@}*/
        private:
            void Load ();
            void Unload ();
            //Element* mDocumentElement{};
            StyleSheetPtr mStyleSheet{};
            USVString mUrl{};
            double mDocumentTime{0.0};
            bool mFullDirty{true};
            std::vector<Element*> mDirtyElements{};
        };
    }
}
#endif
