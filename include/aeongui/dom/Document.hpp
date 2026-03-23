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
            DLL Document();
            /** @brief Load a document from a file.
             *  @param aFilename Path or URL of the SVG/XML file to load.
             */
            DLL void Load ( const USVString& aFilename );
            /** @brief Destructor. Unloads the document. */
            DLL ~Document();
            /** @brief Draw the document onto a canvas.
             *  @param aCanvas The rendering surface.
             */
            DLL void Draw ( Canvas& aCanvas ) const;
            /** @brief Get the document URL.
             *  @return The URL from which the document was loaded.
             */
            DLL const USVString& url() const;
            /**DOM Properties and Methods @{*/
            /** @brief Get the node type (always DOCUMENT_NODE).
             *  @return NodeType::DOCUMENT_NODE. */
            DLL NodeType nodeType() const final;
            /** @brief Find an element by its ID attribute.
             *  @param aElementId The ID to search for.
             *  @return Pointer to the matching Element, or nullptr.
             *  @see https://dom.spec.whatwg.org/#dom-document-getelementbyid
             */
            DLL Element* getElementById ( const DOMString& aElementId ) const;
            /**@}*/
        private:
            void Load ();
            void Unload ();
            //Element* mDocumentElement{};
            StyleSheetPtr mStyleSheet{};
            USVString mUrl{};
        };
    }
}
#endif
