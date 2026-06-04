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

/** @file
 *  xmlcxx — compile an XHTML/SVG document into C++ source that builds the DOM
 *  tree imperatively (no runtime XML parse) and embeds its `type="text/c++"`
 *  scripts and inline `onEVENT` handlers. The generated class subclasses
 *  AeonGUI::CompiledDocument.
 *
 *  Usage:
 *      xmlcxx <input> -o <out.cpp> [--header <out.hpp>]
 *             [--class <Name>] [--namespace <ns>]
 */

#include <libxml/parser.h>
#include <libxml/tree.h>

#include <cctype>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace
{
    struct Options
    {
        std::string input;
        std::string output;
        std::string header;
        std::string className;
        std::string nameSpace;
    };

    /** @brief Escape a string for use inside a C++ "..." string literal. */
    std::string EscapeCxx ( const std::string& aText )
    {
        std::string out;
        out.reserve ( aText.size() + 8 );
        for ( char c : aText )
        {
            switch ( c )
            {
            case '\\':
                out += "\\\\";
                break;
            case '"':
                out += "\\\"";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                out += c;
                break;
            }
        }
        return out;
    }

    /** @brief Derive a PascalCase identifier from a file path stem. */
    std::string PascalCaseStem ( const std::string& aPath )
    {
        // strip directory
        size_t slash = aPath.find_last_of ( "/\\" );
        std::string name = ( slash == std::string::npos ) ? aPath : aPath.substr ( slash + 1 );
        // strip extension
        size_t dot = name.find_last_of ( '.' );
        if ( dot != std::string::npos )
        {
            name = name.substr ( 0, dot );
        }
        std::string out;
        bool upper = true;
        for ( char c : name )
        {
            if ( c == '_' || c == '-' || c == ' ' || c == '.' )
            {
                upper = true;
                continue;
            }
            if ( out.empty() && ( c >= '0' && c <= '9' ) )
            {
                // identifiers cannot start with a digit
                out += '_';
            }
            out += upper ? static_cast<char> ( std::toupper ( static_cast<unsigned char> ( c ) ) ) : c;
            upper = false;
        }
        if ( out.empty() )
        {
            out = "Document";
        }
        return out;
    }

    std::string ToUpperGuard ( const std::string& aText )
    {
        std::string out;
        for ( char c : aText )
        {
            if ( ( c >= 'A' && c <= 'Z' ) || ( c >= '0' && c <= '9' ) )
            {
                out += c;
            }
            else if ( c >= 'a' && c <= 'z' )
            {
                out += static_cast<char> ( std::toupper ( static_cast<unsigned char> ( c ) ) );
            }
            else
            {
                out += '_';
            }
        }
        return out;
    }

    bool IsBlankText ( const xmlChar* aContent )
    {
        if ( aContent == nullptr )
        {
            return true;
        }
        for ( const xmlChar * p = aContent; *p; ++p )
        {
            if ( *p != ' ' && *p != '\t' && *p != '\n' && *p != '\r' )
            {
                return false;
            }
        }
        return true;
    }

    /** @brief True if @p aNode is a `<script type="text/c++">` element. */
    bool IsCxxScript ( xmlNodePtr aNode )
    {
        if ( aNode->type != XML_ELEMENT_NODE )
        {
            return false;
        }
        if ( xmlStrcmp ( aNode->name, reinterpret_cast<const xmlChar * > ( "script" ) ) != 0 )
        {
            return false;
        }
        xmlChar* type = xmlGetProp ( aNode, reinterpret_cast<const xmlChar*> ( "type" ) );
        bool match = ( type != nullptr ) &&
                     ( xmlStrcmp ( type, reinterpret_cast<const xmlChar*> ( "text/c++" ) ) == 0 );
        if ( type != nullptr )
        {
            xmlFree ( type );
        }
        return match;
    }

    /** @brief Gather the concatenated text/CDATA content of an element. */
    std::string GatherText ( xmlNodePtr aNode )
    {
        std::string out;
        for ( xmlNodePtr child = aNode->children; child; child = child->next )
        {
            if ( ( child->type == XML_TEXT_NODE || child->type == XML_CDATA_SECTION_NODE ) &&
                 child->content != nullptr )
            {
                out += reinterpret_cast<const char*> ( child->content );
            }
        }
        return out;
    }

    struct Generator
    {
        std::ostringstream build;       ///< body of BuildDOM
        std::ostringstream scripts;     ///< verbatim text/c++ script bodies
        std::vector<std::string> loadFns; ///< functions bound via onload="Fn"
        int varCounter = 0;

        std::string NewVar()
        {
            return "n" + std::to_string ( varCounter++ );
        }

        /** @brief Trim leading/trailing ASCII whitespace. */
        static std::string Trim ( const std::string& aText )
        {
            size_t b = aText.find_first_not_of ( " \t\r\n" );
            if ( b == std::string::npos )
            {
                return std::string{};
            }
            size_t e = aText.find_last_not_of ( " \t\r\n" );
            return aText.substr ( b, e - b + 1 );
        }

        /** @brief Emit the AttributeMap initializer list, skipping onEVENT
         *         attributes (which bind to script-defined functions).
         *  @param aNode      The element.
         *  @param aHandlers  Output: (event-name, function-name) pairs for
         *                    non-load events. The @c onload event is collected
         *                    into @ref loadFns instead.
         */
        std::string EmitAttributeMap ( xmlNodePtr aNode,
                                       std::vector<std::pair<std::string, std::string>>& aHandlers )
        {
            std::string out = "AttributeMap{";
            bool first = true;
            for ( xmlAttrPtr attr = aNode->properties; attr; attr = attr->next )
            {
                std::string name = reinterpret_cast<const char*> ( attr->name );
                xmlChar* value = xmlGetProp ( aNode, attr->name );
                std::string valStr = ( value != nullptr ) ? reinterpret_cast<const char*> ( value ) : "";
                if ( value != nullptr )
                {
                    xmlFree ( value );
                }
                // onEVENT="Fn" -> bind to the script-defined function Fn,
                // stripped from the DOM attribute map.
                if ( name.size() > 2 && name[0] == 'o' && name[1] == 'n' )
                {
                    std::string event = name.substr ( 2 );
                    std::string fn = Trim ( valStr );
                    if ( !fn.empty() )
                    {
                        if ( event == "load" )
                        {
                            loadFns.push_back ( fn );
                        }
                        else
                        {
                            aHandlers.emplace_back ( event, fn );
                        }
                    }
                    continue;
                }
                if ( !first )
                {
                    out += ", ";
                }
                first = false;
                out += "{\"" + EscapeCxx ( name ) + "\", \"" + EscapeCxx ( valStr ) + "\"}";
            }
            out += "}";
            return out;
        }

        /** @brief Recursively emit BuildDOM statements for @p aNode.
         *  @param aNode      Current XML node.
         *  @param aParentVar C++ variable name of the parent Node, or empty for
         *                    the document root (attaches to @c document).
         */
        void Walk ( xmlNodePtr aNode, const std::string& aParentVar )
        {
            for ( xmlNodePtr node = aNode; node; node = node->next )
            {
                if ( node->type == XML_ELEMENT_NODE )
                {
                    if ( IsCxxScript ( node ) )
                    {
                        // Copy the script body verbatim to file scope; no DOM
                        // node is emitted.
                        scripts << GatherText ( node ) << "\n";
                        continue;
                    }
                    const char* nsUri = ( node->ns && node->ns->href )
                                        ? reinterpret_cast<const char*> ( node->ns->href )
                                        : "";
                    std::vector<std::pair<std::string, std::string>> nodeHandlers;
                    std::string attrs = EmitAttributeMap ( node, nodeHandlers );
                    std::string var = NewVar();
                    std::string tag = reinterpret_cast<const char*> ( node->name );

                    build << "            AeonGUI::DOM::Node* " << var << " = ";
                    if ( aParentVar.empty() )
                    {
                        build << "document.AddNode ( AeonGUI::Construct ( \""
                              << EscapeCxx ( nsUri ) << "\", \"" << EscapeCxx ( tag )
                              << "\", " << attrs << ", &document ) );\n";
                    }
                    else
                    {
                        build << aParentVar << "->AddNode ( AeonGUI::Construct ( \""
                              << EscapeCxx ( nsUri ) << "\", \"" << EscapeCxx ( tag )
                              << "\", " << attrs << ", " << aParentVar << " ) );\n";
                    }

                    // Wire inline event handlers: each calls the named function
                    // Fn ( *this, window, document, event ), where Fn is defined
                    // verbatim in a script block.
                    for ( const auto& h : nodeHandlers )
                    {
                        build << "            {\n";
                        build << "                auto listener = std::make_unique<xmlcxx_Adapter> ( "
                              << "[this] ( AeonGUI::DOM::Event & event )\n";
                        build << "                {\n";
                        build << "                    " << h.second
                              << " ( *this, *this->window(), *this->document(), event );\n";
                        build << "                } );\n";
                        build << "                " << var << "->addEventListener ( \""
                              << EscapeCxx ( h.first ) << "\", listener.get() );\n";
                        build << "                mListeners.push_back ( std::move ( listener ) );\n";
                        build << "            }\n";
                    }

                    Walk ( node->children, var );
                }
                else if ( ( node->type == XML_TEXT_NODE || node->type == XML_CDATA_SECTION_NODE ) &&
                          !IsBlankText ( node->content ) && !aParentVar.empty() )
                {
                    std::string text = reinterpret_cast<const char*> ( node->content );
                    build << "            " << aParentVar
                          << "->AddNode ( std::make_unique<AeonGUI::DOM::Text> ( \""
                          << EscapeCxx ( text ) << "\", " << aParentVar << " ) );\n";
                }
            }
        }
    };

    int Usage ( const char* aProg )
    {
        std::cerr << "Usage: " << aProg
                  << " <input> -o <out.cpp> [--header <out.hpp>]"
           " [--class <Name>] [--namespace <ns>]\n";
        return 1;
    }
}

int main ( int argc, char** argv )
{
    Options options;
    for ( int i = 1; i < argc; ++i )
    {
        std::string arg = argv[i];
        if ( arg == "-o" || arg == "--output" )
        {
            if ( ++i >= argc )
            {
                return Usage ( argv[0] );
            }
            options.output = argv[i];
        }
        else if ( arg == "--header" || arg == "-h" )
        {
            if ( ++i >= argc )
            {
                return Usage ( argv[0] );
            }
            options.header = argv[i];
        }
        else if ( arg == "--class" )
        {
            if ( ++i >= argc )
            {
                return Usage ( argv[0] );
            }
            options.className = argv[i];
        }
        else if ( arg == "--namespace" )
        {
            if ( ++i >= argc )
            {
                return Usage ( argv[0] );
            }
            options.nameSpace = argv[i];
        }
        else if ( !arg.empty() && arg[0] == '-' )
        {
            std::cerr << "Unknown option: " << arg << "\n";
            return Usage ( argv[0] );
        }
        else
        {
            options.input = arg;
        }
    }

    if ( options.input.empty() || options.output.empty() )
    {
        return Usage ( argv[0] );
    }
    if ( options.className.empty() )
    {
        options.className = PascalCaseStem ( options.input ) + "Document";
    }

    xmlDocPtr doc = xmlReadFile ( options.input.c_str(), nullptr,
                                  XML_PARSE_NOENT | XML_PARSE_NOCDATA | XML_PARSE_NONET );
    if ( doc == nullptr )
    {
        std::cerr << "xmlcxx: failed to parse '" << options.input << "'\n";
        return 1;
    }
    xmlNodePtr root = xmlDocGetRootElement ( doc );
    if ( root == nullptr )
    {
        std::cerr << "xmlcxx: empty document '" << options.input << "'\n";
        xmlFreeDoc ( doc );
        return 1;
    }

    Generator gen;
    gen.Walk ( root, std::string{} );
    xmlFreeDoc ( doc );

    // Determine the header file name (for the #include in the cpp).
    std::string headerPath = options.header;
    std::string headerInclude;
    if ( !headerPath.empty() )
    {
        size_t slash = headerPath.find_last_of ( "/\\" );
        headerInclude = ( slash == std::string::npos ) ? headerPath : headerPath.substr ( slash + 1 );
    }

    // --- Emit header (optional) ---
    if ( !headerPath.empty() )
    {
        std::ofstream hdr ( headerPath, std::ios::binary );
        if ( !hdr )
        {
            std::cerr << "xmlcxx: cannot open header '" << headerPath << "'\n";
            return 1;
        }
        std::string guard = "XMLCXX_" + ToUpperGuard ( options.className ) + "_H";
        hdr << "/* Generated by xmlcxx from " << options.input << ". DO NOT EDIT. */\n";
        hdr << "#ifndef " << guard << "\n#define " << guard << "\n";
        hdr << "#include <memory>\n#include <vector>\n";
        hdr << "#include \"aeongui/CompiledDocument.hpp\"\n";
        hdr << "#include \"aeongui/dom/EventListener.hpp\"\n\n";
        if ( !options.nameSpace.empty() )
        {
            hdr << "namespace " << options.nameSpace << "\n{\n";
        }
        hdr << "    /** @brief Compiled document generated from "
            << options.input << ". */\n";
        hdr << "    class " << options.className << " : public AeonGUI::CompiledDocument\n";
        hdr << "    {\n    public:\n";
        hdr << "        " << options.className << "();\n";
        hdr << "        ~" << options.className << "() override;\n";
        hdr << "    protected:\n";
        hdr << "        void BuildDOM ( AeonGUI::DOM::Document& document ) override;\n";
        hdr << "        void OnLoad ( AeonGUI::DOM::Document& document, AeonGUI::DOM::Window& window ) override;\n";
        hdr << "    private:\n";
        hdr << "        std::vector<std::unique_ptr<AeonGUI::DOM::EventListener>> mListeners;\n";
        hdr << "    };\n";
        if ( !options.nameSpace.empty() )
        {
            hdr << "}\n";
        }
        hdr << "#endif\n";
    }

    // --- Emit cpp ---
    std::ofstream cpp ( options.output, std::ios::binary );
    if ( !cpp )
    {
        std::cerr << "xmlcxx: cannot open output '" << options.output << "'\n";
        return 1;
    }
    cpp << "/* Generated by xmlcxx from " << options.input << ". DO NOT EDIT. */\n";
    if ( !headerInclude.empty() )
    {
        cpp << "#include \"" << headerInclude << "\"\n";
    }
    cpp << "#include <functional>\n";
    cpp << "#include <memory>\n";
    cpp << "#include <string>\n";
    cpp << "#include \"aeongui/CompiledDocument.hpp\"\n";
    cpp << "#include \"aeongui/AttributeMap.hpp\"\n";
    cpp << "#include \"aeongui/ElementFactory.hpp\"\n";
    cpp << "#include \"aeongui/dom/Document.hpp\"\n";
    cpp << "#include \"aeongui/dom/Window.hpp\"\n";
    cpp << "#include \"aeongui/dom/Node.hpp\"\n";
    cpp << "#include \"aeongui/dom/Element.hpp\"\n";
    cpp << "#include \"aeongui/dom/Text.hpp\"\n";
    cpp << "#include \"aeongui/dom/Event.hpp\"\n";
    cpp << "#include \"aeongui/dom/EventListener.hpp\"\n\n";

    // Bring the common names into scope so verbatim scripts can use them
    // unqualified: AttributeMap, CompiledDocument, and everything in DOM
    // (Window, Document, Element, Event, Node, Text, ...).
    cpp << "using AeonGUI::AttributeMap;\n";
    cpp << "using AeonGUI::CompiledDocument;\n";
    cpp << "using namespace AeonGUI::DOM;\n\n";

    cpp << "namespace\n{\n";
    cpp << "    /** @brief Adapts a std::function to the EventListener interface. */\n";
    cpp << "    class xmlcxx_Adapter : public AeonGUI::DOM::EventListener\n";
    cpp << "    {\n    public:\n";
    cpp << "        using Fn = std::function<void ( AeonGUI::DOM::Event& )>;\n";
    cpp << "        explicit xmlcxx_Adapter ( Fn aFn ) : mFn ( std::move ( aFn ) ) {}\n";
    cpp << "        ~xmlcxx_Adapter() override = default;\n";
    cpp << "        void handleEvent ( AeonGUI::DOM::Event& event ) override\n";
    cpp << "        {\n            if ( mFn ) { mFn ( event ); }\n        }\n";
    cpp << "    private:\n        Fn mFn;\n    };\n";
    cpp << "}\n\n";

    if ( !options.nameSpace.empty() )
    {
        cpp << "namespace " << options.nameSpace << "\n{\n";
    }

    // Verbatim script bodies at file scope. DOCCLASS expands to the document
    // class name so a script may define members directly (e.g.
    // void DOCCLASS::Foo() { ... }) when declared, in addition to free
    // functions, statics, types, etc.
    cpp << "#define DOCCLASS " << options.className << "\n";
    cpp << gen.scripts.str();
    cpp << "#undef DOCCLASS\n\n";

    cpp << options.className << "::" << options.className << "() = default;\n";
    cpp << options.className << "::~" << options.className << "() = default;\n\n";

    cpp << "void " << options.className
        << "::BuildDOM ( AeonGUI::DOM::Document& document )\n{\n";
    cpp << "    {\n";
    cpp << gen.build.str();
    cpp << "    }\n";
    cpp << "}\n\n";

    cpp << "void " << options.className
        << "::OnLoad ( AeonGUI::DOM::Document& document, AeonGUI::DOM::Window& window )\n{\n";
    cpp << "    (void) document;\n    (void) window;\n";
    for ( const auto& fn : gen.loadFns )
    {
        cpp << "    " << fn << " ( *this, window, document );\n";
    }
    cpp << "}\n";

    if ( !options.nameSpace.empty() )
    {
        cpp << "}\n";
    }

    std::cout << "xmlcxx: wrote " << options.output;
    if ( !headerPath.empty() )
    {
        std::cout << " and " << headerPath;
    }
    std::cout << " (class " << options.className << ", "
              << gen.loadFns.size() << " onload binding(s))\n";
    return 0;
}
