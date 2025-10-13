/*
Copyright (C) 2025 Rodrigo Jose Hernandez Cordoba

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
#ifndef AEONGUI_DOMEXCEPTION_HPP
#define AEONGUI_DOMEXCEPTION_HPP

#include <exception>
#include "aeongui/Platform.hpp"
#include "DOMString.hpp"
namespace AeonGUI
{
    namespace DOM
    {
        class DLL DOMException : public std::exception
        {
        public:
            enum ExceptionCode : unsigned short
            {
                INDEX_SIZE_ERR = 1,
                DOMSTRING_SIZE_ERR = 2,
                HIERARCHY_REQUEST_ERR = 3,
                WRONG_DOCUMENT_ERR = 4,
                INVALID_CHARACTER_ERR = 5,
                NO_DATA_ALLOWED_ERR = 6,
                NO_MODIFICATION_ALLOWED_ERR = 7,
                NOT_FOUND_ERR = 8,
                NOT_SUPPORTED_ERR = 9,
                INUSE_ATTRIBUTE_ERR = 10,
                INVALID_STATE_ERR = 11,
                SYNTAX_ERR = 12,
                INVALID_MODIFICATION_ERR = 13,
                NAMESPACE_ERR = 14,
                INVALID_ACCESS_ERR = 15,
                VALIDATION_ERR = 16,
                TYPE_MISMATCH_ERR = 17,
                SECURITY_ERR = 18,
                NETWORK_ERR = 19,
                ABORT_ERR = 20,
                URL_MISMATCH_ERR = 21,
                QUOTA_EXCEEDED_ERR = 22,
                TIMEOUT_ERR = 23,
                INVALID_NODE_TYPE_ERR = 24,
                DATA_CLONE_ERR = 25
            };
            // Override the virtual what() function
            const char* what() const noexcept override
            {
                return mName.c_str();
            }
            const DOMString& message() const
            {
                return mMessage;
            }
            const DOMString& name() const
            {
                return mName;
            }
            unsigned short code() const
            {
                return mCode;
            }
        protected:
            // Constructor to initialize the message
            DOMException ( unsigned short code, const DOMString& message = "", const DOMString& name = "Error" ) : mMessage ( message ), mName ( name ), mCode ( code ) {}
            virtual ~DOMException() = default;
            DOMString mMessage{};
            DOMString mName{"Error"};
            unsigned short mCode{0};
        };

        class DOMIndexSizeError : public DOMException
        {
        public:
            DOMIndexSizeError ( const DOMString& message = "", const DOMString& name = "Index size error" ) : DOMException ( DOMException::INDEX_SIZE_ERR, message, name ) {}
            virtual ~DOMIndexSizeError() = default;
        };

        class DOMStringSizeError : public DOMException
        {
        public:
            DOMStringSizeError ( const DOMString& message = "", const DOMString& name = "DOM string size error" ) : DOMException ( DOMException::DOMSTRING_SIZE_ERR, message, name ) {}
            virtual ~DOMStringSizeError() = default;
        };

        class DOMHierarchyRequestError : public DOMException
        {
        public:
            DOMHierarchyRequestError ( const DOMString& message = "", const DOMString& name = "Hierarchy request error" ) : DOMException ( DOMException::HIERARCHY_REQUEST_ERR, message, name ) {}
            virtual ~DOMHierarchyRequestError() = default;
        };

        class DOMWrongDocumentError : public DOMException
        {
        public:
            DOMWrongDocumentError ( const DOMString& message = "", const DOMString& name = "Wrong document error" ) : DOMException ( DOMException::WRONG_DOCUMENT_ERR, message, name ) {}
            virtual ~DOMWrongDocumentError() = default;
        };

        class DOMInvalidCharacterError : public DOMException
        {
        public:
            DOMInvalidCharacterError ( const DOMString& message = "", const DOMString& name = "Invalid character error" ) : DOMException ( DOMException::INVALID_CHARACTER_ERR, message, name ) {}
            virtual ~DOMInvalidCharacterError() = default;
        };

        class DOMNoDataAllowedError : public DOMException
        {
        public:
            DOMNoDataAllowedError ( const DOMString& message = "", const DOMString& name = "No data allowed error" ) : DOMException ( DOMException::NO_DATA_ALLOWED_ERR, message, name ) {}
            virtual ~DOMNoDataAllowedError() = default;
        };

        class DOMNoModificationAllowedError : public DOMException
        {
        public:
            DOMNoModificationAllowedError ( const DOMString& message = "", const DOMString& name = "No modification allowed error" ) : DOMException ( DOMException::NO_MODIFICATION_ALLOWED_ERR, message, name ) {}
            virtual ~DOMNoModificationAllowedError() = default;
        };

        class DOMNotFoundError : public DOMException
        {
        public:
            DOMNotFoundError ( const DOMString& message = "", const DOMString& name = "Not found error" ) : DOMException ( DOMException::NOT_FOUND_ERR, message, name ) {}
            virtual ~DOMNotFoundError() = default;
        };

        class DOMNotSupportedError : public DOMException
        {
        public:
            DOMNotSupportedError ( const DOMString& message = "", const DOMString& name = "Not supported error" ) : DOMException ( DOMException::NOT_SUPPORTED_ERR, message, name ) {}
            virtual ~DOMNotSupportedError() = default;
        };

        class DOMInUseAttributeError : public DOMException
        {
        public:
            DOMInUseAttributeError ( const DOMString& message = "", const DOMString& name = "In use attribute error" ) : DOMException ( DOMException::INUSE_ATTRIBUTE_ERR, message, name ) {}
            virtual ~DOMInUseAttributeError() = default;
        };

        class DOMInvalidStateError : public DOMException
        {
        public:
            DOMInvalidStateError ( const DOMString& message = "", const DOMString& name = "Invalid state error" ) : DOMException ( DOMException::INVALID_STATE_ERR, message, name ) {}
            virtual ~DOMInvalidStateError() = default;
        };

        class DOMSyntaxError : public DOMException
        {
        public:
            DOMSyntaxError ( const DOMString& message = "", const DOMString& name = "Syntax error" ) : DOMException ( DOMException::SYNTAX_ERR, message, name ) {}
            virtual ~DOMSyntaxError() = default;
        };

        class DOMInvalidModificationError : public DOMException
        {
        public:
            DOMInvalidModificationError ( const DOMString& message = "", const DOMString& name = "Invalid modification error" ) : DOMException ( DOMException::INVALID_MODIFICATION_ERR, message, name ) {}
            virtual ~DOMInvalidModificationError() = default;
        };

        class DOMNamespaceError : public DOMException
        {
        public:
            DOMNamespaceError ( const DOMString& message = "", const DOMString& name = "Namespace error" ) : DOMException ( DOMException::NAMESPACE_ERR, message, name ) {}
            virtual ~DOMNamespaceError() = default;
        };

        class DOMInvalidAccessError : public DOMException
        {
        public:
            DOMInvalidAccessError ( const DOMString& message = "", const DOMString& name = "Invalid access error" ) : DOMException ( DOMException::INVALID_ACCESS_ERR, message, name ) {}
            virtual ~DOMInvalidAccessError() = default;
        };

        class DOMValidationError : public DOMException
        {
        public:
            DOMValidationError ( const DOMString& message = "", const DOMString& name = "Validation error" ) : DOMException ( DOMException::VALIDATION_ERR, message, name ) {}
            virtual ~DOMValidationError() = default;
        };

        class DOMTypeMismatchError : public DOMException
        {
        public:
            DOMTypeMismatchError ( const DOMString& message = "", const DOMString& name = "Type mismatch error" ) : DOMException ( DOMException::TYPE_MISMATCH_ERR, message, name ) {}
            virtual ~DOMTypeMismatchError() = default;
        };

        class DOMSecurityError : public DOMException
        {
        public:
            DOMSecurityError ( const DOMString& message = "", const DOMString& name = "Security error" ) : DOMException ( DOMException::SECURITY_ERR, message, name ) {}
            virtual ~DOMSecurityError() = default;
        };
        class DOMNetworkError : public DOMException
        {
        public:
            DOMNetworkError ( const DOMString& message = "", const DOMString& name = "Network error" ) : DOMException ( DOMException::NETWORK_ERR, message, name ) {}
            virtual ~DOMNetworkError() = default;
        };
        class DOMAbortError : public DOMException
        {
        public:
            DOMAbortError ( const DOMString& message = "", const DOMString& name = "Abort error" ) : DOMException ( DOMException::ABORT_ERR, message, name ) {}
            virtual ~DOMAbortError() = default;
        };

        class DOMUrlMismatchError : public DOMException
        {
        public:
            DOMUrlMismatchError ( const DOMString& message = "", const DOMString& name = "URL mismatch error" ) : DOMException ( DOMException::URL_MISMATCH_ERR, message, name ) {}
            virtual ~DOMUrlMismatchError() = default;
        };

        class DOMQuotaExceededError : public DOMException
        {
        public:
            DOMQuotaExceededError ( const DOMString& message = "", const DOMString& name = "Quota exceeded error" ) : DOMException ( DOMException::QUOTA_EXCEEDED_ERR, message, name ) {}
            virtual ~DOMQuotaExceededError() = default;
        };

        class DOMTimeoutError : public DOMException
        {
        public:
            DOMTimeoutError ( const DOMString& message = "", const DOMString& name = "Timeout error" ) : DOMException ( DOMException::TIMEOUT_ERR, message, name ) {}
            virtual ~DOMTimeoutError() = default;
        };

        class DOMInvalidNodeTypeError : public DOMException
        {
        public:
            DOMInvalidNodeTypeError ( const DOMString& message = "", const DOMString& name = "Invalid node type error" ) : DOMException ( DOMException::INVALID_NODE_TYPE_ERR, message, name ) {}
            virtual ~DOMInvalidNodeTypeError() = default;
        };

        class DOMDataCloneError : public DOMException
        {
        public:
            DOMDataCloneError ( const DOMString& message = "", const DOMString& name = "Data clone error" ) : DOMException ( DOMException::DATA_CLONE_ERR, message, name ) {}
            virtual ~DOMDataCloneError() = default;
        };
    }
}

#endif
