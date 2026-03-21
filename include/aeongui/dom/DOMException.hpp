/*
Copyright (C) 2025,2026 Rodrigo Jose Hernandez Cordoba

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
        /** @brief Base class for all DOM exceptions.
         *
         *  Encapsulates a numeric error code, a message, and a name.
         *  Derived classes represent specific W3C DOM exception types.
         *  @see https://webidl.spec.whatwg.org/#idl-DOMException
         */
        class DOMException : public std::exception
        {
        public:
            /** @brief W3C DOM exception codes. */
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
            /** @brief Get the human-readable error name.
             *  @return Null-terminated error name.
             */
            const char* what() const noexcept override
            {
                return mName.c_str();
            }
            /** @brief Get the error message.
             *  @return The message string.
             */
            const DOMString& message() const
            {
                return mMessage;
            }
            /** @brief Get the error name.
             *  @return The name string.
             */
            const DOMString& name() const
            {
                return mName;
            }
            /** @brief Get the numeric exception code.
             *  @return The exception code.
             */
            unsigned short code() const
            {
                return mCode;
            }
        protected:
            /** @brief Construct a DOMException.
             *  @param code    Numeric error code.
             *  @param message Human-readable message.
             *  @param name    Error name string.
             */
            DOMException ( unsigned short code, const DOMString& message = "", const DOMString& name = "Error" ) : mMessage ( message ), mName ( name ), mCode ( code ) {}
            virtual ~DOMException() = default;
            DOMString mMessage{}; ///< The error message.
            DOMString mName{"Error"}; ///< The error name.
            unsigned short mCode{ 0 }; ///< The numeric error code.
        };

        /** @brief Thrown when an index is out of range. */
        class DOMIndexSizeError : public DOMException
        {
        public:
            /** @brief Construct a DOMIndexSizeError.
             *  @param message Error message.
             *  @param name    Error name.
             */
            DOMIndexSizeError ( const DOMString& message = "", const DOMString& name = "Index size error" ) : DOMException ( DOMException::INDEX_SIZE_ERR, message, name ) {}
            virtual ~DOMIndexSizeError() = default;
        };

        /** @brief Thrown when a DOMString exceeds implementation limits. */
        class DOMStringSizeError : public DOMException
        {
        public:
            /** @brief Construct a DOMStringSizeError.
             *  @param message Error message.
             *  @param name    Error name.
             */
            DOMStringSizeError ( const DOMString& message = "", const DOMString& name = "DOM string size error" ) : DOMException ( DOMException::DOMSTRING_SIZE_ERR, message, name ) {}
            virtual ~DOMStringSizeError() = default;
        };

        /** @brief Thrown when a node is inserted into an invalid position. */
        class DOMHierarchyRequestError : public DOMException
        {
        public:
            /** @brief Construct a DOMHierarchyRequestError.
             *  @param message Error message.
             *  @param name    Error name.
             */
            DOMHierarchyRequestError ( const DOMString& message = "", const DOMString& name = "Hierarchy request error" ) : DOMException ( DOMException::HIERARCHY_REQUEST_ERR, message, name ) {}
            virtual ~DOMHierarchyRequestError() = default;
        };

        /** @brief Thrown when a node is used in a different document. */
        class DOMWrongDocumentError : public DOMException
        {
        public:
            /** @brief Construct a DOMWrongDocumentError.
             *  @param message Error message.
             *  @param name    Error name.
             */
            DOMWrongDocumentError ( const DOMString& message = "", const DOMString& name = "Wrong document error" ) : DOMException ( DOMException::WRONG_DOCUMENT_ERR, message, name ) {}
            virtual ~DOMWrongDocumentError() = default;
        };

        /** @brief Thrown when an invalid or illegal character is specified. */
        class DOMInvalidCharacterError : public DOMException
        {
        public:
            /** @brief Construct a DOMInvalidCharacterError.
             *  @param message Error message.
             *  @param name    Error name.
             */
            DOMInvalidCharacterError ( const DOMString& message = "", const DOMString& name = "Invalid character error" ) : DOMException ( DOMException::INVALID_CHARACTER_ERR, message, name ) {}
            virtual ~DOMInvalidCharacterError() = default;
        };

        /** @brief Thrown when data is specified for a node that does not support it. */
        class DOMNoDataAllowedError : public DOMException
        {
        public:
            /** @brief Construct a DOMNoDataAllowedError.
             *  @param message Error message.
             *  @param name    Error name.
             */
            DOMNoDataAllowedError ( const DOMString& message = "", const DOMString& name = "No data allowed error" ) : DOMException ( DOMException::NO_DATA_ALLOWED_ERR, message, name ) {}
            virtual ~DOMNoDataAllowedError() = default;
        };

        /** @brief Thrown when a modification is not allowed. */
        class DOMNoModificationAllowedError : public DOMException
        {
        public:
            /** @brief Construct a DOMNoModificationAllowedError.
             *  @param message Error message.
             *  @param name    Error name.
             */
            DOMNoModificationAllowedError ( const DOMString& message = "", const DOMString& name = "No modification allowed error" ) : DOMException ( DOMException::NO_MODIFICATION_ALLOWED_ERR, message, name ) {}
            virtual ~DOMNoModificationAllowedError() = default;
        };

        /** @brief Thrown when a referenced node does not exist. */
        class DOMNotFoundError : public DOMException
        {
        public:
            /** @brief Construct a DOMNotFoundError.
             *  @param message Error message.
             *  @param name    Error name.
             */
            DOMNotFoundError ( const DOMString& message = "", const DOMString& name = "Not found error" ) : DOMException ( DOMException::NOT_FOUND_ERR, message, name ) {}
            virtual ~DOMNotFoundError() = default;
        };

        /** @brief Thrown when a requested operation is not supported. */
        class DOMNotSupportedError : public DOMException
        {
        public:
            /** @brief Construct a DOMNotSupportedError.
             *  @param message Error message.
             *  @param name    Error name.
             */
            DOMNotSupportedError ( const DOMString& message = "", const DOMString& name = "Not supported error" ) : DOMException ( DOMException::NOT_SUPPORTED_ERR, message, name ) {}
            virtual ~DOMNotSupportedError() = default;
        };

        /** @brief Thrown when an attribute is already in use. */
        class DOMInUseAttributeError : public DOMException
        {
        public:
            /** @brief Construct a DOMInUseAttributeError.
             *  @param message Error message.
             *  @param name    Error name.
             */
            DOMInUseAttributeError ( const DOMString& message = "", const DOMString& name = "In use attribute error" ) : DOMException ( DOMException::INUSE_ATTRIBUTE_ERR, message, name ) {}
            virtual ~DOMInUseAttributeError() = default;
        };

        /** @brief Thrown when an object is in an invalid state. */
        class DOMInvalidStateError : public DOMException
        {
        public:
            /** @brief Construct a DOMInvalidStateError.
             *  @param message Error message.
             *  @param name    Error name.
             */
            DOMInvalidStateError ( const DOMString& message = "", const DOMString& name = "Invalid state error" ) : DOMException ( DOMException::INVALID_STATE_ERR, message, name ) {}
            virtual ~DOMInvalidStateError() = default;
        };

        /** @brief Thrown when a string does not match expected syntax. */
        class DOMSyntaxError : public DOMException
        {
        public:
            /** @brief Construct a DOMSyntaxError.
             *  @param message Error message.
             *  @param name    Error name.
             */
            DOMSyntaxError ( const DOMString& message = "", const DOMString& name = "Syntax error" ) : DOMException ( DOMException::SYNTAX_ERR, message, name ) {}
            virtual ~DOMSyntaxError() = default;
        };

        /** @brief Thrown when an invalid modification is attempted. */
        class DOMInvalidModificationError : public DOMException
        {
        public:
            /** @brief Construct a DOMInvalidModificationError.
             *  @param message Error message.
             *  @param name    Error name.
             */
            DOMInvalidModificationError ( const DOMString& message = "", const DOMString& name = "Invalid modification error" ) : DOMException ( DOMException::INVALID_MODIFICATION_ERR, message, name ) {}
            virtual ~DOMInvalidModificationError() = default;
        };

        /** @brief Thrown on a namespace error. */
        class DOMNamespaceError : public DOMException
        {
        public:
            /** @brief Construct a DOMNamespaceError.
             *  @param message Error message.
             *  @param name    Error name.
             */
            DOMNamespaceError ( const DOMString& message = "", const DOMString& name = "Namespace error" ) : DOMException ( DOMException::NAMESPACE_ERR, message, name ) {}
            virtual ~DOMNamespaceError() = default;
        };

        /** @brief Thrown when access to an object is denied. */
        class DOMInvalidAccessError : public DOMException
        {
        public:
            /** @brief Construct a DOMInvalidAccessError.
             *  @param message Error message.
             *  @param name    Error name.
             */
            DOMInvalidAccessError ( const DOMString& message = "", const DOMString& name = "Invalid access error" ) : DOMException ( DOMException::INVALID_ACCESS_ERR, message, name ) {}
            virtual ~DOMInvalidAccessError() = default;
        };

        /** @brief Thrown when validation fails. */
        class DOMValidationError : public DOMException
        {
        public:
            /** @brief Construct a DOMValidationError.
             *  @param message Error message.
             *  @param name    Error name.
             */
            DOMValidationError ( const DOMString& message = "", const DOMString& name = "Validation error" ) : DOMException ( DOMException::VALIDATION_ERR, message, name ) {}
            virtual ~DOMValidationError() = default;
        };

        /** @brief Thrown on a type mismatch. */
        class DOMTypeMismatchError : public DOMException
        {
        public:
            /** @brief Construct a DOMTypeMismatchError.
             *  @param message Error message.
             *  @param name    Error name.
             */
            DOMTypeMismatchError ( const DOMString& message = "", const DOMString& name = "Type mismatch error" ) : DOMException ( DOMException::TYPE_MISMATCH_ERR, message, name ) {}
            virtual ~DOMTypeMismatchError() = default;
        };

        /** @brief Thrown on a security violation. */
        class DOMSecurityError : public DOMException
        {
        public:
            /** @brief Construct a DOMSecurityError.
             *  @param message Error message.
             *  @param name    Error name.
             */
            DOMSecurityError ( const DOMString& message = "", const DOMString& name = "Security error" ) : DOMException ( DOMException::SECURITY_ERR, message, name ) {}
            virtual ~DOMSecurityError() = default;
        };
        /** @brief Thrown on a network error. */
        class DOMNetworkError : public DOMException
        {
        public:
            /** @brief Construct a DOMNetworkError.
             *  @param message Error message.
             *  @param name    Error name.
             */
            DOMNetworkError ( const DOMString& message = "", const DOMString& name = "Network error" ) : DOMException ( DOMException::NETWORK_ERR, message, name ) {}
            virtual ~DOMNetworkError() = default;
        };
        /** @brief Thrown when an operation is aborted. */
        class DOMAbortError : public DOMException
        {
        public:
            /** @brief Construct a DOMAbortError.
             *  @param message Error message.
             *  @param name    Error name.
             */
            DOMAbortError ( const DOMString& message = "", const DOMString& name = "Abort error" ) : DOMException ( DOMException::ABORT_ERR, message, name ) {}
            virtual ~DOMAbortError() = default;
        };

        /** @brief Thrown when a URL does not match expectations. */
        class DOMUrlMismatchError : public DOMException
        {
        public:
            /** @brief Construct a DOMUrlMismatchError.
             *  @param message Error message.
             *  @param name    Error name.
             */
            DOMUrlMismatchError ( const DOMString& message = "", const DOMString& name = "URL mismatch error" ) : DOMException ( DOMException::URL_MISMATCH_ERR, message, name ) {}
            virtual ~DOMUrlMismatchError() = default;
        };

        /** @brief Thrown when a storage quota is exceeded. */
        class DOMQuotaExceededError : public DOMException
        {
        public:
            /** @brief Construct a DOMQuotaExceededError.
             *  @param message Error message.
             *  @param name    Error name.
             */
            DOMQuotaExceededError ( const DOMString& message = "", const DOMString& name = "Quota exceeded error" ) : DOMException ( DOMException::QUOTA_EXCEEDED_ERR, message, name ) {}
            virtual ~DOMQuotaExceededError() = default;
        };

        /** @brief Thrown when an operation times out. */
        class DOMTimeoutError : public DOMException
        {
        public:
            /** @brief Construct a DOMTimeoutError.
             *  @param message Error message.
             *  @param name    Error name.
             */
            DOMTimeoutError ( const DOMString& message = "", const DOMString& name = "Timeout error" ) : DOMException ( DOMException::TIMEOUT_ERR, message, name ) {}
            virtual ~DOMTimeoutError() = default;
        };

        /** @brief Thrown when an invalid node type is used. */
        class DOMInvalidNodeTypeError : public DOMException
        {
        public:
            /** @brief Construct a DOMInvalidNodeTypeError.
             *  @param message Error message.
             *  @param name    Error name.
             */
            DOMInvalidNodeTypeError ( const DOMString& message = "", const DOMString& name = "Invalid node type error" ) : DOMException ( DOMException::INVALID_NODE_TYPE_ERR, message, name ) {}
            virtual ~DOMInvalidNodeTypeError() = default;
        };

        /** @brief Thrown when data cannot be cloned. */
        class DOMDataCloneError : public DOMException
        {
        public:
            /** @brief Construct a DOMDataCloneError.
             *  @param message Error message.
             *  @param name    Error name.
             */
            DOMDataCloneError ( const DOMString& message = "", const DOMString& name = "Data clone error" ) : DOMException ( DOMException::DATA_CLONE_ERR, message, name ) {}
            virtual ~DOMDataCloneError() = default;
        };
    }
}

#endif
