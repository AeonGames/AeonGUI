/*
Copyright (C) 2016-2019 Rodrigo Jose Hernandez Cordoba

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
#include <vector>
#include <functional>
#include <memory>
#include <utility>
#include <tuple>
#include <algorithm>
#include "aeongui/StringLiteral.h"

#define FactoryImplementation(X) \
    std::unique_ptr<X> Construct##X ( const StringLiteral& aIdentifier )\
    { \
        return Factory<X>::Construct ( aIdentifier ); \
    } \
    std::unique_ptr<X> Construct##X ( const std::string& aIdentifier )\
    { \
        return Factory<X>::Construct ( aIdentifier ); \
    } \
    bool Register##X##Constructor ( const StringLiteral& aIdentifier, const std::function<std::unique_ptr<X>() >& aConstructor ) \
    { \
        return Factory<X>::RegisterConstructor ( aIdentifier, aConstructor );\
    }\
    bool Unregister##X##Constructor ( const StringLiteral& aIdentifier )\
    {\
        return Factory<X>::UnregisterConstructor ( aIdentifier );\
    }\
    void Enumerate##X##Constructors ( const std::function<bool ( const StringLiteral& ) >& aEnumerator )\
    {\
        Factory<X>::EnumerateConstructors ( aEnumerator );\
    }

namespace AeonGUI
{
    template<class T>
    class Factory
    {
    public:
        using Constructor = std::tuple<StringLiteral, std::function < std::unique_ptr<T>() >>;
        static std::unique_ptr<T> Construct ( const StringLiteral& aIdentifier )
        {
            auto it = std::find_if ( Constructors.begin(), Constructors.end(),
                                     [&aIdentifier] ( const Constructor & aConstructor )
            {
                return aIdentifier == std::get<0> ( aConstructor );
            } );
            if ( it != Constructors.end() )
            {
                return std::get<1> ( *it ) ();
            }
            return nullptr;
        }
        static std::unique_ptr<T> Construct ( const std::string& aIdentifier )
        {
            auto it = std::find_if ( Constructors.begin(), Constructors.end(),
                                     [&aIdentifier] ( const Constructor & aConstructor )
            {
                return std::get<0> ( aConstructor ) == aIdentifier;
            } );
            if ( it != Constructors.end() )
            {
                return std::get<1> ( *it ) ();
            }
            return nullptr;
        }
        static bool RegisterConstructor ( const StringLiteral& aIdentifier, const std::function < std::unique_ptr<T>() > & aConstructor )
        {
            auto it = std::find_if ( Constructors.begin(), Constructors.end(),
                                     [aIdentifier] ( const Constructor & aConstructor )
            {
                return aIdentifier == std::get<0> ( aConstructor );
            } );
            if ( it == Constructors.end() )
            {
                Constructors.emplace_back ( aIdentifier, aConstructor );
                return true;
            }
            return false;
        }
        static bool UnregisterConstructor ( const StringLiteral& aIdentifier )
        {
            auto it = std::find_if ( Constructors.begin(), Constructors.end(),
                                     [aIdentifier] ( const Constructor & aConstructor )
            {
                return aIdentifier == std::get<0> ( aConstructor );
            } );
            if ( it != Constructors.end() )
            {
                Constructors.erase ( it );
                return true;
            }
            return false;
        }
        static void EnumerateConstructors ( const std::function<bool ( const StringLiteral& ) >& aEnumerator )
        {
            for ( auto& i : Constructors )
            {
                if ( !aEnumerator ( std::get<0> ( i ) ) )
                {
                    return;
                }
            }
        }
    private:
        static std::vector < Constructor > Constructors;
    };
    template<class T>
    std::vector<std::tuple<StringLiteral, std::function < std::unique_ptr<T>() >>> Factory<T>::Constructors;
}
