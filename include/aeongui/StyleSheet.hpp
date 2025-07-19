/*
Copyright (C) 2023-2025 Rodrigo Jose Hernandez Cordoba

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
#ifndef AEONGUI_STYLESHEET_H
#define AEONGUI_STYLESHEET_H

#include <memory>

extern "C"
{
    struct css_stylesheet;
    struct css_select_ctx;
    struct css_select_results;
}

namespace AeonGUI
{
    struct css_stylesheet_deleter
    {
        void operator() ( css_stylesheet* p );
    };

    struct css_select_ctx_deleter
    {
        void operator() ( css_select_ctx* p );
    };

    struct css_select_results_deleter
    {
        void operator() ( css_select_results* p );
    };

    using StyleSheetPtr = std::unique_ptr<css_stylesheet, css_stylesheet_deleter>;
    using SelectCtxPtr = std::unique_ptr<css_select_ctx, css_select_ctx_deleter>;
    using SelectResultsPtr = std::unique_ptr<css_select_results, css_select_results_deleter>;
}
#endif
