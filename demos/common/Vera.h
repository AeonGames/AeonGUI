#ifndef VERA_H
#define VERA_H
/******************************************************************************
Copyright 2010-2013 Rodrigo Hernandez Cordoba

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

#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
extern struct {
uint32_t  size;
uint8_t data[75812];
} Vera;
#ifdef __cplusplus
}
#endif
#endif

