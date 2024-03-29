#!/usr/bin/python
# Copyright (C) 2023 Rodrigo Jose Hernandez Cordoba
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re

ALIAS_FILE = 'build/Aliases'
ALIAS_INC  = 'src/charset/aliases.inc'

UNICODE_CHARSETS = [
   '^ISO-10646-UCS-[24]$',
   '^UTF-16',
   '^UTF-8$',
   '^UTF-32'
  ]

file = open(ALIAS_FILE, 'r')

charsets = {}

for line in file:
   if line.startswith('#'):
      continue
   line = line.rstrip()
   if line == '':
      continue
   elements = line.split()
   charsets[elements[0]] = [elements[1], elements[2:]];
file.close()

unicode_macro = ""

output = """/*
Copyright (C) 2019,2020 Rodrigo Jose Hernandez Cordoba

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Note: This file is automatically generated by make-aliases.py

Do not edit file file, changes will be overwritten during build.
*/

static parserutils_charset_aliases_canon canonical_charset_names[] = {
"""

aliases = {}
canon_number = 0
for (canon, charset) in sorted(charsets.items()):
   output += "\t{ " + charset[0] + ", " + str(len(canon)) + ', "' + canon + '" },' + "\n";
   is_unicode = False
   for unicode_regex in UNICODE_CHARSETS:
      if re.match(unicode_regex, canon) != None:
         unicode_macro += "((x) == " + charset[0] + ") || "
         break
      
   canon = canon.lower()
   canon = ''.join(filter(str.isalnum, canon))
   aliases[canon] = canon_number
   for alias in charset[1]:
      alias = alias.lower()
      alias = ''.join(filter(str.isalnum, alias))
      aliases[alias] = canon_number
   canon_number += 1

output += "};\n\nstatic const uint16_t charset_aliases_canon_count = " + str(canon_number) + ";\n\n"

output += """typedef struct {
	uint16_t name_len;
	const char *name;
	parserutils_charset_aliases_canon *canon;
} parserutils_charset_aliases_alias;

static parserutils_charset_aliases_alias charset_aliases[] = {
"""

for (alias, number) in sorted(aliases.items()):
   output += "\t{ " + str(len(alias)) + ', "' + alias + '", &canonical_charset_names[' + str(number) + "] },\n";

output += "};\n\n"

unicode_macro = unicode_macro[:-4]
output += """static const uint16_t charset_aliases_count = {0};

#define MIBENUM_IS_UNICODE(x) ({1})
""".format(len(aliases), unicode_macro)

file = open(ALIAS_INC, 'wb')
file.write(bytearray(output, 'utf8'))
file.close()

