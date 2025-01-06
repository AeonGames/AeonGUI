#/bin/python3
# Copyright (C) 2024,2025 Rodrigo Jose Hernandez Cordoba
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import json

license = """/*
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
"""

IDL_TYPES = ['DOMString','boolean']

def write_header(filename, idl, includes, members):
            header = open(filename, 'wt')
            header.write(license)
            header.write("#ifndef AEONGUI_{0}_H\n".format(idl['name'].upper()))
            header.write("#define AEONGUI_{0}_H\n".format(idl['name'].upper()))
            for j in includes:
                header.write("#include \"{0}\"\n".format(j))
            header.write("namespace AeonGUI\n{\n\tnamespace DOM\n\t{\n".expandtabs(4))
            header.write("\t\t{} {} {}\n\t\t{{{}\n".expandtabs(4).format("class" if idl['type'] == 'interface' else "struct", idl['name'], '' if idl['inheritance'] == None else ": public {}".format(idl['inheritance']), "\n\t\tpublic:" if idl['type'] == 'interface' else "" ))
            header.write("\t\t\t{};\n".format("\n\t\t\t".join(members)).expandtabs(4))
            header.write("\t\t};\n".expandtabs(4))
            header.write("\t}\n".expandtabs(4))
            header.write("}\n".expandtabs(4))
            header.write("#endif // AEONGUI_{0}_H\n".format(idl['name'].upper()))
            header.close()

def main():
    if len(sys.argv) != 3:
        print("Usage: {} <parsed idl>.json <output directory>".format(sys.argv[0]))
        sys.exit(1)
    idl = json.load(open(sys.argv[1], 'rt'))
    for i in idl['idlNames']:
        if idl['idlNames'][i]['type'] == 'dictionary':
            includes = []
            members = []
            for member in idl['idlNames'][i]['members']:
                if member['type'] == 'field':
                    if member['idlType']['idlType'] in IDL_TYPES:
                        if 'WebIDL_Types.h' not in includes:
                            includes.append('WebIDL_Types.h')
                    else:
                        if member['idlType']['idlType'] not in includes:
                            includes.append("{0}.h".format(member['idlType']['idlType']))
                    print(member['default'])
                    members.append("{0} {1}{2};".format(member['idlType']['idlType'], member['name'], "" if member['default'] == None or 'value' not in member['default'] else "{{{}}}".format(member['default']['value'])))
            write_header("{0}/{1}.h".format(sys.argv[2],i),idl['idlNames'][i],includes,members)
        elif idl['idlNames'][i]['type'] == 'interface':
            includes = []
            members = []
            if idl['idlNames'][i]['inheritance'] != None:
                includes.append('{0}.h'.format(idl['idlNames'][i]['inheritance']))
            for member in idl['idlNames'][i]['members']:
                if member['type'] == 'constructor':
                    arguments = []
                    for argument in member['arguments']:
                        if argument['idlType']['idlType'] in IDL_TYPES:
                            if 'WebIDL_Types.h' not in includes:
                                includes.append('WebIDL_Types.h')
                        else:
                            if argument['idlType']['idlType'] not in includes:
                                includes.append("{0}.h".format(argument['idlType']['idlType']))
                        arguments.append("{0} {1}{2}".format(argument['idlType']['idlType'], argument['name'], " = {}" if argument['optional'] else "" ))
                    members.append("{0}({1})".format(i,", ".join(arguments)))
            write_header("{0}/{1}.h".format(sys.argv[2],i),idl['idlNames'][i],includes,members)
if __name__ == "__main__":
    main()
