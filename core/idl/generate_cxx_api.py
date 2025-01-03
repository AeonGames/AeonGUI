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

def main():
    if len(sys.argv) != 2:
        print("Usage: {} <parsed idl>.json".format(sys.argv[0]))
        sys.exit(1)
    idl = json.load(open(sys.argv[1], 'rt'))
    for i in idl['idlNames']:
        if idl['idlNames'][i]['type'] == 'interface':
            header = open("{0}.h".format(i), 'wt')
            header.write(license)
            header.write("#ifndef AEONGUI_{0}_H\n".format(i.upper()))
            header.write("#define AEONGUI_{0}_H\n".format(i.upper()))
            includes = []
            if idl['idlNames'][i]['inheritance'] != None:
                includes.append('{0}.h'.format(idl['idlNames'][i]['inheritance']))
            for j in includes:
                header.write("#include \"{0}\"\n".format(j))
            header.write("namespace AeonGUI\n{\n\tnamespace DOM\n\t{\n".expandtabs(4))
            header.write("\t}\n".expandtabs(4))
            header.write("}\n".expandtabs(4))
            header.write("#endif // AEONGUI_{0}_H\n".format(i.upper()))
            header.close()
if __name__ == "__main__":
    main()
