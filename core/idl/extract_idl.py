#/bin/python3
# Copyright (C) 2024 Rodrigo Jose Hernandez Cordoba
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

def main():
    if len(sys.argv) != 3:
        print("Usage: extract_idl.py <reffy.json> <output.idl>")
        sys.exit(1)
    reffy = json.load(open(sys.argv[1], 'rt'))
    idl = open(sys.argv[2], 'wt')
    idl.write(reffy[0]['idl'])
    idl.close()

if __name__ == "__main__":
    main()