#!/usr/bin/env python3
"""
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

AeonGUI Class Generator

This script generates C++ class files (.hpp and .cpp) with proper structure
and inheritance for the AeonGUI project.

Usage:
    python generate_class.py <ClassName> [BaseClassName]

Arguments:
    ClassName     - The name of the class to generate (required)
    BaseClassName - The name of the base class to inherit from (optional)

Examples:
    python generate_class.py SVGElement
    python generate_class.py SVGGraphicsElement SVGElement
"""

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def get_copyright_header():
    """Generate the copyright header with current year."""
    current_year = datetime.now().year
    return f"""/*
Copyright (C) {current_year} Rodrigo Jose Hernandez Cordoba

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/"""


def generate_header_guard(class_name):
    """Generate the header guard macro name."""
    return f"AEONGUI_{class_name.upper()}_HPP"


def check_base_class_has_virtual_destructor(project_root, base_class):
    """Check if the base class has a virtual destructor."""
    if not base_class:
        return False
    
    base_header_path = project_root / "include" / "aeongui" / "dom" / f"{base_class}.hpp"
    
    if not base_header_path.exists():
        return False
    
    try:
        with open(base_header_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Look for virtual destructor patterns
        # This is a simple regex that looks for "virtual" followed by "~ClassName"
        virtual_destructor_patterns = [
            rf'virtual\s+~{base_class}\s*\(',
            r'virtual\s+~\w+\s*\(',
        ]
        
        for pattern in virtual_destructor_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return True
        
        return False
        
    except Exception:
        return False


def should_use_virtual_destructor(project_root, class_name, base_class):
    """Determine if virtual destructor should be used for this class."""
    if not base_class:
        # No inheritance, no need for virtual destructor by default
        return False
    
    # If there's a base class, we should use virtual destructors
    return True


def ensure_base_class_has_virtual_destructor(project_root, base_class, force=False):
    """Ensure the base class has a virtual destructor, offer to update if not."""
    if not base_class:
        return True
    
    base_header_path = project_root / "include" / "aeongui" / "dom" / f"{base_class}.hpp"
    
    if not base_header_path.exists():
        print(f"Warning: Base class header {base_header_path} not found")
        return False
    
    if check_base_class_has_virtual_destructor(project_root, base_class):
        return True
    
    # Base class doesn't have virtual destructor
    print(f"Warning: Base class '{base_class}' does not have a virtual destructor.")
    
    if not force:
        try:
            response = input(f"Would you like to add 'virtual' to {base_class}'s destructor? (y/N): ")
        except EOFError:
            print("Cannot update base class destructor (no input available)")
            return False
        
        if response.lower() != 'y':
            print("Proceeding without updating base class destructor (not recommended)")
            return False
    
    # Update the base class header to add virtual destructor
    try:
        with open(base_header_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Look for non-virtual destructor and make it virtual
        destructor_pattern = rf'(\s+)~{base_class}\s*\(\s*\)\s*;'
        match = re.search(destructor_pattern, content)
        
        if match:
            # Replace with virtual destructor
            updated_content = re.sub(
                destructor_pattern, 
                rf'\1virtual ~{base_class}();',
                content
            )
            
            with open(base_header_path, 'w', encoding='utf-8') as f:
                f.write(updated_content)
            
            print(f"Updated: Added 'virtual' to {base_class} destructor in {base_header_path}")
            return True
        else:
            print(f"Could not find destructor pattern in {base_class} header")
            return False
            
    except Exception as e:
        print(f"Error updating base class header: {e}")
        return False


def generate_hpp_content(class_name, base_class=None, project_root=None):
    """Generate the .hpp file content."""
    copyright_header = get_copyright_header()
    header_guard = generate_header_guard(class_name)
    
    # Determine includes
    includes = ['#include "aeongui/Platform.hpp"']
    if base_class:
        includes.append(f'#include "aeongui/dom/{base_class}.hpp"')
    
    includes_str = '\n'.join(includes)
    
    # Determine inheritance
    inheritance_str = ""
    if base_class:
        inheritance_str = f" : public {base_class}"
    
    # Determine if we should use virtual destructor
    use_virtual = should_use_virtual_destructor(project_root, class_name, base_class)
    destructor_decl = f"virtual ~{class_name}();" if use_virtual else f"~{class_name}();"
    
    content = f"""{copyright_header}
#ifndef {header_guard}
#define {header_guard}

{includes_str}

namespace AeonGUI
{{
    namespace DOM
    {{
        class DLL {class_name}{inheritance_str}
        {{
        public:
            {class_name}();
            {destructor_decl}
        private:
        }};
    }}
}}

#endif
"""
    return content


def generate_cpp_content(class_name, base_class=None):
    """Generate the .cpp file content."""
    copyright_header = get_copyright_header()
    
    # Constructor implementation
    if base_class:
        constructor_impl = f"""        {class_name}::{class_name}() : {base_class}()
        {{
        }}"""
    else:
        constructor_impl = f"        {class_name}::{class_name}() = default;"
    
    content = f"""{copyright_header}
#include "aeongui/dom/{class_name}.hpp"

namespace AeonGUI
{{
    namespace DOM
    {{
{constructor_impl}

        {class_name}::~{class_name}() = default;
    }}
}}
"""
    return content


def write_file_safely(file_path, content, force=False):
    """Write content to file, creating directories if needed."""
    # Create parent directories if they don't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Check if file already exists
    if os.path.exists(file_path) and not force:
        try:
            response = input(f"File {file_path} already exists. Overwrite? (y/N): ")
        except EOFError:
            # Handle case where input is piped or not available
            print(f"File {file_path} already exists. Skipping (no input available).")
            return False
        
        if response.lower() != 'y':
            print(f"Skipping {file_path}")
            return False
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    return True


def update_cmake_lists(project_root, class_name, force=False, header_only=False):
    """Update CMakeLists.txt to include the new class files."""
    cmake_path = project_root / "core" / "CMakeLists.txt"
    
    if not cmake_path.exists():
        print(f"Warning: CMakeLists.txt not found at {cmake_path}")
        return False
    
    # Read the current CMakeLists.txt content
    with open(cmake_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Define the entries to add
    header_entry = f"    ../include/aeongui/dom/{class_name}.hpp"
    source_entry = f"    dom/{class_name}.cpp"
    
    # Check if entries already exist
    header_exists = header_entry in content
    source_exists = source_entry in content
    
    if header_only and header_exists:
        print(f"Header file already present in CMakeLists.txt")
        return True
    elif not header_only and header_exists and source_exists:
        print(f"Files already present in CMakeLists.txt")
        return True
    
    # Find the AEONGUI_DOM_HEADERS section and add the header
    headers_pattern = r'(set\(AEONGUI_DOM_HEADERS\n(?:.*\n)*?)(\))'
    headers_match = re.search(headers_pattern, content, re.MULTILINE)
    
    if headers_match:
        if header_entry not in content:
            # Insert the header before the closing parenthesis, maintaining alphabetical order
            headers_section = headers_match.group(1)
            headers_lines = headers_section.split('\n')
            
            # Find the insertion point (alphabetical order)
            insert_index = len(headers_lines) - 1  # Default to end
            for i, line in enumerate(headers_lines[1:], 1):  # Skip the set( line
                if line.strip().startswith('../include/aeongui/dom/'):
                    if line.strip() > header_entry:
                        insert_index = i
                        break
            
            headers_lines.insert(insert_index, header_entry)
            new_headers_section = '\n'.join(headers_lines)
            content = content.replace(headers_match.group(1), new_headers_section)
    else:
        print("Warning: Could not find AEONGUI_DOM_HEADERS section in CMakeLists.txt")
        return False
    
    # Find the AEONGUI_DOM_SOURCES section and add the source (only if not header-only)
    if not header_only:
        sources_pattern = r'(set\(AEONGUI_DOM_SOURCES\n(?:.*\n)*?)(\))'
        sources_match = re.search(sources_pattern, content, re.MULTILINE)
        
        if sources_match:
            if source_entry not in content:
                # Insert the source before the closing parenthesis, maintaining alphabetical order
                sources_section = sources_match.group(1)
                sources_lines = sources_section.split('\n')
                
                # Find the insertion point (alphabetical order)
                insert_index = len(sources_lines) - 1  # Default to end
                for i, line in enumerate(sources_lines[1:], 1):  # Skip the set( line
                    if line.strip().startswith('dom/'):
                        if line.strip() > source_entry:
                            insert_index = i
                            break
                
                sources_lines.insert(insert_index, source_entry)
                new_sources_section = '\n'.join(sources_lines)
                content = content.replace(sources_match.group(1), new_sources_section)
        else:
            print("Warning: Could not find AEONGUI_DOM_SOURCES section in CMakeLists.txt")
            return False
    
    # Write the updated content back
    try:
        if not force and cmake_path.exists():
            try:
                response = input(f"Update CMakeLists.txt? (y/N): ")
            except EOFError:
                print("Cannot update CMakeLists.txt (no input available)")
                return False
            
            if response.lower() != 'y':
                print("Skipping CMakeLists.txt update")
                return False
        
        with open(cmake_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Updated: {cmake_path}")
        return True
        
    except Exception as e:
        print(f"Error updating CMakeLists.txt: {e}")
        return False


def add_files_to_git(project_root, class_name, force=False, header_only=False):
    """Add the generated files to git."""
    hpp_path = project_root / "include" / "aeongui" / "dom" / f"{class_name}.hpp"
    cpp_path = project_root / "core" / "dom" / f"{class_name}.cpp"
    
    files_to_add = []
    if hpp_path.exists():
        files_to_add.append(str(hpp_path))
    if not header_only and cpp_path.exists():
        files_to_add.append(str(cpp_path))
    
    if not files_to_add:
        print("No files to add to git")
        return False
    
    try:
        # Check if we're in a git repository
        result = subprocess.run(['git', 'rev-parse', '--git-dir'], 
                              cwd=str(project_root), 
                              capture_output=True, 
                              text=True)
        if result.returncode != 0:
            print("Warning: Not in a git repository")
            return False
        
        # Add files to git
        cmd = ['git', 'add'] + files_to_add
        result = subprocess.run(cmd, 
                              cwd=str(project_root), 
                              capture_output=True, 
                              text=True)
        
        if result.returncode == 0:
            print(f"Added to git: {', '.join([os.path.basename(f) for f in files_to_add])}")
            return True
        else:
            print(f"Error adding files to git: {result.stderr}")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"Error running git command: {e}")
        return False
    except FileNotFoundError:
        print("Error: git command not found")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Generate C++ class files for AeonGUI project',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s SVGElement
  %(prog)s SVGGraphicsElement SVGElement
  %(prog)s MyCustomClass BaseClass
  %(prog)s --force ExistingClass  # Overwrite without prompting
  %(prog)s --cmake --git NewClass  # Generate, update CMake, and add to git
  %(prog)s --header-only MyTemplateClass  # Generate only header file
        """
    )
    
    parser.add_argument('class_name', 
                       help='Name of the class to generate')
    parser.add_argument('base_class', 
                       nargs='?', 
                       help='Name of the base class to inherit from (optional)')
    parser.add_argument('--force', '-f',
                       action='store_true',
                       help='Overwrite existing files without prompting')
    parser.add_argument('--cmake', '-c',
                       action='store_true',
                       help='Automatically add files to CMakeLists.txt')
    parser.add_argument('--git', '-g',
                       action='store_true',
                       help='Automatically add files to git')
    parser.add_argument('--header-only', '-ho',
                       action='store_true',
                       help='Generate only the header file (.hpp), no source file (.cpp)')
    
    args = parser.parse_args()
    
    class_name = args.class_name
    base_class = args.base_class
    
    # Validate class name
    if not class_name.isidentifier():
        print(f"Error: '{class_name}' is not a valid C++ class name")
        return 1
    
    if base_class and not base_class.isidentifier():
        print(f"Error: '{base_class}' is not a valid C++ class name")
        return 1
    
    # Get project root directory (assuming script is in tools/)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Check and ensure proper virtual destructor setup for inheritance
    if base_class:
        base_updated = ensure_base_class_has_virtual_destructor(project_root, base_class, args.force)
        if not base_updated:
            print("Warning: Base class may not have proper virtual destructor")
    
    # Define output paths
    hpp_path = project_root / "include" / "aeongui" / "dom" / f"{class_name}.hpp"
    cpp_path = project_root / "core" / "dom" / f"{class_name}.cpp"
    
    # Generate content
    hpp_content = generate_hpp_content(class_name, base_class, project_root)
    
    # Write files
    print(f"Generating class '{class_name}'", end="")
    if base_class:
        print(f" inheriting from '{base_class}'")
    else:
        print()
    
    if args.header_only:
        print("(header-only mode)")
    
    hpp_written = write_file_safely(str(hpp_path), hpp_content, args.force)
    
    # Only generate and write .cpp file if not in header-only mode
    cpp_written = False
    if not args.header_only:
        cpp_content = generate_cpp_content(class_name, base_class)
        cpp_written = write_file_safely(str(cpp_path), cpp_content, args.force)
    
    if hpp_written:
        print(f"Created: {hpp_path}")
    if cpp_written:
        print(f"Created: {cpp_path}")
    elif not args.header_only:
        # .cpp file was supposed to be created but wasn't
        pass
    
    # Update CMakeLists.txt if requested and files were created
    cmake_updated = False
    if args.cmake and (hpp_written or cpp_written):
        cmake_updated = update_cmake_lists(project_root, class_name, args.force, args.header_only)
    
    # Add files to git if requested and files were created
    git_added = False
    if args.git and (hpp_written or cpp_written):
        git_added = add_files_to_git(project_root, class_name, args.force, args.header_only)
    
    if hpp_written or cpp_written:
        print("\nDon't forget to:")
        next_step = 1
        if not args.header_only and (not args.cmake or not cmake_updated):
            print(f"{next_step}. Add the .cpp file to the appropriate CMakeLists.txt")
            next_step += 1
        if not args.git or not git_added:
            files_text = "files" if not args.header_only else "file"
            print(f"{next_step}. Add the {files_text} to git (git add)")
            next_step += 1
        print(f"{next_step}. Include any additional headers needed for your class")
        if not args.header_only:
            print(f"{next_step + 1}. Implement the class methods as needed")
        else:
            print(f"{next_step + 1}. Implement inline methods in the header if needed")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())