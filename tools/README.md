# AeonGUI Class Generator

A Python script to generate C++ class files with proper structure and inheritance for the AeonGUI project.

## Usage

```bash
python tools/generate_class.py <ClassName> [BaseClassName] [--force]
```

### Arguments

- `ClassName` (required): The name of the class to generate
- `BaseClassName` (optional): The name of the base class to inherit from
- `--force` or `-f` (optional): Overwrite existing files without prompting

### Examples

```bash
# Generate a simple class without inheritance
python tools/generate_class.py SVGElement

# Generate a class that inherits from another class
python tools/generate_class.py SVGGraphicsElement SVGElement

# Overwrite existing files without prompting
python tools/generate_class.py --force ExistingClass
```

## Features

- **Automatic file structure**: Creates both `.hpp` and `.cpp` files in the correct directories
- **Proper inheritance**: Automatically handles inheritance with proper include statements and constructor initialization
- **Copyright headers**: Includes the standard AeonGUI copyright header with current year
- **Header guards**: Generates proper header guard macros
- **Namespace structure**: Uses the correct `AeonGUI::DOM` namespace structure
- **Safety checks**: Prompts before overwriting existing files (unless `--force` is used)
- **Input validation**: Validates that class names are valid C++ identifiers

## Generated Structure

### Header File (.hpp)
- Located in `include/aeongui/dom/`
- Contains copyright header
- Proper header guards
- Necessary includes (`Platform.hpp` and base class if applicable)
- Class declaration with inheritance
- Public constructor and destructor declarations
- Empty private section for future members

### Implementation File (.cpp)
- Located in `core/dom/`
- Contains copyright header
- Includes the corresponding header file
- Constructor with proper base class initialization (if applicable)
- Destructor implementation
- Both use `= default` for standard behavior

## Directory Structure

The script assumes the following project structure:
```
project_root/
├── tools/
│   └── generate_class.py
├── include/aeongui/dom/
│   └── [generated .hpp files]
└── core/dom/
    └── [generated .cpp files]
```

## Post-Generation Tasks

After generating a class, remember to:

1. Add the `.cpp` file to the appropriate `CMakeLists.txt`
2. Include any additional headers needed for your class
3. Implement the class methods as needed
4. Add any necessary member variables to the private section
5. Update the constructor/destructor if needed (remove `= default` if custom logic is required)