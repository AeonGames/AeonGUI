# AeonGUI

[![Windows Build](https://github.com/AeonGames/AeonGUI/actions/workflows/build-windows.yml/badge.svg)](https://github.com/AeonGames/AeonGUI/actions/workflows/build-windows.yml)
[![MSYS2 Build](https://github.com/AeonGames/AeonGUI/actions/workflows/build-msys2.yml/badge.svg)](https://github.com/AeonGames/AeonGUI/actions/workflows/build-msys2.yml)
[![Ubuntu Build](https://github.com/AeonGames/AeonGUI/actions/workflows/build-ubuntu.yml/badge.svg)](https://github.com/AeonGames/AeonGUI/actions/workflows/build-ubuntu.yml)
[![macOS Build](https://github.com/AeonGames/AeonGUI/actions/workflows/build-macos.yml/badge.svg)](https://github.com/AeonGames/AeonGUI/actions/workflows/build-macos.yml)

AeonGUI is a cross-platform C++ GUI and SVG rendering library focused on game UI and interactive applications.
It implements a subset of the SVG DOM and CSS styling pipeline, with a backend designed to remain rendering-API agnostic.

Keywords: `C++ GUI`, `SVG renderer`, `game UI`, `cross-platform UI library`, `CMake`, `Cairo`, `Pango`, `libxml2`.

## Project Status

AeonGUI is under active development and still evolving. APIs and behavior may change.

- Good fit: experimentation, prototyping, engine integration research, SVG-based UI workflows.
- Not yet ideal: long-term API stability guarantees.

## What It Includes

- SVG DOM subset with scene graph traversal and element factory architecture.
- CSS-based styling through `libcss`.
- Text layout and font shaping via `Pango` and `Fontconfig`.
- XML parsing via `libxml2`.
- Raster image support with magic-based detection (PNG optional, JPEG via `libjpeg-turbo` optional, and PCX).
- OpenGL demo application (`OpenGLDemo`) for quick validation.
- Unit tests with GoogleTest/GoogleMock.

## Platform Support

CI builds currently run on:

- Windows (MSVC)
- Windows (MSYS2: `mingw64`, `ucrt64`, `clang64`)
- Ubuntu
- macOS

## Dependencies

Core dependencies used by the project:

- `cairo`
- `pango`
- `fontconfig`
- `libxml2`
- `zlib`
- `libpng` (optional, controlled by `USE_PNG`)
- `libjpeg-turbo` (optional, controlled by `USE_JPEG`)
- `gtest`/`gmock` for tests

For MSVC builds, the repository is configured for `vcpkg` manifests (`vcpkg.json`).

## Quick Start

### 1) Clone

```bash
git clone https://github.com/AeonGames/AeonGUI.git
cd AeonGUI
```

### 2) Configure + Build

#### Windows (MSVC + vcpkg)

```bash
cmake -B build -DCMAKE_TOOLCHAIN_FILE="C:/vcpkg/scripts/buildsystems/vcpkg.cmake"
cmake --build build
```

#### Windows (MSYS2 / MinGW)

```bash
cmake -G "MSYS Makefiles" -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

#### Linux / macOS

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

### 3) Run tests

```bash
ctest --test-dir build --output-on-failure
```

### 4) Run the demo

`OpenGLDemo` is produced under `build/bin` (platform naming may vary).

## Common CMake Options

- `-DUSE_ZLIB=ON|OFF` Enable/disable zlib integration.
- `-DUSE_PNG=ON|OFF` Enable/disable PNG decoding.
- `-DUSE_JPEG=ON|OFF` Enable/disable JPEG decoding (`libjpeg-turbo`).
- `-DUSE_CUDA=ON|OFF` Enable optional CUDA code path (unused right now, so it has no effect).
- `-DCMAKE_DISABLE_PRECOMPILE_HEADERS=ON` Disable PCH during iteration.
- `-DCMAKE_BUILD_TYPE=Debug|Release|RelWithDebInfo|MinSizeRel`

Example:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DUSE_PNG=ON -DUSE_JPEG=ON
```

## Repository Layout

- `core/`: main library implementation.
- `include/aeongui/`: public headers.
- `core/dom/`: SVG/DOM classes.
- `css/`: CSS engine dependencies and integration.
- `tests/`: unit tests.
- `demos/OpenGL/`: sample app.
- `tools/`: developer utilities.

## Notes For New Contributors

- Use CMake out-of-source builds (`-B build`).
- A pre-commit hook is configured by CMake; it validates style and notices.
- If iteration speed matters, disable PCH with `-DCMAKE_DISABLE_PRECOMPILE_HEADERS=ON`.
- CI workflows in `.github/workflows/` are the best reference for known-good dependency sets.

## Developer Environment Tips (Optional)

### Installing Zsh on MSYS2

If you want to use Zsh on MSYS2 with the `powerlevel10k` theme:

```bash
pacman -S zsh
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
```

After installing oh-my-zsh, install `powerlevel10k` with:

```bash
git clone --depth=1 https://github.com/romkatv/powerlevel10k.git ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/themes/powerlevel10k
sed -i "s/^ZSH_THEME=.*/ZSH_THEME=\"powerlevel10k\/powerlevel10k\"/" ~/.zshrc
```

To make MSYS2 terminals default to zsh, edit the corresponding `.ini` files under `C:\msys64` and add:

```ini
SHELL=/usr/bin/zsh
```

### VS Code shell integration for MSYS2 terminals

If you use MSYS2 `bash` or `zsh` in VS Code, add this to `.bashrc` or `.zshrc`:

```bash
export PATH="$PATH:$(cygpath "$LOCALAPPDATA/Programs/Microsoft VS Code/bin")"
[[ "$TERM_PROGRAM" == "vscode" ]] && . "$(cygpath "$(code --locate-shell-integration-path <zsh|bash>)")"
```

Replace `<zsh|bash>` with the shell you are configuring.

## FAQ

### Does AeonGUI depend on V8/JavaScript?

No. Historical experiments existed, but the current codebase is C++ focused and does not require V8.

## License

AeonGUI is released under the [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0).

The Aeon Games logo is **not** covered by Apache 2.0 and may not be used without permission.

## Authors

- Rodrigo Hernandez (`kwizatz` at `aeongames` dot `com`)
