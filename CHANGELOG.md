# Changelog

All notable changes to AeonGUI are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [0.9.0] — 2026-04-02

First pre-release milestone, covering the full SVG rendering pipeline,
dual-backend support, cross-platform CI, and a comprehensive test suite.

### Added

- **Dual 2D backend** — Cairo and Skia, selectable at build time via
  `BACKEND`.  Both backends produce identical pixel-buffer output
  and share the Pango + HarfBuzz text pipeline.
- **SVG DOM** — parser (libxml2) builds a DOM tree with support for
  `<svg>`, `<g>`, `<defs>`, `<use>`, `<rect>`, `<circle>`, `<ellipse>`,
  `<line>`, `<polyline>`, `<polygon>`, `<path>`, `<text>`, `<textPath>`,
  `<image>`, `<linearGradient>`, `<radialGradient>`, `<stop>`,
  `<feDropShadow>`, `<animate>`, and `<set>`.
- **CSS styling** — vendored libcss resolves cascaded styles; presentational
  hints map SVG attributes to CSS properties.
- **SVG path parser** — Flex/Bison grammar handles all SVG path commands
  (M/m, Z/z, L/l, H/h, V/v, C/c, S/s, Q/q, T/t, A/a).
- **SVG transform support** — `ParseSVGTransform()` handles `matrix()`,
  `translate()`, `scale()`, `rotate()`, `skewX()`, `skewY()`.
- **Text layout** — Pango + HarfBuzz for shaping, Fontconfig for font
  discovery, `<textPath>` with `getPointAtLength()`.
- **Image loading** — PNG, JPEG, and PCX raster image support.
- **Gradient fills** — linear and radial gradients via `CSS_PAINT_URI`.
- **SMIL animation** — basic `<animate>` and `<set>` element support.
- **Hit testing** — pick-buffer based hit testing with dirty-flag system
  and AABB dirty-rect partial redraw.
- **querySelector / querySelectorAll** — CSS selector queries on the DOM
  (type, id, class, compound, descendant, child, comma-list).
- **setAttribute / onAttributeChanged** — live DOM mutation with
  attribute-change callbacks.
- **Native plugin system** — loadable shared-library plugins for element
  construction.
- **DOMMatrix / DOMPoint** — W3C Geometry Interfaces (`DOMMatrix`,
  `DOMMatrixReadOnly`, `DOMPoint`, `DOMPointReadOnly`).
- **Thread safety** — `ParsePathData` (mutex), `FontDatabase`
  (recursive_mutex, Meyer's singleton), `ElementFactory` (mutex, Meyer's
  singleton), `Document::Load` (mutex for libwapcaplet), `SkiaCanvas`
  (magic statics).
- **Error handling** — normalized to exceptions with color-coded
  `LogLevel` logging.
- **Path data hint** — parser computes `cairo_path_data_t` size estimate
  at parse time; `CairoPath::Construct` falls back to an O(n) upper-bound
  scan when no hint is provided.
- **Cross-platform CI** — GitHub Actions workflows for Windows (MSVC +
  vcpkg, Cairo and Skia), MSYS2 (MinGW64, UCRT64, Clang64), Ubuntu (GCC),
  and macOS (Clang).
- **Rendering demos** — OpenGL, Vulkan, Direct3D 12, and Metal demo
  applications.
- **Unit tests** — 331 tests across 35 test suites (DOM, geometry,
  selectors, hit testing, path, thread safety, error handling).
- **Install targets** — CMake install rules with proper export sets.
- **Documentation** — Doxygen-generated API docs, `README.md` with
  build instructions, architecture overview, and developer tips.

### Fixed

- URL regex now handles tilde (`~`) in Windows 8.3 short paths.
- MSYS2 winpthreads `std::shared_mutex` crash on UCRT64 — replaced with
  plain `std::mutex` and Meyer's singleton pattern.
- vcpkg Skia include paths propagated correctly via `BUILD_INTERFACE`.
- FontSub.lib / Usp10.lib discovery on MSVC via Windows SDK path.
- `xmlInitParser()` called in `AeonGUI::Initialize()` for thread safety.

### Known Limitations

- SVG filter elements are stubbed but not rendered.
- No JavaScript / scripting engine.
- API is not yet stable.
