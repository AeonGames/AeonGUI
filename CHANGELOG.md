# Changelog

All notable changes to AeonGUI are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added

- **Basic native form controls (XHTML)** — `<form>`, `<input>`, `<button>`,
  `<textarea>`, `<label>`, `<fieldset>`, and `<legend>` are now real DOM
  elements with UA-stylesheet chrome, intrinsic layout sizes, widget
  painting, and interaction. `<input>` supports the original type set:
  `text`, `password`, `search`, `tel`, `url`, `email`, `number`, `hidden`,
  `checkbox`, `radio`, `button`, `submit`, and `reset`. Clicking hit-tests
  through the pick buffer to toggle checkables and run activation
  behavior; a focused control consumes keys for caret movement and text
  editing. `HTMLFormElement::GetFormData()` builds the submission entry
  list and `submit()`/`reset()` fire cancelable DOM events so the host
  decides what happens next — there is no networking. `type="file"` and
  `type="image"` are deliberately unsupported: they need privileges and
  submission semantics an embedded UI should not assume.
- **`<input type="range">`** — a horizontal slider with `min`, `max`, and
  `step` (including `step="any"`), value sanitization per the HTML value
  algorithm, `valueAsNumber()`/`setValueAsNumber()`, arrow/Page/Home/End
  keyboard stepping, and pointer capture so a drag keeps tracking after
  the cursor leaves the widget. The track and thumb pick up the computed
  `color`, so authors theme it with plain CSS.
- **CSS `:enabled` / `:disabled` / `:checked` and attribute selectors** —
  the libcss select handler now answers these from the DOM instead of
  returning a constant false, so `input[type="checkbox"]:checked` and
  friends match.

- **xmlcxx compiled documents** — a build-time tool (`tools/xmlcxx`) parses an
  `.xhtml`/`.svg` file with libxml2 and emits C++ that builds the DOM tree
  imperatively (no runtime XML parse) and embeds `type="text/c++"` `<script>`
  bodies and inline `onEVENT` handlers. Generated classes subclass the new
  `AeonGUI::CompiledDocument` black-box base, which exposes an id-agnostic
  host/guest contract (`SetCallback`/`Emit` named events over the DOM event
  infrastructure, plus `SetProperty`/`GetProperty`). The host owns the
  `DOM::Window` and loads a compiled document with the new
  `Window::Load(CompiledDocument&)` overload. A `cmake/xmlcxx.cmake` helper
  (`xmlcxx_generate`) wires generation into the build. Purely additive; the
  existing `Document::Load(filename)`/`location()` flow is unchanged.

### Fixed

- **Partial redraws clipped to the wrong rectangle** — the cached pick
  bounds kept only the last path an element drew instead of the union of
  all of them. Elements that emit several paths under one pick ID (form
  control chrome draws a background plus four border edges) reported a
  border sliver as their bounds, so `:hover`/`:active`/`:focus` changes
  did not reach the screen until an unrelated full redraw happened.

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
