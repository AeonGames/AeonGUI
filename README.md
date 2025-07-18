AeonGUI
=======

[![Windows Build status](https://github.com/AeonGames/AeonGUI/actions/workflows/build-windows.yml/badge.svg)](https://github.com/AeonGames/AeonGUI/actions/workflows/build-windows.yml) [![Ubuntu Build status](https://github.com/AeonGames/AeonGUI/actions/workflows/build-ubuntu.yml/badge.svg)](https://github.com/AeonGames/AeonGUI/actions/workflows/build-ubuntu.yml)

Last Updated: 11-23-2020

DESCRIPTION
-----------

Right now AeonGUI is a project to create a graphic user interface library to be used primarily on video games and interactive media.
It's primary goal is to stay agnostic of any advanced/dedicated graphics rendering API such as OpenGL, Vulkan, Direct3D, X11 or GDI.

**WARNING!!!: The library is not currently on a stable or even usable state, and is mostly a sandbox for ideas and POCs.**

HISTORY
-------

The idea for the library was born around 2004 or 2005 under the name 'Glitch', and has evolved overtime but never quite reached maturity.
The name change to AeonGUI came about as a way to identify the library as part of the AeonGames brand
and to avoid confusion since originally the name 'GLitch' was chosen to emphasize OpenGL support, which is no longer the priority.

On 2013 the goal of the library was to create a widget toolkit similar to Qt or GTK+ to be rendered inside the contexts of different
graphic APIs, but that idea was dropped due to the complexity required and the fact that the target for the library was video games,
not spreadsheets. The library remained mostly dormant while restart development on the [AeonEngine](https://github.com/AeonGames/AeonEngine)
was going on without the need for a GUI.

Somewhere between 2013 and 2018 the idea of the GUI as an HTML agent lingered, it would have been easy to create UIs for anyone
with CSS and HTML experience, no need to learn a new paradigm, but that idea died as well due to the complexity required plus
the actual similarities between HTML documents and video game UIs being very limited.

Right now, in 2019 the idea is to create a SVG agent that implements a subset of the SVG DOM, making UIs easily implemented via SVG files.
**I really hope this is the last iteration of what AeonGUI actually IS**

2023 Update: At one point in time implementing a full SVG agent with JavaScript bindings seemed like a good idea, it may be,
but compiling and maintaining V8 for 2 different compilers on 2 different platforms just for this purpose is very futile.
At this time GCC 11/12 has too many incompatibilities to be able to keep up, more time is spent trying to catch up than it is
really worth it, so JavaScript support is being dropped at this time. C# via Mono is being considered instead,
but the code will stay C++ for the moment and for the near future.

COMPILATION
-----------

The library uses CMake to build the required files used to build the binary depending on the platform.
For information on how to use CMake refer to its [official site](https://www.cmake.org).
There is only official support for Windows (MSYS2 and Visual Studio) and Linux at this time,
however, Linux builds may break from time to time as most development is done on Windows.

As of 11/23/2020 the library has a hard dependency on [v8](https://v8.dev/), Google's JavaScript engine,
the binaries for which can be obtained from [vcpkg](https://github.com/microsoft/vcpkg) for MSVC or Ubuntu,
or directly from [MSYS2](https://www.msys2.org/)'s repositories in the case of mingw-w64 builds.

LICENSE
-------

The library is released under the terms of the permissive [Apache 2.0 license](http://www.apache.org/licenses/LICENSE-2.0)
The Aeon Games logo is __NOT__ covered by the Apache license, it is a Trade Mark and may not be used for any purpose without permission.

TO-DO List

* Have the code that reads a document create element nodes wrapped into JavaScript classes.

MISCELLANEOUS
-------------

## Installing Zsh on MSYS2

If you want to use Zsh on MSYS2, with powerlevel10k theme you can install it with the following commands:

```bash
pacman -S zsh
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
```

After installing oh-my-zsh, you will be inside a zsh shell. You can install the powerlevel10k theme with:

```bash
git clone --depth=1 https://github.com/romkatv/powerlevel10k.git ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/themes/powerlevel10k
sed -i "s/^ZSH_THEME=.*/ZSH_THEME=\"powerlevel10k\/powerlevel10k\"/" ~/.zshrc
```

Then you can restart the shell to configure powerlevel10k.

To have the multiple Msys2 terminals use zsh instead of bash, you can edit the corresponding .ini files under C:\msys64 to add a `SHELL=/usr/bin/zsh` line.

AUTHORS
-------

Rodrigo Hernandez (kwizatz at aeongames dot com).
