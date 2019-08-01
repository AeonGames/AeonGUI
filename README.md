AeonGUI [![MinGW64 Build status](https://ci.appveyor.com/api/projects/status/g1hx08cchdmkbw3m?svg=true)](https://ci.appveyor.com/project/Kwizatz/aeongui) [![MinGW32 Build status](https://ci.appveyor.com/api/projects/status/yogupd65ow1dr8pq?svg=true)](https://ci.appveyor.com/project/Kwizatz/aeongui-altq2)
=======

Last Updated: 07-01-2019

This file contains the following sections:

DESCRIPTION
HISTORY
COMPILATION
LICENSE
AUTHOR

DESCRIPTION

Right now AeonGUI is a project to create a graphic user interface library to be used primarily on video games and interactive media.
It's primary goal is to stay agnostic of any advanced/dedicated graphics rendering API such as OpenGL, Vulkan, Direct3D, X11 or GDI.

WARNING!!!: The library is not currently on a stable or even usable state, and is mostly a sandbox for ideas and POCs.

HISTORY

The idea for the library was born around 2004 or 2005 under the name 'Glitch', and has evolved overtime but never quite reached maturity.
The name change to AeonGUI came about as a way to identify the library as part of the AeonGames brand
and to avoid confusion since originaly the name 'GLitch' was chosen to emphasize OpenGL support, which is no longer the priority.

On 2013 the goal of the library was to create a widget toolkit similar to Qt or GTK+ to be rendered inside the contexts of different
graphic APIs, but that idea was dropped due to the complexity required and the fact that the target for the library was video games,
not spreadsheets. The library remained mostly dormant while restart development on the [AeonEngine](https://github.com/AeonGames/AeonEngine)
was going on without the need for a GUI.

Somewhere between 2013 and 2018 the idea of the GUI as an HTML agent lingered, it would have been easy to create UIs for anyone
with CSS and HTML experience, no need to learn a new paradigm, but that idea died as well due to the complexity required plus
the actual similarities between HTML documents and video game UIs being very limited.

Right now, in 2019 the idea is to create a SVG agent that implements a subset of the SVG DOM, making UIs easily implemented via SVG files.
**We really hope this is the last iteration of what AeonGUI IS**

COMPILATION

The library uses CMake to build the required files used to build the binary depending on the platform.
For information on how to use CMake refer to the [CMake official site](https://www.cmake.org).
There is only "official" support for Windows(MSYS2 and Visual Studio) and Linux at this time,
however, Linux builds may break from time to time as most development is done on Windows.

Optional USE variables are available to add support for various features such as freetype font rendering and PNG file format loading,
these add external dependencies to the library, so they are all initially set to OFF, if turned ON, the user has the option to provide
the dependencies or let CMake download and configure the dependencies which will show up as projects inside the build environment.
The rules to download and configure external dependencies are most useful to Windows users as Linux distributions usually provide
development libraries for most software packages.

LICENSE

The library is released under the terms of the permisive [Apache 2.0 license] (http://www.apache.org/licenses/LICENSE-2.0)

The Aeon Games logo is __NOT__ covered by the Apache license,
it is a Trade Mark and may not be used for any purpose without permision.

Some of the code and assets are not covered by the Apache license:

fonts     is Copyright (c) 2003 by Bitstream, Inc., refer to COPYRIGHT.TXT for license terms.

AUTHOR

The only author (for now) of AeonGUI is Rodrigo Hernandez and can be reached at kwizatz at aeongames dot com.
