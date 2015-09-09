if(WIN32)
	set(PLATFORM_SOURCES
		win32/agg_platform_support.cpp
		win32/agg_win32_bmp.cpp
	)
endif()
add_library(aggplatform STATIC ${PLATFORM_SOURCES})