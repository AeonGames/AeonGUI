if(WIN32)
	add_definitions(-D_CRT_NONSTDC_NO_DEPRECATE)
	set(PLATFORM_SOURCES
		win32/agg_platform_support.cpp
		win32/agg_win32_bmp.cpp
	)
endif()
add_library(aggplatform STATIC ${PLATFORM_SOURCES})
set_target_properties(aggplatform PROPERTIES FOLDER "AGG")
