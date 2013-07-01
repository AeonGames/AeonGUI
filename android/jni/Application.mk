APP_CFLAGS += -Wno-psabi -DUSE_JAVA -I../include -I../engine
APP_CPPFLAGS += -g -std=c++0x -D_DEBUG
#APP_STL := system
#APP_STL := stlport_static
APP_STL := gnustl_static
