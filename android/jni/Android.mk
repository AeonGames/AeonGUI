LOCAL_PATH := $(call my-dir)/../..

include $(CLEAR_VARS)

LOCAL_MODULE    := 	core
LOCAL_SRC_FILES := 	core/Button.cpp  \
                    core/Font.cpp  \
                    core/MainWindow.cpp \
                    core/Renderer.cpp \
                    core/Widget.cpp \
                    core/Color.cpp \
                    core/Image.cpp \
                    core/Rect.cpp  \
                    core/ScrollBar.cpp \
                    common/pcx/pcx.cpp \
                    core/resources/close.c \
                    core/resources/minimize.c \
                    core/resources/scrolldown.c \
                    core/resources/scrollright.c \
                    core/resources/close_down.c \
                    core/resources/minimize_down.c \
                    core/resources/scrolldownpressed.c \
                    core/resources/scrollrightpressed.c \
                    core/resources/maximize.c \
                    core/resources/restore.c \
                    core/resources/scrollleft.c \
                    core/resources/scrollup.c \
                    core/resources/maximize_down.c \
                    core/resources/restore_down.c \
                    core/resources/scrollleftpressed.c \
                    core/resources/scrolluppressed.c

LOCAL_CFLAGS += -I$(LOCAL_PATH)/include -I$(LOCAL_PATH)/common -I$(LOCAL_PATH)/common/pcx -I$(LOCAL_PATH)/core/resources
LOCAL_LDLIBS := -ldl -llog

include $(BUILD_SHARED_LIBRARY)

#include $(CLEAR_VARS)
				
#LOCAL_MODULE    := 	game
#LOCAL_SRC_FILES := 	demo/Player.cpp \
					demo/Game.cpp
					
#LOCAL_LDLIBS := -llog

#LOCAL_SHARED_LIBRARIES := engine

#include $(BUILD_SHARED_LIBRARY)
