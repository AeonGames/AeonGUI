# Copyright (C) 2023 Rodrigo Jose Hernandez Cordoba
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

include(functions)

gitclone("https://github.com/netsurf-browser/libparserutils.git" "${CMAKE_CURRENT_SOURCE_DIR}/libparserutils" 96cdd0ff114299f520e76538ab8fde39358b87f9)
gitclone("https://github.com/netsurf-browser/libwapcaplet.git" "${CMAKE_CURRENT_SOURCE_DIR}/libwapcaplet" b5e42b12211a92339b0b62cb90f1a86a397e146e)
gitclone("https://github.com/netsurf-browser/libcss.git" "${CMAKE_CURRENT_SOURCE_DIR}/libcss" 58f1673186a41b396d4ca4e13f43b502a305bf01)

find_package(Python COMPONENTS Interpreter)

add_custom_command(OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/libparserutils/src/charset/aliases.inc
                   COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/make-aliases.py
                   MAIN_DEPENDENCY ${CMAKE_CURRENT_SOURCE_DIR}/libparserutils/build/Aliases
                   WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/libparserutils
                   COMMENT "Generating Aliases")

set(LIBPARSERUTILS_SOURCES
    libparserutils/src/charset/codecs/codec_ascii.c
    libparserutils/src/charset/codecs/codec_8859.c
    libparserutils/src/charset/codecs/codec_ext8.c
	libparserutils/src/charset/codecs/codec_utf8.c
    libparserutils/src/charset/codecs/codec_utf16.c
    libparserutils/src/charset/encodings/utf8.c
    libparserutils/src/charset/encodings/utf16.c
    libparserutils/src/charset/aliases.c
    libparserutils/src/charset/codec.c
    libparserutils/src/charset/aliases.inc
    libparserutils/src/input/filter.c
    libparserutils/src/input/inputstream.c
    libparserutils/src/utils/buffer.c
    libparserutils/src/utils/errors.c
    libparserutils/src/utils/stack.c
    libparserutils/src/utils/vector.c
)

add_library(libparserutils STATIC ${LIBPARSERUTILS_SOURCES})
set_target_properties(libparserutils
                        PROPERTIES
                        PREFIX "")
target_include_directories(libparserutils PUBLIC libparserutils/include PRIVATE libparserutils/src)

option(WITHOUT_ICONV_FILTER "Disable iconv on libparserutils" OFF)
find_package(Iconv)
if(NOT Iconv_FOUND OR WITHOUT_ICONV_FILTER)
  add_compile_definitions(-DWITHOUT_ICONV_FILTER)
else()
    target_link_libraries(libparserutils PRIVATE ${Iconv_LIBRARIES})
    target_include_directories(libparserutils PRIVATE libparserutils/src ${Iconv_INCLUDE_DIRS})
endif()

set(LIBWAPCAPLET_SOURCES
    libwapcaplet/src/libwapcaplet.c
)
add_library(libwapcaplet STATIC ${LIBWAPCAPLET_SOURCES})
set_target_properties(libwapcaplet
                        PROPERTIES
                        PREFIX "")
target_include_directories(libwapcaplet PUBLIC libwapcaplet/include)

set(LIBCSS_SOURCES
    libcss/src/charset/detect.c
    libcss/src/lex/lex.c
    libcss/src/stylesheet.c
    libcss/src/parse/parse.c
    libcss/src/parse/language.c
    libcss/src/parse/important.c
    libcss/src/parse/propstrings.c
    libcss/src/parse/font_face.c
    libcss/src/parse/mq.c
    libcss/src/parse/properties/azimuth.c
    libcss/src/parse/properties/background.c
    libcss/src/parse/properties/background_position.c
    libcss/src/parse/properties/border.c
    libcss/src/parse/properties/border_color.c
    libcss/src/parse/properties/border_spacing.c
    libcss/src/parse/properties/border_style.c
    libcss/src/parse/properties/border_width.c
    libcss/src/parse/properties/clip.c
    libcss/src/parse/properties/columns.c
	libcss/src/parse/properties/column_rule.c
	libcss/src/parse/properties/content.c
	libcss/src/parse/properties/cue.c
	libcss/src/parse/properties/cursor.c
	libcss/src/parse/properties/elevation.c
	libcss/src/parse/properties/flex.c
	libcss/src/parse/properties/flex_flow.c
	libcss/src/parse/properties/font.c
	libcss/src/parse/properties/font_family.c
	libcss/src/parse/properties/font_weight.c
	libcss/src/parse/properties/list_style.c
	libcss/src/parse/properties/list_style_type.c
	libcss/src/parse/properties/margin.c
	libcss/src/parse/properties/opacity.c
	libcss/src/parse/properties/outline.c
	libcss/src/parse/properties/overflow.c
	libcss/src/parse/properties/padding.c
	libcss/src/parse/properties/pause.c
	libcss/src/parse/properties/play_during.c
	libcss/src/parse/properties/properties.c
	libcss/src/parse/properties/quotes.c
	libcss/src/parse/properties/text_decoration.c
	libcss/src/parse/properties/utils.c
	libcss/src/parse/properties/voice_family.c
    libcss/src/select/arena.c
    libcss/src/select/computed.c
    libcss/src/select/dispatch.c
    libcss/src/select/hash.c
    libcss/src/select/select.c
    libcss/src/select/strings.c
    libcss/src/select/font_face.c
    libcss/src/select/format_list_style.c
    libcss/src/select/unit.c
    libcss/src/select/properties/helpers.c
    libcss/src/select/properties/align_content.c
    libcss/src/select/properties/align_items.c
    libcss/src/select/properties/align_self.c
    libcss/src/select/properties/azimuth.c
    libcss/src/select/properties/background_attachment.c
    libcss/src/select/properties/background_color.c
    libcss/src/select/properties/background_image.c
    libcss/src/select/properties/background_position.c
    libcss/src/select/properties/background_repeat.c
    libcss/src/select/properties/border_bottom_color.c
    libcss/src/select/properties/border_bottom_style.c
    libcss/src/select/properties/border_bottom_width.c
    libcss/src/select/properties/border_collapse.c
    libcss/src/select/properties/border_left_color.c
    libcss/src/select/properties/border_left_style.c
    libcss/src/select/properties/border_left_width.c
    libcss/src/select/properties/border_right_color.c
    libcss/src/select/properties/border_right_style.c
    libcss/src/select/properties/border_right_width.c
    libcss/src/select/properties/border_spacing.c
    libcss/src/select/properties/border_top_color.c
    libcss/src/select/properties/border_top_style.c
    libcss/src/select/properties/border_top_width.c
    libcss/src/select/properties/bottom.c
    libcss/src/select/properties/box_sizing.c
    libcss/src/select/properties/break_after.c
    libcss/src/select/properties/break_before.c
    libcss/src/select/properties/break_inside.c
    libcss/src/select/properties/caption_side.c
    libcss/src/select/properties/clear.c
    libcss/src/select/properties/clip.c
    libcss/src/select/properties/color.c
    libcss/src/select/properties/column_count.c
    libcss/src/select/properties/column_fill.c
    libcss/src/select/properties/column_gap.c
    libcss/src/select/properties/column_rule_color.c
    libcss/src/select/properties/column_rule_style.c
    libcss/src/select/properties/column_rule_width.c
    libcss/src/select/properties/column_span.c
    libcss/src/select/properties/column_width.c
    libcss/src/select/properties/content.c
    libcss/src/select/properties/counter_increment.c
    libcss/src/select/properties/counter_reset.c
    libcss/src/select/properties/cue_after.c
    libcss/src/select/properties/cue_before.c
    libcss/src/select/properties/cursor.c
    libcss/src/select/properties/direction.c
    libcss/src/select/properties/display.c
    libcss/src/select/properties/elevation.c
    libcss/src/select/properties/empty_cells.c
    libcss/src/select/properties/flex_basis.c
    libcss/src/select/properties/flex_direction.c
    libcss/src/select/properties/flex_grow.c
    libcss/src/select/properties/flex_shrink.c
    libcss/src/select/properties/flex_wrap.c
    libcss/src/select/properties/float.c
    libcss/src/select/properties/font_family.c
    libcss/src/select/properties/font_size.c
    libcss/src/select/properties/font_style.c
    libcss/src/select/properties/font_variant.c
    libcss/src/select/properties/font_weight.c
    libcss/src/select/properties/height.c
    libcss/src/select/properties/justify_content.c
    libcss/src/select/properties/left.c
    libcss/src/select/properties/letter_spacing.c
    libcss/src/select/properties/line_height.c
    libcss/src/select/properties/list_style_image.c
    libcss/src/select/properties/list_style_position.c
    libcss/src/select/properties/list_style_type.c
    libcss/src/select/properties/margin_bottom.c
    libcss/src/select/properties/margin_left.c
    libcss/src/select/properties/margin_right.c
    libcss/src/select/properties/margin_top.c
    libcss/src/select/properties/max_height.c
    libcss/src/select/properties/max_width.c
    libcss/src/select/properties/min_height.c
    libcss/src/select/properties/min_width.c
    libcss/src/select/properties/opacity.c
    libcss/src/select/properties/order.c
    libcss/src/select/properties/orphans.c
    libcss/src/select/properties/outline_color.c
    libcss/src/select/properties/outline_style.c
    libcss/src/select/properties/outline_width.c
    libcss/src/select/properties/overflow_x.c
    libcss/src/select/properties/overflow_y.c
    libcss/src/select/properties/padding_bottom.c
    libcss/src/select/properties/padding_left.c
    libcss/src/select/properties/padding_right.c
    libcss/src/select/properties/padding_top.c
    libcss/src/select/properties/page_break_after.c
    libcss/src/select/properties/page_break_before.c
    libcss/src/select/properties/page_break_inside.c
    libcss/src/select/properties/pause_after.c
    libcss/src/select/properties/pause_before.c
    libcss/src/select/properties/pitch.c
    libcss/src/select/properties/pitch_range.c
    libcss/src/select/properties/play_during.c
    libcss/src/select/properties/position.c
    libcss/src/select/properties/quotes.c
    libcss/src/select/properties/richness.c
    libcss/src/select/properties/right.c
    libcss/src/select/properties/speech_rate.c
    libcss/src/select/properties/speak.c
    libcss/src/select/properties/speak_header.c
    libcss/src/select/properties/speak_numeral.c
    libcss/src/select/properties/speak_punctuation.c
    libcss/src/select/properties/stress.c
    libcss/src/select/properties/table_layout.c
    libcss/src/select/properties/text_align.c
    libcss/src/select/properties/text_decoration.c
    libcss/src/select/properties/text_indent.c
    libcss/src/select/properties/text_transform.c
    libcss/src/select/properties/top.c
    libcss/src/select/properties/unicode_bidi.c
    libcss/src/select/properties/vertical_align.c
    libcss/src/select/properties/visibility.c
    libcss/src/select/properties/voice_family.c
    libcss/src/select/properties/volume.c
    libcss/src/select/properties/white_space.c
    libcss/src/select/properties/widows.c
    libcss/src/select/properties/width.c
    libcss/src/select/properties/word_spacing.c
    libcss/src/select/properties/writing_mode.c
    libcss/src/select/properties/z_index.c
    libcss/src/utils/errors.c
    libcss/src/utils/utils.c
)

add_executable(gen_parser libcss/src/parse/properties/css_property_parser_gen.c)

file(STRINGS libcss/src/parse/properties/properties.gen PROPS REGEX "^[^#][^:]+:")

foreach(PROP IN LISTS PROPS)
    string(REGEX MATCH  "^[^:]+" PROP_NAME ${PROP})
    add_custom_command(OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/libcss/src/parse/properties/autogenerated_${PROP_NAME}.c
    COMMAND gen_parser -o ${CMAKE_CURRENT_SOURCE_DIR}/libcss/src/parse/properties/autogenerated_${PROP_NAME}.c ${PROP}
    MAIN_DEPENDENCY ${CMAKE_CURRENT_SOURCE_DIR}/libcss/src/parse/properties/properties.gen
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/libcss/src/parse/properties
    COMMENT "Generating ${PROP_NAME} source" VERBATIM)
    list(APPEND LIBCSS_AUTOGENERATED_SOURCES libcss/src/parse/properties/autogenerated_${PROP_NAME}.c)
endforeach()

add_library(libcss STATIC ${LIBCSS_SOURCES} ${LIBCSS_AUTOGENERATED_SOURCES})
set_target_properties(libcss PROPERTIES PREFIX "")
target_include_directories(libcss PUBLIC libcss/include libparserutils/include libwapcaplet/include PRIVATE libcss/src)
target_link_libraries(libcss PUBLIC libparserutils libwapcaplet)
if(MSVC)
    file(TOUCH libcss/include/strings.h)
    set_target_properties(libcss PROPERTIES COMPILE_FLAGS "/D_CRT_SECURE_NO_WARNINGS /Drestrict= /Dstrncasecmp=_strnicmp /Dstrdup=_strdup /wd4244 /wd4267 /wd4133 /wd4018")
    set_target_properties(gen_parser PROPERTIES COMPILE_FLAGS "/D_CRT_SECURE_NO_WARNINGS /Dstrdup=_strdup /wd4244 /wd4267 /wd4101")
    set_target_properties(libparserutils PROPERTIES COMPILE_FLAGS "/wd4267")
elseif(LINUX)
    set_target_properties(libcss PROPERTIES COMPILE_FLAGS "-fPIC")
    set_target_properties(gen_parser PROPERTIES COMPILE_FLAGS "-fPIC")
    set_target_properties(libparserutils PROPERTIES COMPILE_FLAGS "-fPIC")
endif()
