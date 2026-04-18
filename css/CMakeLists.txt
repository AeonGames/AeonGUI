# Copyright (C) 2023,2024 Rodrigo Jose Hernandez Cordoba
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

include(FetchContent)

FetchContent_Declare(libparserutils
    GIT_REPOSITORY https://github.com/netsurf-browser/libparserutils.git
    GIT_TAG        6b0cbf086ca8eb8fe74b69f0c9ecf274eb2397ca
    GIT_SHALLOW    ON
)
FetchContent_Declare(libwapcaplet
    GIT_REPOSITORY https://github.com/netsurf-browser/libwapcaplet.git
    GIT_TAG        c7c128d3eb3223b216c974471f82e9337fbcf4ba
    GIT_SHALLOW    ON
)
FetchContent_Declare(libcss
    GIT_REPOSITORY https://github.com/AeonGames/libcss.git
    GIT_TAG        aeongui
    GIT_SHALLOW    ON
)

FetchContent_MakeAvailable(libparserutils libwapcaplet libcss)

find_package(Python COMPONENTS Interpreter)

add_custom_command(OUTPUT ${libparserutils_SOURCE_DIR}/src/charset/aliases.inc
                   COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/make-aliases.py
                   MAIN_DEPENDENCY ${libparserutils_SOURCE_DIR}/build/Aliases
                   WORKING_DIRECTORY ${libparserutils_SOURCE_DIR}
                   COMMENT "Generating Aliases")

set(LIBPARSERUTILS_SOURCES
    ${libparserutils_SOURCE_DIR}/src/charset/codecs/codec_ascii.c
    ${libparserutils_SOURCE_DIR}/src/charset/codecs/codec_8859.c
    ${libparserutils_SOURCE_DIR}/src/charset/codecs/codec_ext8.c
	${libparserutils_SOURCE_DIR}/src/charset/codecs/codec_utf8.c
    ${libparserutils_SOURCE_DIR}/src/charset/codecs/codec_utf16.c
    ${libparserutils_SOURCE_DIR}/src/charset/encodings/utf8.c
    ${libparserutils_SOURCE_DIR}/src/charset/encodings/utf16.c
    ${libparserutils_SOURCE_DIR}/src/charset/aliases.c
    ${libparserutils_SOURCE_DIR}/src/charset/codec.c
    ${libparserutils_SOURCE_DIR}/src/charset/aliases.inc
    ${libparserutils_SOURCE_DIR}/src/input/filter.c
    ${libparserutils_SOURCE_DIR}/src/input/inputstream.c
    ${libparserutils_SOURCE_DIR}/src/utils/buffer.c
    ${libparserutils_SOURCE_DIR}/src/utils/errors.c
    ${libparserutils_SOURCE_DIR}/src/utils/stack.c
    ${libparserutils_SOURCE_DIR}/src/utils/vector.c
)

add_library(libparserutils STATIC ${LIBPARSERUTILS_SOURCES})
set_target_properties(libparserutils
                        PROPERTIES
                        PREFIX "")
target_include_directories(libparserutils PUBLIC ${libparserutils_SOURCE_DIR}/include PRIVATE ${libparserutils_SOURCE_DIR}/src)

option(WITHOUT_ICONV_FILTER "Disable iconv on libparserutils" OFF)
find_package(Iconv)
if(NOT Iconv_FOUND OR WITHOUT_ICONV_FILTER)
  add_compile_definitions(-DWITHOUT_ICONV_FILTER)
else()
    target_link_libraries(libparserutils PRIVATE ${Iconv_LIBRARIES})
    target_include_directories(libparserutils PRIVATE ${libparserutils_SOURCE_DIR}/src ${Iconv_INCLUDE_DIRS})
endif()

set(LIBWAPCAPLET_SOURCES
    ${libwapcaplet_SOURCE_DIR}/src/libwapcaplet.c
)
add_library(libwapcaplet STATIC ${LIBWAPCAPLET_SOURCES})
set_target_properties(libwapcaplet
                        PROPERTIES
                        PREFIX "")
target_include_directories(libwapcaplet PUBLIC ${libwapcaplet_SOURCE_DIR}/include)

set(LIBCSS_SOURCES
    ${libcss_SOURCE_DIR}/src/charset/detect.c
    ${libcss_SOURCE_DIR}/src/lex/lex.c
    ${libcss_SOURCE_DIR}/src/stylesheet.c
    ${libcss_SOURCE_DIR}/src/parse/parse.c
    ${libcss_SOURCE_DIR}/src/parse/language.c
    ${libcss_SOURCE_DIR}/src/parse/important.c
    ${libcss_SOURCE_DIR}/src/parse/propstrings.c
    ${libcss_SOURCE_DIR}/src/parse/font_face.c
    ${libcss_SOURCE_DIR}/src/parse/mq.c
    ${libcss_SOURCE_DIR}/src/parse/properties/azimuth.c
    ${libcss_SOURCE_DIR}/src/parse/properties/background.c
    ${libcss_SOURCE_DIR}/src/parse/properties/background_position.c
    ${libcss_SOURCE_DIR}/src/parse/properties/border.c
    ${libcss_SOURCE_DIR}/src/parse/properties/border_color.c
    ${libcss_SOURCE_DIR}/src/parse/properties/border_spacing.c
    ${libcss_SOURCE_DIR}/src/parse/properties/border_style.c
    ${libcss_SOURCE_DIR}/src/parse/properties/border_width.c
    ${libcss_SOURCE_DIR}/src/parse/properties/clip.c
    ${libcss_SOURCE_DIR}/src/parse/properties/columns.c
	${libcss_SOURCE_DIR}/src/parse/properties/column_rule.c
	${libcss_SOURCE_DIR}/src/parse/properties/content.c
	${libcss_SOURCE_DIR}/src/parse/properties/cue.c
	${libcss_SOURCE_DIR}/src/parse/properties/cursor.c
	${libcss_SOURCE_DIR}/src/parse/properties/elevation.c
	${libcss_SOURCE_DIR}/src/parse/properties/fill_opacity.c
	${libcss_SOURCE_DIR}/src/parse/properties/flex.c
	${libcss_SOURCE_DIR}/src/parse/properties/flex_flow.c
	${libcss_SOURCE_DIR}/src/parse/properties/font.c
	${libcss_SOURCE_DIR}/src/parse/properties/font_family.c
	${libcss_SOURCE_DIR}/src/parse/properties/font_weight.c
	${libcss_SOURCE_DIR}/src/parse/properties/list_style.c
	${libcss_SOURCE_DIR}/src/parse/properties/list_style_type.c
	${libcss_SOURCE_DIR}/src/parse/properties/margin.c
	${libcss_SOURCE_DIR}/src/parse/properties/opacity.c
	${libcss_SOURCE_DIR}/src/parse/properties/outline.c
	${libcss_SOURCE_DIR}/src/parse/properties/overflow.c
	${libcss_SOURCE_DIR}/src/parse/properties/padding.c
	${libcss_SOURCE_DIR}/src/parse/properties/pause.c
	${libcss_SOURCE_DIR}/src/parse/properties/play_during.c
	${libcss_SOURCE_DIR}/src/parse/properties/properties.c
	${libcss_SOURCE_DIR}/src/parse/properties/quotes.c
	${libcss_SOURCE_DIR}/src/parse/properties/stroke_opacity.c
	${libcss_SOURCE_DIR}/src/parse/properties/text_decoration.c
	${libcss_SOURCE_DIR}/src/parse/properties/utils.c
	${libcss_SOURCE_DIR}/src/parse/properties/voice_family.c
    ${libcss_SOURCE_DIR}/src/select/arena.c
    ${libcss_SOURCE_DIR}/src/select/calc.c
    ${libcss_SOURCE_DIR}/src/select/computed.c
    ${libcss_SOURCE_DIR}/src/select/dispatch.c
    ${libcss_SOURCE_DIR}/src/select/hash.c
    ${libcss_SOURCE_DIR}/src/select/select.c
    ${libcss_SOURCE_DIR}/src/select/strings.c
    ${libcss_SOURCE_DIR}/src/select/font_face.c
    ${libcss_SOURCE_DIR}/src/select/format_list_style.c
    ${libcss_SOURCE_DIR}/src/select/unit.c
    ${libcss_SOURCE_DIR}/src/select/properties/helpers.c
    ${libcss_SOURCE_DIR}/src/select/properties/align_content.c
    ${libcss_SOURCE_DIR}/src/select/properties/align_items.c
    ${libcss_SOURCE_DIR}/src/select/properties/align_self.c
    ${libcss_SOURCE_DIR}/src/select/properties/azimuth.c
    ${libcss_SOURCE_DIR}/src/select/properties/background_attachment.c
    ${libcss_SOURCE_DIR}/src/select/properties/background_color.c
    ${libcss_SOURCE_DIR}/src/select/properties/background_image.c
    ${libcss_SOURCE_DIR}/src/select/properties/background_position.c
    ${libcss_SOURCE_DIR}/src/select/properties/background_repeat.c
    ${libcss_SOURCE_DIR}/src/select/properties/border_bottom_color.c
    ${libcss_SOURCE_DIR}/src/select/properties/border_bottom_style.c
    ${libcss_SOURCE_DIR}/src/select/properties/border_bottom_width.c
    ${libcss_SOURCE_DIR}/src/select/properties/border_collapse.c
    ${libcss_SOURCE_DIR}/src/select/properties/border_left_color.c
    ${libcss_SOURCE_DIR}/src/select/properties/border_left_style.c
    ${libcss_SOURCE_DIR}/src/select/properties/border_left_width.c
    ${libcss_SOURCE_DIR}/src/select/properties/border_right_color.c
    ${libcss_SOURCE_DIR}/src/select/properties/border_right_style.c
    ${libcss_SOURCE_DIR}/src/select/properties/border_right_width.c
    ${libcss_SOURCE_DIR}/src/select/properties/border_spacing.c
    ${libcss_SOURCE_DIR}/src/select/properties/border_top_color.c
    ${libcss_SOURCE_DIR}/src/select/properties/border_top_style.c
    ${libcss_SOURCE_DIR}/src/select/properties/border_top_width.c
    ${libcss_SOURCE_DIR}/src/select/properties/bottom.c
    ${libcss_SOURCE_DIR}/src/select/properties/box_sizing.c
    ${libcss_SOURCE_DIR}/src/select/properties/break_after.c
    ${libcss_SOURCE_DIR}/src/select/properties/break_before.c
    ${libcss_SOURCE_DIR}/src/select/properties/break_inside.c
    ${libcss_SOURCE_DIR}/src/select/properties/caption_side.c
    ${libcss_SOURCE_DIR}/src/select/properties/clear.c
    ${libcss_SOURCE_DIR}/src/select/properties/clip.c
    ${libcss_SOURCE_DIR}/src/select/properties/color.c
    ${libcss_SOURCE_DIR}/src/select/properties/column_count.c
    ${libcss_SOURCE_DIR}/src/select/properties/column_fill.c
    ${libcss_SOURCE_DIR}/src/select/properties/column_gap.c
    ${libcss_SOURCE_DIR}/src/select/properties/column_rule_color.c
    ${libcss_SOURCE_DIR}/src/select/properties/column_rule_style.c
    ${libcss_SOURCE_DIR}/src/select/properties/column_rule_width.c
    ${libcss_SOURCE_DIR}/src/select/properties/column_span.c
    ${libcss_SOURCE_DIR}/src/select/properties/column_width.c
    ${libcss_SOURCE_DIR}/src/select/properties/content.c
    ${libcss_SOURCE_DIR}/src/select/properties/counter_increment.c
    ${libcss_SOURCE_DIR}/src/select/properties/counter_reset.c
    ${libcss_SOURCE_DIR}/src/select/properties/cue_after.c
    ${libcss_SOURCE_DIR}/src/select/properties/cue_before.c
    ${libcss_SOURCE_DIR}/src/select/properties/cursor.c
    ${libcss_SOURCE_DIR}/src/select/properties/direction.c
    ${libcss_SOURCE_DIR}/src/select/properties/display.c
    ${libcss_SOURCE_DIR}/src/select/properties/elevation.c
    ${libcss_SOURCE_DIR}/src/select/properties/empty_cells.c
    ${libcss_SOURCE_DIR}/src/select/properties/fill.c
    ${libcss_SOURCE_DIR}/src/select/properties/fill_opacity.c
    ${libcss_SOURCE_DIR}/src/select/properties/flex_basis.c
    ${libcss_SOURCE_DIR}/src/select/properties/flex_direction.c
    ${libcss_SOURCE_DIR}/src/select/properties/flex_grow.c
    ${libcss_SOURCE_DIR}/src/select/properties/flex_shrink.c
    ${libcss_SOURCE_DIR}/src/select/properties/flex_wrap.c
    ${libcss_SOURCE_DIR}/src/select/properties/float.c
    ${libcss_SOURCE_DIR}/src/select/properties/font_family.c
    ${libcss_SOURCE_DIR}/src/select/properties/font_size.c
    ${libcss_SOURCE_DIR}/src/select/properties/font_style.c
    ${libcss_SOURCE_DIR}/src/select/properties/font_variant.c
    ${libcss_SOURCE_DIR}/src/select/properties/font_weight.c
    ${libcss_SOURCE_DIR}/src/select/properties/height.c
    ${libcss_SOURCE_DIR}/src/select/properties/justify_content.c
    ${libcss_SOURCE_DIR}/src/select/properties/left.c
    ${libcss_SOURCE_DIR}/src/select/properties/letter_spacing.c
    ${libcss_SOURCE_DIR}/src/select/properties/line_height.c
    ${libcss_SOURCE_DIR}/src/select/properties/list_style_image.c
    ${libcss_SOURCE_DIR}/src/select/properties/list_style_position.c
    ${libcss_SOURCE_DIR}/src/select/properties/list_style_type.c
    ${libcss_SOURCE_DIR}/src/select/properties/margin_bottom.c
    ${libcss_SOURCE_DIR}/src/select/properties/margin_left.c
    ${libcss_SOURCE_DIR}/src/select/properties/margin_right.c
    ${libcss_SOURCE_DIR}/src/select/properties/margin_top.c
    ${libcss_SOURCE_DIR}/src/select/properties/max_height.c
    ${libcss_SOURCE_DIR}/src/select/properties/max_width.c
    ${libcss_SOURCE_DIR}/src/select/properties/min_height.c
    ${libcss_SOURCE_DIR}/src/select/properties/min_width.c
    ${libcss_SOURCE_DIR}/src/select/properties/opacity.c
    ${libcss_SOURCE_DIR}/src/select/properties/order.c
    ${libcss_SOURCE_DIR}/src/select/properties/orphans.c
    ${libcss_SOURCE_DIR}/src/select/properties/outline_color.c
    ${libcss_SOURCE_DIR}/src/select/properties/outline_style.c
    ${libcss_SOURCE_DIR}/src/select/properties/outline_width.c
    ${libcss_SOURCE_DIR}/src/select/properties/overflow_x.c
    ${libcss_SOURCE_DIR}/src/select/properties/overflow_y.c
    ${libcss_SOURCE_DIR}/src/select/properties/padding_bottom.c
    ${libcss_SOURCE_DIR}/src/select/properties/padding_left.c
    ${libcss_SOURCE_DIR}/src/select/properties/padding_right.c
    ${libcss_SOURCE_DIR}/src/select/properties/padding_top.c
    ${libcss_SOURCE_DIR}/src/select/properties/page_break_after.c
    ${libcss_SOURCE_DIR}/src/select/properties/page_break_before.c
    ${libcss_SOURCE_DIR}/src/select/properties/page_break_inside.c
    ${libcss_SOURCE_DIR}/src/select/properties/pause_after.c
    ${libcss_SOURCE_DIR}/src/select/properties/pause_before.c
    ${libcss_SOURCE_DIR}/src/select/properties/pitch.c
    ${libcss_SOURCE_DIR}/src/select/properties/pitch_range.c
    ${libcss_SOURCE_DIR}/src/select/properties/play_during.c
    ${libcss_SOURCE_DIR}/src/select/properties/position.c
    ${libcss_SOURCE_DIR}/src/select/properties/quotes.c
    ${libcss_SOURCE_DIR}/src/select/properties/richness.c
    ${libcss_SOURCE_DIR}/src/select/properties/right.c
    ${libcss_SOURCE_DIR}/src/select/properties/speech_rate.c
    ${libcss_SOURCE_DIR}/src/select/properties/speak.c
    ${libcss_SOURCE_DIR}/src/select/properties/speak_header.c
    ${libcss_SOURCE_DIR}/src/select/properties/speak_numeral.c
    ${libcss_SOURCE_DIR}/src/select/properties/speak_punctuation.c
    ${libcss_SOURCE_DIR}/src/select/properties/stress.c
    ${libcss_SOURCE_DIR}/src/select/properties/stroke.c
    ${libcss_SOURCE_DIR}/src/select/properties/stroke_opacity.c
    ${libcss_SOURCE_DIR}/src/select/properties/stroke_width.c
    ${libcss_SOURCE_DIR}/src/select/properties/table_layout.c
    ${libcss_SOURCE_DIR}/src/select/properties/text_align.c
    ${libcss_SOURCE_DIR}/src/select/properties/text_anchor.c
    ${libcss_SOURCE_DIR}/src/select/properties/text_decoration.c
    ${libcss_SOURCE_DIR}/src/select/properties/text_indent.c
    ${libcss_SOURCE_DIR}/src/select/properties/text_transform.c
    ${libcss_SOURCE_DIR}/src/select/properties/top.c
    ${libcss_SOURCE_DIR}/src/select/properties/transform.c
    ${libcss_SOURCE_DIR}/src/select/properties/unicode_bidi.c
    ${libcss_SOURCE_DIR}/src/select/properties/vertical_align.c
    ${libcss_SOURCE_DIR}/src/select/properties/visibility.c
    ${libcss_SOURCE_DIR}/src/select/properties/voice_family.c
    ${libcss_SOURCE_DIR}/src/select/properties/volume.c
    ${libcss_SOURCE_DIR}/src/select/properties/white_space.c
    ${libcss_SOURCE_DIR}/src/select/properties/widows.c
    ${libcss_SOURCE_DIR}/src/select/properties/width.c
    ${libcss_SOURCE_DIR}/src/select/properties/word_spacing.c
    ${libcss_SOURCE_DIR}/src/select/properties/writing_mode.c
    ${libcss_SOURCE_DIR}/src/select/properties/z_index.c
    ${libcss_SOURCE_DIR}/src/utils/errors.c
    ${libcss_SOURCE_DIR}/src/utils/utils.c
)

add_executable(gen_parser ${libcss_SOURCE_DIR}/src/parse/properties/css_property_parser_gen.c)

set(LIBCSS_AUTOGEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/libcss_autogenerated")
file(MAKE_DIRECTORY "${LIBCSS_AUTOGEN_DIR}")

file(STRINGS ${libcss_SOURCE_DIR}/src/parse/properties/properties.gen PROPS REGEX "^[^#][^:]+:")
foreach(PROP IN LISTS PROPS)
    string(REGEX MATCH  "^[^:]+" PROP_NAME ${PROP})
    add_custom_command(OUTPUT ${LIBCSS_AUTOGEN_DIR}/autogenerated_${PROP_NAME}.c
    COMMAND gen_parser -o ${LIBCSS_AUTOGEN_DIR}/autogenerated_${PROP_NAME}.c ${PROP}
    MAIN_DEPENDENCY ${libcss_SOURCE_DIR}/src/parse/properties/properties.gen
    WORKING_DIRECTORY ${libcss_SOURCE_DIR}/src/parse/properties
    COMMENT "Generating ${PROP_NAME} source" VERBATIM)
    list(APPEND LIBCSS_AUTOGENERATED_SOURCES ${LIBCSS_AUTOGEN_DIR}/autogenerated_${PROP_NAME}.c)
endforeach()

add_library(libcss STATIC ${LIBCSS_SOURCES} ${LIBCSS_AUTOGENERATED_SOURCES})
set_target_properties(libcss PROPERTIES PREFIX "")
target_include_directories(libcss PUBLIC ${libcss_SOURCE_DIR}/include ${libparserutils_SOURCE_DIR}/include ${libwapcaplet_SOURCE_DIR}/include PRIVATE ${libcss_SOURCE_DIR}/src)
target_link_libraries(libcss PUBLIC libparserutils libwapcaplet)
if(MSVC)
    # MSVC lacks <strings.h>; create an empty shim in the build tree so
    # libcss sources that #include <strings.h> compile without error.
    # The actual functions (strncasecmp, strdup) are mapped via /D flags below.
    set(LIBCSS_SHIM_DIR "${CMAKE_CURRENT_BINARY_DIR}/libcss_shims")
    file(MAKE_DIRECTORY "${LIBCSS_SHIM_DIR}")
    file(WRITE "${LIBCSS_SHIM_DIR}/strings.h" "/* MSVC shim: intentionally empty */\n")
    target_include_directories(libcss PRIVATE "${LIBCSS_SHIM_DIR}")
    set_target_properties(libcss PROPERTIES COMPILE_FLAGS "/D_CRT_SECURE_NO_WARNINGS /Drestrict= /Dstrncasecmp=_strnicmp /Dstrdup=_strdup /wd4244 /wd4267 /wd4133 /wd4018")
    set_target_properties(gen_parser PROPERTIES COMPILE_FLAGS "/D_CRT_SECURE_NO_WARNINGS /Dstrdup=_strdup /wd4244 /wd4267 /wd4101")
    set_target_properties(libparserutils PROPERTIES COMPILE_FLAGS "/wd4267")
elseif(LINUX)
    set_target_properties(libcss PROPERTIES COMPILE_FLAGS "-fPIC")
    set_target_properties(gen_parser PROPERTIES COMPILE_FLAGS "-fPIC")
    set_target_properties(libparserutils PROPERTIES COMPILE_FLAGS "-fPIC")
endif()
