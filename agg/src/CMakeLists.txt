add_definitions(-D_CRT_SECURE_NO_WARNINGS)
subdirs(platform)
include_directories(../clipper ${FREETYPE_INCLUDE_DIRS} ../font_freetype ../font_win32_tt)
set(AGG_SOURCES
	agg_arc.cpp
	agg_arrowhead.cpp
	agg_bezier_arc.cpp
	agg_bspline.cpp
	agg_curves.cpp
	agg_vcgen_contour.cpp
	agg_vcgen_dash.cpp
	agg_vcgen_markers_term.cpp
	agg_vcgen_smooth_poly1.cpp
	agg_vcgen_stroke.cpp
	agg_vcgen_bspline.cpp
	agg_gsv_text.cpp
	agg_image_filters.cpp
	agg_line_aa_basics.cpp
	agg_line_profile_aa.cpp
	agg_rounded_rect.cpp
	agg_sqrt_tables.cpp
	agg_embedded_raster_fonts.cpp
	agg_trans_affine.cpp
	agg_trans_warp_magnifier.cpp
	agg_trans_single_path.cpp
	agg_trans_double_path.cpp
	agg_vpgen_clip_polygon.cpp
	agg_vpgen_clip_polyline.cpp
	agg_vpgen_segmentator.cpp
	)
set(AGG_CTRL_SOURCES
	ctrl/agg_cbox_ctrl.cpp
	ctrl/agg_gamma_ctrl.cpp
	ctrl/agg_gamma_spline.cpp
	ctrl/agg_rbox_ctrl.cpp
	ctrl/agg_slider_ctrl.cpp
	ctrl/agg_spline_ctrl.cpp
	ctrl/agg_scale_ctrl.cpp
	ctrl/agg_polygon_ctrl.cpp
	ctrl/agg_bezier_ctrl.cpp
	)
	
set(CLIPPER_SOURCES
	../clipper/clipper.hpp
	../clipper/clipper.cpp
	)
	
set(AGG_HEADERS
	../include/agg_alpha_mask_u8.h          ../include/agg_glyph_raster_bin.h          ../include/agg_span_allocator.h
	../include/agg_arc.h                    ../include/agg_gsv_text.h                  ../include/agg_span_converter.h
	../include/agg_array.h                  ../include/agg_image_accessors.h           ../include/agg_span_gouraud.h
	../include/agg_arrowhead.h              ../include/agg_image_filters.h             ../include/agg_span_gouraud_gray.h
	../include/agg_basics.h                 ../include/agg_line_aa_basics.h            ../include/agg_span_gouraud_rgba.h
	../include/agg_bezier_arc.h             ../include/agg_math.h                      ../include/agg_span_gradient.h
	../include/agg_bitset_iterator.h        ../include/agg_blur.h                      ../include/agg_math_stroke.h
	../include/agg_span_gradient_alpha.h    ../include/agg_gradient_lut.h
	../include/agg_bounding_rect.h          ../include/agg_path_length.h               ../include/agg_span_image_filter.h
	../include/agg_bspline.h                ../include/agg_path_storage.h              ../include/agg_span_image_filter_gray.h
	../include/agg_clip_liang_barsky.h      ../include/agg_path_storage_integer.h      ../include/agg_span_image_filter_rgb.h
	../include/agg_color_gray.h             ../include/agg_pattern_filters_rgba.h      ../include/agg_span_image_filter_rgba.h
	../include/agg_color_rgba.h             ../include/agg_pixfmt_amask_adaptor.h      ../include/agg_span_interpolator_adaptor.h
	../include/agg_config.h                 ../include/agg_pixfmt_gray.h               ../include/agg_span_interpolator_linear.h
	../include/agg_conv_adaptor_vcgen.h     ../include/agg_pixfmt_rgb.h                ../include/agg_span_interpolator_persp.h
	../include/agg_conv_adaptor_vpgen.h     ../include/agg_pixfmt_rgb_packed.h         ../include/agg_span_interpolator_trans.h
	../include/agg_conv_bspline.h           ../include/agg_pixfmt_rgba.h               ../include/agg_pixfmt_transposer.h
	../include/agg_span_pattern_gray.h
	../include/agg_conv_clip_polygon.h      ../include/agg_rasterizer_cells_aa.h       ../include/agg_span_pattern_rgb.h
	../include/agg_conv_clip_polyline.h     ../include/agg_rasterizer_compound_aa.h    ../include/agg_span_pattern_rgba.h
	../include/agg_conv_close_polygon.h     ../include/agg_rasterizer_outline.h        ../include/agg_span_solid.h
	../include/agg_conv_concat.h            ../include/agg_rasterizer_outline_aa.h     ../include/agg_span_subdiv_adaptor.h
	../include/agg_conv_contour.h           ../include/agg_rasterizer_scanline_aa.h    ../include/agg_trans_affine.h
	../include/agg_conv_curve.h             ../include/agg_rasterizer_sl_clip.h        ../include/agg_trans_bilinear.h
	../include/agg_conv_dash.h              ../include/agg_renderer_base.h             ../include/agg_trans_double_path.h
	../include/agg_renderer_markers.h
	../include/agg_conv_marker.h            ../include/agg_renderer_mclip.h            ../include/agg_trans_perspective.h
	../include/agg_conv_marker_adaptor.h    ../include/agg_renderer_outline_aa.h       ../include/agg_trans_single_path.h
	../include/agg_conv_segmentator.h       ../include/agg_renderer_outline_image.h    ../include/agg_trans_viewport.h
	../include/agg_conv_shorten_path.h      ../include/agg_renderer_primitives.h       ../include/agg_trans_warp_magnifier.h
	../include/agg_conv_smooth_poly1.h      ../include/agg_renderer_raster_text.h      ../include/agg_vcgen_bspline.h
	../include/agg_conv_stroke.h            ../include/agg_renderer_scanline.h         ../include/agg_vcgen_contour.h
	../include/agg_conv_transform.h         ../include/agg_rendering_buffer.h          ../include/agg_vcgen_dash.h
	../include/agg_conv_unclose_polygon.h   ../include/agg_rendering_buffer_dynarow.h  ../include/agg_vcgen_markers_term.h
	../include/agg_curves.h                 ../include/agg_rounded_rect.h              ../include/agg_vcgen_smooth_poly1.h
	../include/agg_scanline_bin.h           ../include/agg_vcgen_stroke.h
	../include/agg_dda_line.h               ../include/agg_scanline_boolean_algebra.h  ../include/agg_vcgen_vertex_sequence.h
	../include/agg_ellipse.h                ../include/agg_scanline_p.h                ../include/agg_vertex_sequence.h
	../include/agg_ellipse_bresenham.h      ../include/agg_scanline_storage_aa.h       ../include/agg_vpgen_clip_polygon.h
	../include/agg_embedded_raster_fonts.h  ../include/agg_scanline_storage_bin.h      ../include/agg_vpgen_clip_polyline.h
	../include/agg_font_cache_manager.h     ../include/agg_scanline_u.h                ../include/agg_vpgen_segmentator.h
	../include/agg_gamma_functions.h        ../include/agg_shorten_path.h
	../include/agg_gamma_lut.h              ../include/agg_simul_eq.h
	)
source_group("AGG Sources" FILES ${AGG_SOURCES})
source_group("AGG Headers" FILES ${AGG_HEADERS})
source_group("AGG Control Sources" FILES ${AGG_CTRL_SOURCES})
source_group("Polygon Clipper" FILES ${CLIPPER_SOURCES})

add_library(agg STATIC ${AGG_HEADERS} ${AGG_SOURCES} ${AGG_CTRL_SOURCES} ${CLIPPER_SOURCES})

set_target_properties(agg PROPERTIES FOLDER "AGG")

add_library(aggexamples STATIC 
	../examples/parse_lion.cpp 
	../examples/make_gb_poly.cpp 
	../examples/make_arrows.cpp 
	../examples/interactive_polygon.cpp)	
set_target_properties(aggexamples PROPERTIES FOLDER "AGG/examples")

add_library(aggfontwin32 STATIC ../font_win32_tt/agg_font_win32_tt.cpp)	
set_target_properties(aggfontwin32 PROPERTIES FOLDER "AGG")

add_library(aggfontfreetype STATIC ../font_freetype/agg_font_freetype.cpp)	
set_target_properties(aggfontfreetype PROPERTIES FOLDER "AGG")

add_executable(clipper_test WIN32 ../examples/clipper_test.cpp)
target_link_libraries(clipper_test agg aggplatform aggexamples)
set_target_properties(clipper_test PROPERTIES FOLDER "AGG/examples")

add_executable(aa_demo WIN32 ../examples/aa_demo.cpp)
target_link_libraries(aa_demo agg aggplatform aggexamples)
set_target_properties(aa_demo PROPERTIES FOLDER "AGG/examples")

add_executable(aa_test WIN32 ../examples/aa_test.cpp)
target_link_libraries(aa_test agg aggplatform aggexamples)
set_target_properties(aa_test PROPERTIES FOLDER "AGG/examples")

add_executable(alpha_gradient WIN32 ../examples/alpha_gradient.cpp)
target_link_libraries(alpha_gradient agg aggplatform aggexamples)
set_target_properties(alpha_gradient PROPERTIES FOLDER "AGG/examples")

add_executable(alpha_mask2 WIN32 ../examples/alpha_mask2.cpp)
target_link_libraries(alpha_mask2 agg aggplatform aggexamples)
set_target_properties(alpha_mask2 PROPERTIES FOLDER "AGG/examples")

add_executable(alpha_mask3 WIN32 ../examples/alpha_mask3.cpp)
target_link_libraries(alpha_mask3 agg aggplatform aggexamples)
set_target_properties(alpha_mask3 PROPERTIES FOLDER "AGG/examples")

add_executable(alpha_mask WIN32 ../examples/alpha_mask.cpp)
target_link_libraries(alpha_mask agg aggplatform aggexamples)
set_target_properties(alpha_mask PROPERTIES FOLDER "AGG/examples")

add_executable(bezier_div WIN32 ../examples/bezier_div.cpp)
target_link_libraries(bezier_div agg aggplatform aggexamples)
set_target_properties(bezier_div PROPERTIES FOLDER "AGG/examples")

add_executable(bspline WIN32 ../examples/bspline.cpp)
target_link_libraries(bspline agg aggplatform aggexamples)
set_target_properties(bspline PROPERTIES FOLDER "AGG/examples")

add_executable(circles WIN32 ../examples/circles.cpp)
target_link_libraries(circles agg aggplatform aggexamples)
set_target_properties(circles PROPERTIES FOLDER "AGG/examples")

add_executable(component_rendering WIN32 ../examples/component_rendering.cpp)
target_link_libraries(component_rendering agg aggplatform aggexamples)
set_target_properties(component_rendering PROPERTIES FOLDER "AGG/examples")

add_executable(compositing WIN32 ../examples/compositing.cpp)
target_link_libraries(compositing agg aggplatform aggexamples)
set_target_properties(compositing PROPERTIES FOLDER "AGG/examples")

add_executable(conv_contour WIN32 ../examples/conv_contour.cpp)
target_link_libraries(conv_contour agg aggplatform aggexamples)
set_target_properties(conv_contour PROPERTIES FOLDER "AGG/examples")

add_executable(conv_dash_marker WIN32 ../examples/conv_dash_marker.cpp)
target_link_libraries(conv_dash_marker agg aggplatform aggexamples)
set_target_properties(conv_dash_marker PROPERTIES FOLDER "AGG/examples")

add_executable(conv_stroke WIN32 ../examples/conv_stroke.cpp)
target_link_libraries(conv_stroke agg aggplatform aggexamples)
set_target_properties(conv_stroke PROPERTIES FOLDER "AGG/examples")

add_executable(distortions WIN32 ../examples/distortions.cpp)
target_link_libraries(distortions agg aggplatform aggexamples)
set_target_properties(distortions PROPERTIES FOLDER "AGG/examples")

add_executable(gamma_correction WIN32 ../examples/gamma_correction.cpp)
target_link_libraries(gamma_correction agg aggplatform aggexamples)
set_target_properties(gamma_correction PROPERTIES FOLDER "AGG/examples")

add_executable(gamma_ctrl WIN32 ../examples/gamma_ctrl.cpp)
target_link_libraries(gamma_ctrl agg aggplatform aggexamples)
set_target_properties(gamma_ctrl PROPERTIES FOLDER "AGG/examples")

add_executable(gouraud WIN32 ../examples/gouraud.cpp)
target_link_libraries(gouraud agg aggplatform aggexamples)
set_target_properties(gouraud PROPERTIES FOLDER "AGG/examples")

add_executable(gradients WIN32 ../examples/gradients.cpp)
target_link_libraries(gradients agg aggplatform aggexamples)
set_target_properties(gradients PROPERTIES FOLDER "AGG/examples")

add_executable(graph_test WIN32 ../examples/graph_test.cpp)
target_link_libraries(graph_test agg aggplatform aggexamples)
set_target_properties(graph_test PROPERTIES FOLDER "AGG/examples")

add_executable(idea WIN32 ../examples/idea.cpp)
target_link_libraries(idea agg aggplatform aggexamples)
set_target_properties(idea PROPERTIES FOLDER "AGG/examples")

add_executable(image1 WIN32 ../examples/image1.cpp)
target_link_libraries(image1 agg aggplatform aggexamples)
set_target_properties(image1 PROPERTIES FOLDER "AGG/examples")

add_executable(image_alpha WIN32 ../examples/image_alpha.cpp)
target_link_libraries(image_alpha agg aggplatform aggexamples)
set_target_properties(image_alpha PROPERTIES FOLDER "AGG/examples")

add_executable(image_filters2 WIN32 ../examples/image_filters2.cpp)
target_link_libraries(image_filters2 agg aggplatform aggexamples)
set_target_properties(image_filters2 PROPERTIES FOLDER "AGG/examples")

add_executable(image_filters WIN32 ../examples/image_filters.cpp)
target_link_libraries(image_filters agg aggplatform aggexamples)
set_target_properties(image_filters PROPERTIES FOLDER "AGG/examples")

add_executable(image_fltr_graph WIN32 ../examples/image_fltr_graph.cpp)
target_link_libraries(image_fltr_graph agg aggplatform aggexamples)
set_target_properties(image_fltr_graph PROPERTIES FOLDER "AGG/examples")

add_executable(image_perspective WIN32 ../examples/image_perspective.cpp)
target_link_libraries(image_perspective agg aggplatform aggexamples)
set_target_properties(image_perspective PROPERTIES FOLDER "AGG/examples")

add_executable(image_resample WIN32 ../examples/image_resample.cpp)
target_link_libraries(image_resample agg aggplatform aggexamples)
set_target_properties(image_resample PROPERTIES FOLDER "AGG/examples")

add_executable(image_transforms WIN32 ../examples/image_transforms.cpp)
target_link_libraries(image_transforms agg aggplatform aggexamples)
set_target_properties(image_transforms PROPERTIES FOLDER "AGG/examples")

add_executable(line_patterns WIN32 ../examples/line_patterns.cpp)
target_link_libraries(line_patterns agg aggplatform aggexamples)
set_target_properties(line_patterns PROPERTIES FOLDER "AGG/examples")

add_executable(lion WIN32 ../examples/lion.cpp)
target_link_libraries(lion agg aggplatform aggexamples)
set_target_properties(lion PROPERTIES FOLDER "AGG/examples")

add_executable(lion_lens WIN32 ../examples/lion_lens.cpp)
target_link_libraries(lion_lens agg aggplatform aggexamples)
set_target_properties(lion_lens PROPERTIES FOLDER "AGG/examples")

add_executable(lion_outline WIN32 ../examples/lion_outline.cpp)
target_link_libraries(lion_outline agg aggplatform aggexamples)
set_target_properties(lion_outline PROPERTIES FOLDER "AGG/examples")

add_executable(mol_view WIN32 ../examples/mol_view.cpp)
target_link_libraries(mol_view agg aggplatform aggexamples)
set_target_properties(mol_view PROPERTIES FOLDER "AGG/examples")

add_executable(multi_clip WIN32 ../examples/multi_clip.cpp)
target_link_libraries(multi_clip agg aggplatform aggexamples)
set_target_properties(multi_clip PROPERTIES FOLDER "AGG/examples")

add_executable(pattern_fill WIN32 ../examples/pattern_fill.cpp)
target_link_libraries(pattern_fill agg aggplatform aggexamples)
set_target_properties(pattern_fill PROPERTIES FOLDER "AGG/examples")

add_executable(pattern_perspective WIN32 ../examples/pattern_perspective.cpp)
target_link_libraries(pattern_perspective agg aggplatform aggexamples)
set_target_properties(pattern_perspective PROPERTIES FOLDER "AGG/examples")

add_executable(pattern_resample WIN32 ../examples/pattern_resample.cpp)
target_link_libraries(pattern_resample agg aggplatform aggexamples)
set_target_properties(pattern_resample PROPERTIES FOLDER "AGG/examples")

add_executable(perspective WIN32 ../examples/perspective.cpp)
target_link_libraries(perspective agg aggplatform aggexamples)
set_target_properties(perspective PROPERTIES FOLDER "AGG/examples")

add_executable(polymorphic_renderer WIN32 ../examples/polymorphic_renderer.cpp)
target_link_libraries(polymorphic_renderer agg aggplatform aggexamples)
set_target_properties(polymorphic_renderer PROPERTIES FOLDER "AGG/examples")

add_executable(rasterizers2 WIN32 ../examples/rasterizers2.cpp)
target_link_libraries(rasterizers2 agg aggplatform aggexamples)
set_target_properties(rasterizers2 PROPERTIES FOLDER "AGG/examples")

add_executable(rasterizers WIN32 ../examples/rasterizers.cpp)
target_link_libraries(rasterizers agg aggplatform aggexamples)
set_target_properties(rasterizers PROPERTIES FOLDER "AGG/examples")

add_executable(raster_text WIN32 ../examples/raster_text.cpp)
target_link_libraries(raster_text agg aggplatform aggexamples)
set_target_properties(raster_text PROPERTIES FOLDER "AGG/examples")

add_executable(rounded_rect WIN32 ../examples/rounded_rect.cpp)
target_link_libraries(rounded_rect agg aggplatform aggexamples)
set_target_properties(rounded_rect PROPERTIES FOLDER "AGG/examples")

add_executable(scanline_boolean2 WIN32 ../examples/scanline_boolean2.cpp)
target_link_libraries(scanline_boolean2 agg aggplatform aggexamples)
set_target_properties(scanline_boolean2 PROPERTIES FOLDER "AGG/examples")

add_executable(scanline_boolean WIN32 ../examples/scanline_boolean.cpp)
target_link_libraries(scanline_boolean agg aggplatform aggexamples)
set_target_properties(scanline_boolean PROPERTIES FOLDER "AGG/examples")

add_executable(simple_blur WIN32 ../examples/simple_blur.cpp)
target_link_libraries(simple_blur agg aggplatform aggexamples)
set_target_properties(simple_blur PROPERTIES FOLDER "AGG/examples")

add_executable(trans_polar WIN32 ../examples/trans_polar.cpp)
target_link_libraries(trans_polar agg aggplatform aggexamples)
set_target_properties(trans_polar PROPERTIES FOLDER "AGG/examples")

add_executable(truetype_test WIN32 ../examples/truetype_test.cpp)
target_link_libraries(truetype_test agg aggplatform aggexamples aggfontwin32)
set_target_properties(truetype_test PROPERTIES FOLDER "AGG/examples")

add_executable(trans_curve1 WIN32 ../examples/trans_curve1.cpp)
target_link_libraries(trans_curve1 agg aggplatform aggexamples aggfontwin32)
set_target_properties(trans_curve1 PROPERTIES FOLDER "AGG/examples")

add_executable(trans_curve2 WIN32 ../examples/trans_curve2.cpp)
target_link_libraries(trans_curve2 agg aggplatform aggexamples aggfontwin32)
set_target_properties(trans_curve2 PROPERTIES FOLDER "AGG/examples")

add_executable(freetype_test WIN32 ../examples/freetype_test.cpp)
target_link_libraries(freetype_test agg aggplatform aggexamples ${FREETYPE_LIBRARIES} aggfontfreetype)
set_target_properties(freetype_test PROPERTIES FOLDER "AGG/examples")

add_executable(trans_curve2_ft WIN32 ../examples/trans_curve2_ft.cpp)
target_link_libraries(trans_curve2_ft agg aggplatform aggexamples ${FREETYPE_LIBRARIES} aggfontfreetype)
set_target_properties(trans_curve2_ft PROPERTIES FOLDER "AGG/examples")

add_executable(trans_curve1_ft WIN32 ../examples/trans_curve1_ft.cpp)
target_link_libraries(trans_curve1_ft agg aggplatform aggexamples ${FREETYPE_LIBRARIES} aggfontfreetype)
set_target_properties(trans_curve1_ft PROPERTIES FOLDER "AGG/examples")
