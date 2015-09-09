subdirs(platform)
include_directories(../clipper)
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

set(CLIPPER_TEST_SOURCES 
	../examples/clipper_test.cpp 
	../examples/make_gb_poly.cpp
	../examples/make_arrows.cpp
	)
add_executable(clipper_test WIN32 ${CLIPPER_TEST_SOURCES})
target_link_libraries(clipper_test agg aggplatform)