/*
Copyright (C) 2026 Rodrigo Jose Hernandez Cordoba

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Phase 1 spike: prove Yoga is linked and behaves as expected.
//
// No DOM, no rendering — just builds a small flex tree by hand, runs layout,
// and asserts coordinates. Validates the integration shape end-to-end before
// any HTML element classes or LayoutEngine are introduced.
//
// Layout under test:
//
//   root: 800 x 600, flex column
//     header:    flex-grow 0, height 40, full width
//     body:      flex-grow 1                       (fills remaining height)
//     fps overlay: position absolute, top 8, right 8, 80 x 20

#include <gtest/gtest.h>
#include <yoga/Yoga.h>

namespace
{
    constexpr float kRootWidth  = 800.0f;
    constexpr float kRootHeight = 600.0f;
    constexpr float kHeaderHeight = 40.0f;
    constexpr float kFpsWidth = 80.0f;
    constexpr float kFpsHeight = 20.0f;
    constexpr float kFpsTop = 8.0f;
    constexpr float kFpsRight = 8.0f;
}

TEST ( HTMLLayoutSpike, FlexColumnWithAbsoluteOverlay )
{
    YGNodeRef root = YGNodeNew();
    YGNodeStyleSetFlexDirection ( root, YGFlexDirectionColumn );
    YGNodeStyleSetWidth ( root, kRootWidth );
    YGNodeStyleSetHeight ( root, kRootHeight );

    YGNodeRef header = YGNodeNew();
    YGNodeStyleSetHeight ( header, kHeaderHeight );

    YGNodeRef body = YGNodeNew();
    YGNodeStyleSetFlexGrow ( body, 1.0f );

    YGNodeRef fps = YGNodeNew();
    YGNodeStyleSetPositionType ( fps, YGPositionTypeAbsolute );
    YGNodeStyleSetPosition ( fps, YGEdgeTop, kFpsTop );
    YGNodeStyleSetPosition ( fps, YGEdgeRight, kFpsRight );
    YGNodeStyleSetWidth ( fps, kFpsWidth );
    YGNodeStyleSetHeight ( fps, kFpsHeight );

    YGNodeInsertChild ( root, header, 0 );
    YGNodeInsertChild ( root, body, 1 );
    YGNodeInsertChild ( root, fps, 2 );

    YGNodeCalculateLayout ( root, YGUndefined, YGUndefined, YGDirectionLTR );

    EXPECT_FLOAT_EQ ( YGNodeLayoutGetWidth ( root ), kRootWidth );
    EXPECT_FLOAT_EQ ( YGNodeLayoutGetHeight ( root ), kRootHeight );

    EXPECT_FLOAT_EQ ( YGNodeLayoutGetTop ( header ), 0.0f );
    EXPECT_FLOAT_EQ ( YGNodeLayoutGetLeft ( header ), 0.0f );
    EXPECT_FLOAT_EQ ( YGNodeLayoutGetWidth ( header ), kRootWidth );
    EXPECT_FLOAT_EQ ( YGNodeLayoutGetHeight ( header ), kHeaderHeight );

    EXPECT_FLOAT_EQ ( YGNodeLayoutGetTop ( body ), kHeaderHeight );
    EXPECT_FLOAT_EQ ( YGNodeLayoutGetLeft ( body ), 0.0f );
    EXPECT_FLOAT_EQ ( YGNodeLayoutGetWidth ( body ), kRootWidth );
    EXPECT_FLOAT_EQ ( YGNodeLayoutGetHeight ( body ), kRootHeight - kHeaderHeight );

    // Absolute overlay: pinned top/right, ignored by sibling flex flow.
    EXPECT_FLOAT_EQ ( YGNodeLayoutGetTop ( fps ), kFpsTop );
    EXPECT_FLOAT_EQ ( YGNodeLayoutGetLeft ( fps ), kRootWidth - kFpsWidth - kFpsRight );
    EXPECT_FLOAT_EQ ( YGNodeLayoutGetWidth ( fps ), kFpsWidth );
    EXPECT_FLOAT_EQ ( YGNodeLayoutGetHeight ( fps ), kFpsHeight );

    YGNodeFreeRecursive ( root );
}
