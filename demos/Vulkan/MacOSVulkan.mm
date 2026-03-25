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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <mach/mach_time.h>

#define VK_USE_PLATFORM_METAL_EXT
#include <vulkan/vulkan.h>

#import <Cocoa/Cocoa.h>
#import <QuartzCore/CAMetalLayer.h>

#include "aeongui/AeonGUI.hpp"
#include "aeongui/dom/Window.hpp"
#include "aeongui/dom/Document.hpp"
#include "VulkanRenderer.h"

// ─────────────────────────────────────────────────────────────────────────────
// VulkanView – an NSView subclass backed by a CAMetalLayer for Vulkan
// ─────────────────────────────────────────────────────────────────────────────
@interface VulkanView : NSView
{
@private
    NSTrackingArea*         trackingArea;
    AeonGUI::DOM::Window*   mWindow;
    VulkanRenderer*         mRenderer;
    NSTimer*                renderTimer;
    std::string             mVertPath;
    std::string             mFragPath;
}

- (instancetype)initWithFrame:(NSRect)frameRect
                       window:(AeonGUI::DOM::Window*)window
                     renderer:(VulkanRenderer*)renderer
                     vertPath:(const std::string&)vertPath
                     fragPath:(const std::string&)fragPath;
- (void)render:(NSTimer*)timer;
- (void)stopRendering;

@end

@implementation VulkanView

- (instancetype)initWithFrame:(NSRect)frameRect
                       window:(AeonGUI::DOM::Window*)window
                     renderer:(VulkanRenderer*)renderer
                     vertPath:(const std::string&)vertPath
                     fragPath:(const std::string&)fragPath
{
    self = [super initWithFrame:frameRect];
    if (self) {
        mWindow   = window;
        mRenderer = renderer;
        mVertPath = vertPath;
        mFragPath = fragPath;

        self.wantsLayer = YES;
        self.layer = [CAMetalLayer layer];

        trackingArea = [[NSTrackingArea alloc]
                        initWithRect:[self bounds]
                        options:(NSTrackingActiveAlways | NSTrackingMouseMoved)
                        owner:self
                        userInfo:nil];
        [self addTrackingArea:trackingArea];

        renderTimer = [NSTimer scheduledTimerWithTimeInterval:1.0 / 60.0
                                                       target:self
                                                     selector:@selector(render:)
                                                     userInfo:nil
                                                      repeats:YES];
    }
    return self;
}

- (void)dealloc
{
    if (renderTimer) {
        [renderTimer invalidate];
        renderTimer = nil;
    }
    if (trackingArea) {
        [self removeTrackingArea:trackingArea];
        [trackingArea release];
    }
    [super dealloc];
}

- (CALayer*)makeBackingLayer
{
    return [CAMetalLayer layer];
}

- (void)setFrameSize:(NSSize)newSize
{
    [super setFrameSize:newSize];
    uint32_t w = static_cast<uint32_t>(newSize.width);
    uint32_t h = static_cast<uint32_t>(newSize.height);
    if (w > 0 && h > 0 && mRenderer) {
        mRenderer->RecreateSwapchain(w, h, *mWindow, mVertPath, mFragPath);
    }
}

- (void)render:(NSTimer*)timer
{
    if (mRenderer && mWindow) {
        mRenderer->DrawFrame(*mWindow);
    }
}

- (void)stopRendering
{
    if (renderTimer) {
        [renderTimer invalidate];
        renderTimer = nil;
    }
    mRenderer = nullptr;
    mWindow = nullptr;
}

- (void)mouseDown:(NSEvent*)event
{
    NSPoint location = [self convertPoint:[event locationInWindow] fromView:nil];
    location.y = [self bounds].size.height - location.y;
    short button = ([event buttonNumber] == 0) ? 0 : ([event buttonNumber] == 1) ? 2 : 1;
    mWindow->HandleMouseDown(static_cast<double>(location.x),
                             static_cast<double>(location.y), button);
}

- (void)mouseUp:(NSEvent*)event
{
    NSPoint location = [self convertPoint:[event locationInWindow] fromView:nil];
    location.y = [self bounds].size.height - location.y;
    short button = ([event buttonNumber] == 0) ? 0 : ([event buttonNumber] == 1) ? 2 : 1;
    mWindow->HandleMouseUp(static_cast<double>(location.x),
                           static_cast<double>(location.y), button);
}

- (void)rightMouseDown:(NSEvent*)event
{
    NSPoint location = [self convertPoint:[event locationInWindow] fromView:nil];
    location.y = [self bounds].size.height - location.y;
    mWindow->HandleMouseDown(static_cast<double>(location.x),
                             static_cast<double>(location.y), 2);
}

- (void)rightMouseUp:(NSEvent*)event
{
    NSPoint location = [self convertPoint:[event locationInWindow] fromView:nil];
    location.y = [self bounds].size.height - location.y;
    mWindow->HandleMouseUp(static_cast<double>(location.x),
                           static_cast<double>(location.y), 2);
}

- (void)otherMouseDown:(NSEvent*)event
{
    NSPoint location = [self convertPoint:[event locationInWindow] fromView:nil];
    location.y = [self bounds].size.height - location.y;
    mWindow->HandleMouseDown(static_cast<double>(location.x),
                             static_cast<double>(location.y), 1);
}

- (void)otherMouseUp:(NSEvent*)event
{
    NSPoint location = [self convertPoint:[event locationInWindow] fromView:nil];
    location.y = [self bounds].size.height - location.y;
    mWindow->HandleMouseUp(static_cast<double>(location.x),
                           static_cast<double>(location.y), 1);
}

- (void)mouseMoved:(NSEvent*)event
{
    NSPoint location = [self convertPoint:[event locationInWindow] fromView:nil];
    location.y = [self bounds].size.height - location.y;
    mWindow->HandleMouseMove(static_cast<double>(location.x),
                             static_cast<double>(location.y));
}

- (void)mouseDragged:(NSEvent*)event
{
    [self mouseMoved:event];
}

- (void)rightMouseDragged:(NSEvent*)event
{
    [self mouseMoved:event];
}

- (void)otherMouseDragged:(NSEvent*)event
{
    [self mouseMoved:event];
}

- (void)scrollWheel:(NSEvent*)event
{
    NSPoint location = [self convertPoint:[event locationInWindow] fromView:nil];
    location.y = [self bounds].size.height - location.y;
    mWindow->HandleWheel(static_cast<double>(location.x),
                         static_cast<double>(location.y),
                         static_cast<double>([event deltaX]),
                         static_cast<double>([event deltaY]));
}

- (void)keyDown:(NSEvent*)event     { }
- (void)keyUp:(NSEvent*)event       { }

- (BOOL)acceptsFirstResponder
{
    return YES;
}

@end

// ─────────────────────────────────────────────────────────────────────────────
// AppDelegate
// ─────────────────────────────────────────────────────────────────────────────
@interface AppDelegate : NSObject <NSApplicationDelegate>
{
@private
    NSWindow*               window;
    VulkanView*             vulkanView;
    AeonGUI::DOM::Window*   aeonWindow;
    VulkanRenderer*         renderer;
}

- (id)initWithFilename:(const char*)filename
                 width:(uint32_t)width
                height:(uint32_t)height
              vertPath:(const std::string&)vertPath
              fragPath:(const std::string&)fragPath;

@end

@implementation AppDelegate

- (id)initWithFilename:(const char*)filename
                 width:(uint32_t)width
                height:(uint32_t)height
              vertPath:(const std::string&)vertPath
              fragPath:(const std::string&)fragPath
{
    self = [super init];
    if (self) {
        aeonWindow = new AeonGUI::DOM::Window(width, height);
        if (filename) {
            aeonWindow->location() = filename;
        }

        renderer = new VulkanRenderer();
        renderer->CreateInstance({
            VK_EXT_METAL_SURFACE_EXTENSION_NAME
        });

        NSRect frame = NSMakeRect(0, 0, width, height);
        window = [[NSWindow alloc]
                  initWithContentRect:frame
                  styleMask:(NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskResizable)
                  backing:NSBackingStoreBuffered
                  defer:NO];

        [window setTitle:@"AeonGUI \xe2\x80\x93 Vulkan"];
        [window center];

        vulkanView = [[VulkanView alloc] initWithFrame:frame
                                                window:aeonWindow
                                              renderer:renderer
                                              vertPath:vertPath
                                              fragPath:fragPath];
        [window setContentView:vulkanView];

        // Create Vulkan surface from the CAMetalLayer
        VkMetalSurfaceCreateInfoEXT surfaceInfo{};
        surfaceInfo.sType  = VK_STRUCTURE_TYPE_METAL_SURFACE_CREATE_INFO_EXT;
        surfaceInfo.pLayer = (CAMetalLayer*)[vulkanView layer];

        VkSurfaceKHR surface;
        VK_CHECK(vkCreateMetalSurfaceEXT(renderer->GetInstance(), &surfaceInfo, nullptr, &surface));
        renderer->SetSurface(surface);

        renderer->Initialize(width, height, vertPath, fragPath,
                             static_cast<uint32_t>(aeonWindow->GetWidth()),
                             static_cast<uint32_t>(aeonWindow->GetHeight()));

        [window makeKeyAndOrderFront:nil];
    }
    return self;
}

- (void)dealloc
{
    [vulkanView stopRendering];
    if (renderer) { renderer->Cleanup(); delete renderer; renderer = nullptr; }
    delete aeonWindow;
    aeonWindow = nullptr;
    [vulkanView release];
    [window release];
    [super dealloc];
}

- (void)applicationDidFinishLaunching:(NSNotification*)notification { }

- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication*)app
{
    return YES;
}

- (void)applicationWillTerminate:(NSNotification*)notification { }

@end

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv)
{
    AeonGUI::Initialize(argc, argv);

    // Determine shader path (next to executable)
    std::string binDir = argv[0];
    binDir = binDir.substr(0, binDir.rfind('/') + 1);
    std::string vertPath = binDir + "screen.vert.spv";
    std::string fragPath = binDir + "screen.frag.spv";

    @autoreleasepool {
        NSApplication* app = [NSApplication sharedApplication];
        [app setActivationPolicy:NSApplicationActivationPolicyRegular];

        AppDelegate* delegate = [[AppDelegate alloc]
                                 initWithFilename:(argc > 1) ? argv[1] : nullptr
                                 width:800
                                 height:600
                                 vertPath:vertPath
                                 fragPath:fragPath];
        [app setDelegate:delegate];
        [app run];
        [delegate release];
    }

    AeonGUI::Finalize();
    return 0;
}
