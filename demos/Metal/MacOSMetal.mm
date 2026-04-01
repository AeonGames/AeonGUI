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

#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>
#import <Cocoa/Cocoa.h>

#include "aeongui/AeonGUI.hpp"
#include "aeongui/dom/Window.hpp"
#include "aeongui/dom/Document.hpp"

// ─────────────────────────────────────────────────────────────────────────────
// Vertex data – full-screen quad (positions + tex-coords, same as OpenGL demo)
// ─────────────────────────────────────────────────────────────────────────────
struct Vertex {
    float position[2];
    float texCoord[2];
};

static const Vertex vertices[] = {
    // positions    // texCoords
    { {-1.0f,  1.0f}, {0.0f, 0.0f} },
    { {-1.0f, -1.0f}, {0.0f, 1.0f} },
    { { 1.0f, -1.0f}, {1.0f, 1.0f} },
    { { 1.0f,  1.0f}, {1.0f, 0.0f} },
};

static const uint16_t indices[] = { 0, 1, 2, 0, 2, 3 };

// ─────────────────────────────────────────────────────────────────────────────
// MetalView – an MTKView subclass that drives rendering
// ─────────────────────────────────────────────────────────────────────────────
@interface MetalView : MTKView <MTKViewDelegate>
{
@private
    NSTrackingArea*             trackingArea;
    AeonGUI::DOM::Window*       mWindow;

    id<MTLCommandQueue>         mCommandQueue;
    id<MTLRenderPipelineState>  mPipelineState;
    id<MTLBuffer>               mVertexBuffer;
    id<MTLBuffer>               mIndexBuffer;
    id<MTLTexture>              mScreenTexture;
}

- (instancetype)initWithFrame:(NSRect)frameRect
                       device:(id<MTLDevice>)device
                       window:(AeonGUI::DOM::Window*)window;
@end

@implementation MetalView

- (instancetype)initWithFrame:(NSRect)frameRect
                       device:(id<MTLDevice>)device
                       window:(AeonGUI::DOM::Window*)window
{
    self = [super initWithFrame:frameRect device:device];
    if (self) {
        mWindow = window;

        self.delegate               = self;
        self.colorPixelFormat        = MTLPixelFormatBGRA8Unorm;
        self.clearColor              = MTLClearColorMake(1.0, 1.0, 1.0, 1.0);
        self.preferredFramesPerSecond = 60;

        trackingArea = [[NSTrackingArea alloc]
                        initWithRect:[self bounds]
                        options:(NSTrackingActiveAlways | NSTrackingMouseMoved)
                        owner:self
                        userInfo:nil];
        [self addTrackingArea:trackingArea];

        [self setupMetal];
    }
    return self;
}

- (void)dealloc
{
    if (trackingArea) {
        [self removeTrackingArea:trackingArea];
        [trackingArea release];
    }
    [mVertexBuffer release];
    [mIndexBuffer release];
    [mScreenTexture release];
    [mCommandQueue release];
    [mPipelineState release];
    [super dealloc];
}

// ── Metal setup ──────────────────────────────────────────────────────────────

- (void)setupMetal
{
    id<MTLDevice> device = self.device;
    mCommandQueue = [device newCommandQueue];

    // ── Load shaders ─────────────────────────────────────────────────────
    NSError* error = nil;

    // Try loading from a pre-compiled .metallib next to the executable first,
    // fall back to compiling the .metal source at runtime.
    NSString* libPath = [[[NSBundle mainBundle] executablePath]
                          stringByDeletingLastPathComponent];
    NSString* metalLibPath = [libPath stringByAppendingPathComponent:@"Shaders.metallib"];

    id<MTLLibrary> library = nil;
    if ([[NSFileManager defaultManager] fileExistsAtPath:metalLibPath]) {
        NSURL* metalLibURL = [NSURL fileURLWithPath:metalLibPath];
        library = [device newLibraryWithURL:metalLibURL error:&error];
    }

    if (!library) {
        // Try compiling from source next to the executable
        NSString* metalSrcPath = [libPath stringByAppendingPathComponent:@"Shaders.metal"];
        if ([[NSFileManager defaultManager] fileExistsAtPath:metalSrcPath]) {
            NSString* source = [NSString stringWithContentsOfFile:metalSrcPath
                                                        encoding:NSUTF8StringEncoding
                                                           error:&error];
            if (source) {
                library = [device newLibraryWithSource:source options:nil error:&error];
            }
        }
    }

    if (!library) {
        NSLog(@"Failed to load Metal library: %@", error);
        return;
    }

    id<MTLFunction> vertexFunc   = [library newFunctionWithName:@"vertexShader"];
    id<MTLFunction> fragmentFunc = [library newFunctionWithName:@"fragmentShader"];

    // ── Vertex descriptor ────────────────────────────────────────────────
    MTLVertexDescriptor* vertexDescriptor = [MTLVertexDescriptor vertexDescriptor];
    vertexDescriptor.attributes[0].format      = MTLVertexFormatFloat2;
    vertexDescriptor.attributes[0].offset      = offsetof(Vertex, position);
    vertexDescriptor.attributes[0].bufferIndex = 0;
    vertexDescriptor.attributes[1].format      = MTLVertexFormatFloat2;
    vertexDescriptor.attributes[1].offset      = offsetof(Vertex, texCoord);
    vertexDescriptor.attributes[1].bufferIndex = 0;
    vertexDescriptor.layouts[0].stride         = sizeof(Vertex);

    // ── Pipeline state ───────────────────────────────────────────────────
    MTLRenderPipelineDescriptor* pipeDesc = [[MTLRenderPipelineDescriptor alloc] init];
    pipeDesc.vertexFunction                           = vertexFunc;
    pipeDesc.fragmentFunction                         = fragmentFunc;
    pipeDesc.vertexDescriptor                         = vertexDescriptor;
    pipeDesc.colorAttachments[0].pixelFormat           = self.colorPixelFormat;
    pipeDesc.colorAttachments[0].blendingEnabled       = YES;
    pipeDesc.colorAttachments[0].sourceRGBBlendFactor  = MTLBlendFactorSourceAlpha;
    pipeDesc.colorAttachments[0].destinationRGBBlendFactor  = MTLBlendFactorOneMinusSourceAlpha;
    pipeDesc.colorAttachments[0].sourceAlphaBlendFactor     = MTLBlendFactorOne;
    pipeDesc.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;

    mPipelineState = [device newRenderPipelineStateWithDescriptor:pipeDesc error:&error];
    [pipeDesc release];
    [vertexFunc release];
    [fragmentFunc release];
    [library release];

    if (!mPipelineState) {
        NSLog(@"Failed to create pipeline state: %@", error);
        return;
    }

    // ── Vertex / index buffers ───────────────────────────────────────────
    mVertexBuffer = [device newBufferWithBytes:vertices
                                        length:sizeof(vertices)
                                       options:MTLResourceStorageModeShared];
    mIndexBuffer  = [device newBufferWithBytes:indices
                                        length:sizeof(indices)
                                       options:MTLResourceStorageModeShared];

    // ── Screen texture ───────────────────────────────────────────────────
    [self rebuildTextureWithWidth:mWindow->GetWidth() height:mWindow->GetHeight()];
}

- (void)rebuildTextureWithWidth:(NSUInteger)width height:(NSUInteger)height
{
    if (mScreenTexture) {
        [mScreenTexture release];
        mScreenTexture = nil;
    }

    MTLTextureDescriptor* texDesc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                          width:width
                                                         height:height
                                                      mipmapped:NO];
    texDesc.usage = MTLTextureUsageShaderRead;
    mScreenTexture = [self.device newTextureWithDescriptor:texDesc];
}

// ── MTKViewDelegate ──────────────────────────────────────────────────────────

- (void)drawInMTKView:(nonnull MTKView*)view
{
    if ( mWindow->Draw() )
    {
        // Upload pixels from AeonGUI into the Metal texture
        NSUInteger width  = mWindow->GetWidth();
        NSUInteger height = mWindow->GetHeight();
        NSUInteger stride = mWindow->GetStride();

        [mScreenTexture replaceRegion:MTLRegionMake2D(0, 0, width, height)
                          mipmapLevel:0
                            withBytes:mWindow->GetPixels()
                          bytesPerRow:stride];
    }

    // ── Render ───────────────────────────────────────────────────────────
    id<MTLCommandBuffer> commandBuffer = [mCommandQueue commandBuffer];
    MTLRenderPassDescriptor* rpd = view.currentRenderPassDescriptor;
    if (!rpd) return;

    id<MTLRenderCommandEncoder> encoder =
        [commandBuffer renderCommandEncoderWithDescriptor:rpd];

    [encoder setRenderPipelineState:mPipelineState];
    [encoder setVertexBuffer:mVertexBuffer offset:0 atIndex:0];
    [encoder setFragmentTexture:mScreenTexture atIndex:0];
    [encoder drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                        indexCount:6
                         indexType:MTLIndexTypeUInt16
                       indexBuffer:mIndexBuffer
                 indexBufferOffset:0];

    [encoder endEncoding];
    [commandBuffer presentDrawable:view.currentDrawable];
    [commandBuffer commit];
}

- (void)mtkView:(nonnull MTKView*)view drawableSizeWillChange:(CGSize)size
{
    mWindow->ResizeViewport((size_t)size.width, (size_t)size.height);
    [self rebuildTextureWithWidth:mWindow->GetWidth() height:mWindow->GetHeight()];
}

// ── Input events (stubs, matching OpenGL demo) ───────────────────────────────

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

- (void)keyDown:(NSEvent*)event { }
- (void)keyUp:(NSEvent*)event   { }

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
    NSWindow*                window;
    MetalView*               metalView;
    AeonGUI::DOM::Window*    aeonWindow;
}

- (id)initWithFilename:(const char*)filename width:(uint32_t)width height:(uint32_t)height;

@end

@implementation AppDelegate

- (id)initWithFilename:(const char*)filename width:(uint32_t)width height:(uint32_t)height
{
    self = [super init];
    if (self) {
        aeonWindow = new AeonGUI::DOM::Window(width, height);
        if (filename) {
            aeonWindow->location() = filename;
        }

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            NSLog(@"Metal is not supported on this device");
            return nil;
        }

        NSRect frame = NSMakeRect(0, 0, width, height);
        window = [[NSWindow alloc]
                  initWithContentRect:frame
                  styleMask:(NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskResizable)
                  backing:NSBackingStoreBuffered
                  defer:NO];

        [window setTitle:@"AeonGUI - Metal"];
        [window center];

        metalView = [[MetalView alloc] initWithFrame:frame device:device window:aeonWindow];
        [window setContentView:metalView];
        [window makeKeyAndOrderFront:nil];
    }
    return self;
}

- (void)dealloc
{
    delete aeonWindow;
    [metalView release];
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

    @autoreleasepool {
        NSApplication* app = [NSApplication sharedApplication];
        if ( [NSWindow respondsToSelector:@selector ( setAllowsAutomaticWindowTabbing: )] ) {
            [NSWindow setAllowsAutomaticWindowTabbing:NO];
        }
        [app setActivationPolicy:NSApplicationActivationPolicyRegular];

        AppDelegate* delegate = [[AppDelegate alloc]
                                 initWithFilename:(argc > 1) ? argv[1] : nullptr
                                 width:800
                                 height:600];
        [app setDelegate:delegate];
        [app run];
        [delegate release];
    }

    AeonGUI::Finalize();
    return 0;
}
