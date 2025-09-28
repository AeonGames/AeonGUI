/*
Copyright (C) 2025 Rodrigo Jose Hernandez Cordoba

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
#include <assert.h>
#include <time.h>
#include <unistd.h>
#include <iostream>
#include <sstream>
#include <mach/mach_time.h>

#define GL_SILENCE_DEPRECATION
#include <OpenGL/gl3.h>
#include <Cocoa/Cocoa.h>

#include "aeongui/AeonGUI.hpp"
#include "aeongui/dom/Window.hpp"
#include "aeongui/dom/Document.hpp"
#include "Common.h"

@interface OpenGLView : NSOpenGLView
{
@private
    NSTrackingArea* trackingArea;
    AeonGUI::DOM::Window* mWindow;
    GLuint mProgram;
    GLuint mVAO;
    GLuint mScreenQuad;
    GLuint mScreenTexture;
    NSTimer* renderTimer;
    uint64_t lastTime;
}

- (id)initWithFrame:(NSRect)frameRect window:(AeonGUI::DOM::Window*)window;
- (void)setupOpenGL;
- (void)render:(NSTimer*)timer;
- (void)cleanup;

@end

@implementation OpenGLView

- (id)initWithFrame:(NSRect)frameRect window:(AeonGUI::DOM::Window*)window
{
    NSOpenGLPixelFormatAttribute attrs[] = {
        NSOpenGLPFADoubleBuffer,
        NSOpenGLPFADepthSize, 24,
        NSOpenGLPFAStencilSize, 8,
        NSOpenGLPFAOpenGLProfile, NSOpenGLProfileVersion4_1Core,
        0
    };
    
    NSOpenGLPixelFormat* format = [[NSOpenGLPixelFormat alloc] initWithAttributes:attrs];
    
    if (!format) {
        NSLog(@"Failed to create OpenGL pixel format");
        return nil;
    }
    
    self = [super initWithFrame:frameRect pixelFormat:format];
    [format release];
    
    if (self) {
        mWindow = window;
        mProgram = 0;
        mVAO = 0;
        mScreenQuad = 0;
        mScreenTexture = 0;
        renderTimer = nil;
        lastTime = mach_absolute_time();
        
        trackingArea = [[NSTrackingArea alloc] 
                       initWithRect:[self bounds]
                       options:(NSTrackingActiveAlways | NSTrackingMouseMoved)
                       owner:self
                       userInfo:nil];
        [self addTrackingArea:trackingArea];
    }
    
    return self;
}

- (void)dealloc
{
    [self cleanup];
    if (trackingArea) {
        [self removeTrackingArea:trackingArea];
        [trackingArea release];
    }
    [super dealloc];
}

- (void)prepareOpenGL
{
    [super prepareOpenGL];
    [self setupOpenGL];
    
    // Start render timer
    renderTimer = [NSTimer scheduledTimerWithTimeInterval:1.0/60.0
                                                   target:self
                                                 selector:@selector(render:)
                                                 userInfo:nil
                                                  repeats:YES];
}

- (void)setupOpenGL
{
    [[self openGLContext] makeCurrentContext];
    
    // Enable VSync
    GLint swapInt = 1;
    [[self openGLContext] setValues:&swapInt forParameter:NSOpenGLContextParameterSwapInterval];
    
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glViewport(0, 0, (GLsizei)self.bounds.size.width, (GLsizei)self.bounds.size.height);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    OPENGL_CHECK_ERROR;
    glEnable(GL_BLEND);
    OPENGL_CHECK_ERROR;
    
    // Create shader program
    GLint compile_status;
    mProgram = glCreateProgram();
    OPENGL_CHECK_ERROR;
    
    GLuint vertex_shader = glCreateShader(GL_VERTEX_SHADER);
    OPENGL_CHECK_ERROR;
    glShaderSource(vertex_shader, 1, &vertex_shader_code_ptr, &vertex_shader_len);
    OPENGL_CHECK_ERROR;
    glCompileShader(vertex_shader);
    OPENGL_CHECK_ERROR;
    glGetShaderiv(vertex_shader, GL_COMPILE_STATUS, &compile_status);
    OPENGL_CHECK_ERROR;
    
    if (compile_status != GL_TRUE) {
        GLint log_length;
        glGetShaderiv(vertex_shader, GL_INFO_LOG_LENGTH, &log_length);
        if (log_length > 0) {
            GLchar* log = (GLchar*)malloc(log_length);
            glGetShaderInfoLog(vertex_shader, log_length, &log_length, log);
            NSLog(@"Vertex shader compilation failed: %s", log);
            free(log);
        }
    }
    
    glAttachShader(mProgram, vertex_shader);
    OPENGL_CHECK_ERROR;
    
    GLuint fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
    OPENGL_CHECK_ERROR;
    glShaderSource(fragment_shader, 1, &fragment_shader_code_ptr, &fragment_shader_len);
    OPENGL_CHECK_ERROR;
    glCompileShader(fragment_shader);
    OPENGL_CHECK_ERROR;
    glGetShaderiv(fragment_shader, GL_COMPILE_STATUS, &compile_status);
    OPENGL_CHECK_ERROR;
    
    if (compile_status != GL_TRUE) {
        GLint log_length;
        glGetShaderiv(fragment_shader, GL_INFO_LOG_LENGTH, &log_length);
        if (log_length > 0) {
            GLchar* log = (GLchar*)malloc(log_length);
            glGetShaderInfoLog(fragment_shader, log_length, &log_length, log);
            NSLog(@"Fragment shader compilation failed: %s", log);
            free(log);
        }
    }
    
    glAttachShader(mProgram, fragment_shader);
    OPENGL_CHECK_ERROR;
    
    glLinkProgram(mProgram);
    OPENGL_CHECK_ERROR;
    
    glDetachShader(mProgram, vertex_shader);
    OPENGL_CHECK_ERROR;
    glDetachShader(mProgram, fragment_shader);
    OPENGL_CHECK_ERROR;
    glDeleteShader(vertex_shader);
    OPENGL_CHECK_ERROR;
    glDeleteShader(fragment_shader);
    OPENGL_CHECK_ERROR;
    
    glUseProgram(mProgram);
    OPENGL_CHECK_ERROR;
    
    GLint texture_location = glGetUniformLocation(mProgram, "screenTexture");
    glUniform1i(texture_location, 0);
    OPENGL_CHECK_ERROR;
    
    // Setup vertex array and buffer
    glGenVertexArrays(1, &mVAO);
    OPENGL_CHECK_ERROR;
    glBindVertexArray(mVAO);
    OPENGL_CHECK_ERROR;
    
    glGenBuffers(1, &mScreenQuad);
    OPENGL_CHECK_ERROR;
    glBindBuffer(GL_ARRAY_BUFFER, mScreenQuad);
    OPENGL_CHECK_ERROR;
    glBufferData(GL_ARRAY_BUFFER, vertex_size, vertices, GL_STATIC_DRAW);
    OPENGL_CHECK_ERROR;
    
    glEnableVertexAttribArray(0);
    OPENGL_CHECK_ERROR;
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 4, (void*)0);
    OPENGL_CHECK_ERROR;
    
    glEnableVertexAttribArray(1);
    OPENGL_CHECK_ERROR;
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 4, (void*)(sizeof(float) * 2));
    OPENGL_CHECK_ERROR;
    
    // Setup texture
    glGenTextures(1, &mScreenTexture);
    OPENGL_CHECK_ERROR;
    glBindTexture(GL_TEXTURE_2D, mScreenTexture);
    OPENGL_CHECK_ERROR;
    glTexImage2D(GL_TEXTURE_2D,
                 0,
                 GL_RGBA,
                 (GLsizei)mWindow->GetWidth(),
                 (GLsizei)mWindow->GetHeight(),
                 0,
                 GL_BGRA,
                 GL_UNSIGNED_INT_8_8_8_8_REV,
                 mWindow->GetPixels());
    OPENGL_CHECK_ERROR;
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    OPENGL_CHECK_ERROR;
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    OPENGL_CHECK_ERROR;
    glActiveTexture(GL_TEXTURE0);
    OPENGL_CHECK_ERROR;
}

- (void)render:(NSTimer*)timer
{
    [[self openGLContext] makeCurrentContext];
    
    uint64_t currentTime = mach_absolute_time();
    static mach_timebase_info_data_t timebase;
    if (timebase.denom == 0) {
        mach_timebase_info(&timebase);
    }
    
    float delta = (float)(currentTime - lastTime) * timebase.numer / timebase.denom / 1e9f;
    if (delta > 0.1f) {
        delta = 1.0f / 30.0f;
    }
    lastTime = currentTime;
    
    mWindow->Draw();
    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    OPENGL_CHECK_ERROR;
    
    glUseProgram(mProgram);
    OPENGL_CHECK_ERROR;
    glBindVertexArray(mVAO);
    OPENGL_CHECK_ERROR;
    
    glBindTexture(GL_TEXTURE_2D, mScreenTexture);
    OPENGL_CHECK_ERROR;
    glTexSubImage2D(GL_TEXTURE_2D,
                    0,
                    0,
                    0,
                    (GLsizei)mWindow->GetWidth(),
                    (GLsizei)mWindow->GetHeight(),
                    GL_BGRA,
                    GL_UNSIGNED_INT_8_8_8_8_REV,
                    mWindow->GetPixels());
    OPENGL_CHECK_ERROR;
    
    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
    OPENGL_CHECK_ERROR;
    
    [[self openGLContext] flushBuffer];
}

- (void)cleanup
{
    if (renderTimer) {
        [renderTimer invalidate];
        renderTimer = nil;
    }
    
    [[self openGLContext] makeCurrentContext];
    
    if (glIsTexture(mScreenTexture)) {
        glDeleteTextures(1, &mScreenTexture);
        mScreenTexture = 0;
    }
    
    if (glIsBuffer(mScreenQuad)) {
        glDeleteBuffers(1, &mScreenQuad);
        mScreenQuad = 0;
    }
    
    if (glIsVertexArray(mVAO)) {
        glDeleteVertexArrays(1, &mVAO);
        mVAO = 0;
    }
    
    if (glIsProgram(mProgram)) {
        glDeleteProgram(mProgram);
        mProgram = 0;
    }
}

- (void)reshape
{
    [super reshape];
    [[self openGLContext] makeCurrentContext];
    
    NSRect bounds = [self bounds];
    glViewport(0, 0, (GLsizei)bounds.size.width, (GLsizei)bounds.size.height);
    OPENGL_CHECK_ERROR;
    
    mWindow->ResizeViewport((size_t)bounds.size.width, (size_t)bounds.size.height);
    
    glBindTexture(GL_TEXTURE_2D, mScreenTexture);
    glTexImage2D(GL_TEXTURE_2D,
                 0,
                 GL_RGBA,
                 (GLsizei)mWindow->GetWidth(),
                 (GLsizei)mWindow->GetHeight(),
                 0,
                 GL_BGRA,
                 GL_UNSIGNED_INT_8_8_8_8_REV,
                 mWindow->GetPixels());
    OPENGL_CHECK_ERROR;
}

- (void)mouseDown:(NSEvent*)event
{
    NSPoint location = [self convertPoint:[event locationInWindow] fromView:nil];
    (void)location;
    // Handle mouse down event
}

- (void)mouseUp:(NSEvent*)event
{
    NSPoint location = [self convertPoint:[event locationInWindow] fromView:nil];
    (void)location;
    // Handle mouse up event
}

- (void)mouseMoved:(NSEvent*)event
{
    NSPoint location = [self convertPoint:[event locationInWindow] fromView:nil];
    (void)location;
    // Handle mouse move event
}

- (void)keyDown:(NSEvent*)event
{
    // Handle key down event
}

- (void)keyUp:(NSEvent*)event
{
    // Handle key up event
}

@end

@interface AppDelegate : NSObject <NSApplicationDelegate>
{
@private
    NSWindow* window;
    OpenGLView* openGLView;
    AeonGUI::DOM::Window* aeonWindow;
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
        
        NSRect frame = NSMakeRect(0, 0, width, height);
        window = [[NSWindow alloc] 
                 initWithContentRect:frame
                 styleMask:(NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskResizable)
                 backing:NSBackingStoreBuffered
                 defer:NO];
        
        [window setTitle:@"AeonGUI"];
        [window center];
        
        openGLView = [[OpenGLView alloc] initWithFrame:frame window:aeonWindow];
        [window setContentView:openGLView];
        
        [window makeKeyAndOrderFront:nil];
    }
    return self;
}

- (void)dealloc
{
    delete aeonWindow;
    [openGLView release];
    [window release];
    [super dealloc];
}

- (void)applicationDidFinishLaunching:(NSNotification*)notification
{
    // Application setup complete
}

- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication*)app
{
    return YES;
}

- (void)applicationWillTerminate:(NSNotification*)notification
{
    [openGLView cleanup];
}

@end

int main(int argc, char** argv)
{
    AeonGUI::Initialize(argc, argv);
    
    @autoreleasepool {
        NSApplication* app = [NSApplication sharedApplication];
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
