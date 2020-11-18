/*
Copyright (C) 2019,2020 Rodrigo Jose Hernandez Cordoba

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
#ifndef COMMON_H
#define COMMON_H
#define OPENGL_CHECK_ERROR \
 { \
     if (int glError = glGetError()) \
     { \
         const char* error_string = (glError == GL_INVALID_ENUM) ? "GL_INVALID_ENUM" : \
             (glError == GL_INVALID_VALUE) ? "GL_INVALID_VALUE" : \
             (glError == GL_INVALID_OPERATION) ? "GL_INVALID_OPERATION" : \
             (glError == GL_STACK_OVERFLOW) ? "GL_STACK_OVERFLOW" : \
             (glError == GL_STACK_UNDERFLOW) ? "GL_STACK_UNDERFLOW" : \
             (glError == GL_OUT_OF_MEMORY) ? "GL_OUT_OF_MEMORY" : "Unknown Error Code"; \
         std::ostringstream stream; \
         stream << "OpenGL Error " << error_string << " (Code " << glError << " ) " << __FILE__ << ":" << __LINE__; \
         std::cout << stream.str() << std::endl; \
     } \
 }

#define GLDEFINEFUNCTION(glFunctionType,glFunction) glFunctionType glFunction = nullptr

const GLchar vertex_shader_code[] =
    R"(#version 450 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aTexCoords;

out vec2 Pos;
out vec2 TexCoords;

void main()
{
    gl_Position = vec4(aPos.x, aPos.y, 0.0, 1.0); 
    Pos = aPos;
    TexCoords = aTexCoords;
}
)";
const GLint vertex_shader_len { sizeof(vertex_shader_code) /*/ sizeof(vertex_shader_code[0])*/};
const GLchar* const vertex_shader_code_ptr = vertex_shader_code;

const GLchar fragment_shader_code[] =
R"(#version 450 core
out vec4 FragColor;
  
in vec2 Pos;
in vec2 TexCoords;

layout (location = 0) uniform sampler2D screenTexture;

void main()
{ 
    FragColor = texture(screenTexture, TexCoords);
}
)";
const GLint fragment_shader_len { sizeof(fragment_shader_code) /*/ sizeof(fragment_shader_code[0])*/};
const GLchar* const fragment_shader_code_ptr = fragment_shader_code;

const float vertices[] = {  
    // positions   // texCoords
    -1.0f,  1.0f,  0.0f, 0.0f,
    -1.0f, -1.0f,  0.0f, 1.0f,
    1.0f, -1.0f,  1.0f, 1.0f,
    1.0f,  1.0f,  1.0f, 0.0f
};
const GLuint vertex_size{sizeof(vertices)};
#endif