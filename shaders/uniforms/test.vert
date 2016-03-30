#version 400
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout (std140, binding = 0) uniform bufferVals {
    mat4 mv;
    mat4 p;
    vec3 lightpos;
} myBufferVals;

layout (location = 0) in vec4 pos;
layout (location = 1) in vec4 inColor;
layout (location = 0) out vec4 outColor;

out gl_PerVertex { 
    vec4 gl_Position;
};

void main() {
   outColor = inColor;
   mat4 mvp = myBufferVals.p * myBufferVals.mv;
   gl_Position =  mvp * pos;

   // GL->VK conventions
   gl_Position.y = -gl_Position.y;
   gl_Position.z = (gl_Position.z + gl_Position.w) / 2.0;
}
