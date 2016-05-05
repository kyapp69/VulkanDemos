#version 400
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout (set = 0, binding = 0) uniform bufferVals {
    mat4 model;
} modelBufferVals;

layout (std140, set = 0, binding = 1) uniform bufferVals1 {
    vec3 lightDirection;
    mat4 view;
    mat4 projection;
} sceneBufferVals;

layout (location = 0) in vec4 pos;
layout (location = 1) in vec4 inColor;
layout (location = 2) in vec3 normal;

layout (location = 0) out vec4 outColor;

out gl_PerVertex { 
    vec4 gl_Position;
};

void main() {

   mat3 modelmatrix3 = mat3(modelBufferVals.model[0].xyz, modelBufferVals.model[1].xyz, modelBufferVals.model[2].xyz);
   vec3 normalWS = normalize(modelmatrix3 * vec3(normal));
   float cosLightAngle = max(dot(normalWS, sceneBufferVals.lightDirection), 0.0);
   outColor = inColor * cosLightAngle * 0.7 + inColor * 0.3;
   mat4 mvp = sceneBufferVals.projection * sceneBufferVals.view * modelBufferVals.model;
   gl_Position = mvp * pos;

   // GL->VK conventions
   gl_Position.y = -gl_Position.y;
   gl_Position.z = (gl_Position.z + gl_Position.w) / 2.0;
}
