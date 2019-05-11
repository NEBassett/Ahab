#version 430 core

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0, r32f) writeonly uniform image2D new;

void main()
{
    float defaultDist[9] = float[9](1.0f/9.0f, 1.0f/36.0f, 1.0f/36.0f,1.0f/9.0f,4.0f/9.0f, 1.0f/9.0f, 1.0f/36.0f, 1.0f/36.0f,1.0f/9.0f);
    ivec2 ind = ivec2(gl_GlobalInvocationID.x, gl_GlobalInvocationID.y);
    imageStore(new, ind, vec4(defaultDist[ind.y%9]));
}
