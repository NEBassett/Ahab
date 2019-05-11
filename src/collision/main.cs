#version 430 core

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

uniform float c;
uniform float tau;

layout(binding = 0, r32f) readonly uniform image2D nodes;
layout(binding = 2, r32f) readonly uniform image2D macden;
layout(binding = 1, rg32f) readonly uniform image2D macvel;
layout(binding = 3, r32f) writeonly uniform image2D newnodes;

void main()
{
    float defaultDist[9];// = float[9](1.0f/9.0f, 1.0f/36.0f, 1.0f/36.0f,1.0f/9.0f,4.0f/9.0f, 1.0f/9.0f, 1.0f/36.0f, 1.0f/36.0f,1.0f/9.0f);
    defaultDist[0] = 1.0f/9.0f;
    defaultDist[8] = 1.0f/9.0f;
    defaultDist[1] = 1.0f/36.0f;
    defaultDist[7] = 1.0f/36.0f;
    defaultDist[2] = 1.0f/36.0f;
    defaultDist[6] = 1.0f/36.0f;
    defaultDist[3] = 1.0f/9.0f;
    defaultDist[5] = 1.0f/9.0f;
    defaultDist[4] = 4.0f/9.0f;


    vec2 dirs[9];
    dirs[0] = vec2(1,0);
    dirs[8] = vec2(-1,0);
    dirs[1] = vec2(1,1);
    dirs[7] = vec2(-1,-1);
    dirs[2] = vec2(1,-1);
    dirs[6] = vec2(-1,1);
    dirs[3] = vec2(0,1);
    dirs[5] = vec2(0,-1);
    dirs[4] = vec2(0,0);

    ivec2 ind = ivec2(gl_GlobalInvocationID.x, gl_GlobalInvocationID.y);
    ivec2 ind2 = ivec2(gl_GlobalInvocationID.x, int(gl_GlobalInvocationID.y)/9);

    float density = imageLoad(macden, ind2).x;
    vec2 vel = imageLoad(macvel, ind2).xy;
    vec2 dir = dirs[ind.y%9];
    float w_i = defaultDist[ind.y%9];
    float nodeval = imageLoad(nodes, ind).x;
    float prod = dot(dir,vel);

    //float equi = density*(w_i*(1 + 3*prod/c + 9*prod*prod/(2*c*c) - 3*dot(vel, vel)/(2*c*c)));
    float equi = density * w_i + density * (w_i * ( 3 * prod / c + 9 * prod * prod / (c * c * 2) - 3 * dot(vel, vel) / (c * c * 2) ) );
    imageStore(newnodes, ind, vec4(nodeval - (nodeval - equi)/tau));
}
