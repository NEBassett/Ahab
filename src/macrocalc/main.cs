#version 430 core

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0, r32f) readonly uniform image2D nodes;
layout(binding = 1, rg32f) writeonly uniform image2D macvels;
layout(binding = 2, r32f) writeonly uniform image2D macdens;

uniform float c; // lattice speed
uniform float tau;
uniform float time;

void main()
{
    ivec2 ind = ivec2(gl_GlobalInvocationID.x, gl_GlobalInvocationID.y*9);  // multiply y by 9 since each node is 9 floats long

    vec2 macvel = vec2(0);
    float macden = 0;

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

    for(int i = 0; i < 9; i++)
    {
      macvel = macvel + dirs[i]*imageLoad(nodes, ivec2(ind + vec2(0,i))).x;
      macden = macden + imageLoad(nodes, ivec2(ind + vec2(0,i))).x;
    }

    float div = macden;
    if(div <= 0 || length(div) < 0.01)
    {
      div = 1;
    }

    macvel = c*macvel/div + abs(vec2(cos(time)*cos(time),sin(time)))*1.3*tau*(1/(1 + length(vec2(200, 200) - gl_GlobalInvocationID.xy)))/div;

    if(length(macvel) > 10)
    {
      macvel = normalize(macvel) * (10);
    }

    imageStore(macvels, ivec2(gl_GlobalInvocationID.xy), macvel.xyxy);
    imageStore(macdens, ivec2(gl_GlobalInvocationID.xy), macden.xxxx);
}
