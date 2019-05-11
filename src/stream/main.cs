#version 430 core

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

uniform int dim;

uniform sampler2D boundary;
layout(binding = 0, r32f) readonly uniform image2D old;
layout(binding = 1, r32f) writeonly uniform image2D new;

void main()
{
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

    ivec2 ind = ivec2(gl_GlobalInvocationID.x, gl_GlobalInvocationID.y*9);  // multiply y by 9 since each node is 9 floats long
    vec2 fInd = vec2(gl_GlobalInvocationID.xy+vec2(0.5))/(dim);
    float texel = 1.0/dim;

    for(int i = 0; i < 9; i++)
    {
      if(texture(boundary, fInd + dirs[i]*texel).x > 0) // boundary in streaming direction
      {
                                                                                        // bounce back by
        imageStore(new, ind + ivec2(0, 8 - i), imageLoad(old, ind + ivec2(0,i)).xxxx);  // replacing this nodes inverse direction distribution with this one
      } else {
        // replace node along this directions distribution along this direction with this distribution
        imageStore(new, ind + ivec2(dirs[i])*ivec2(1,9) + ivec2(0,i), imageLoad(old, ind + ivec2(0,i)).xxxx);
      }
    }
}
