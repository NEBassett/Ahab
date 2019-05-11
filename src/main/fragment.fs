#version 450 core

out vec4 color;

in vec4 simPos;

uniform float time;
uniform sampler2D density;

void main()
{

  //color = vec4(texture(density, simPos.zw).xy, 0, 1);
  color = pow(abs(texture(density, simPos.zw).xxxx - 0.1), vec4(4));
}
