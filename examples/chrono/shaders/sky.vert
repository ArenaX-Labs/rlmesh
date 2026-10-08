#version 450

layout(location = 0) out vec2 ndc;

// One triangle covering the screen, at the far plane.
void main() {
  ndc = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2) * 2.0 - 1.0;
  gl_Position = vec4(ndc, 1.0, 1.0);
}
