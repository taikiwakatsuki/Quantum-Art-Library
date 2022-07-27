#version 400 core

in vec2 position;
in vec4 color;
out vec4 outColor;

void main(void) {
    outColor = color;
    gl_Position = vec4(position, 0.1, 1.0);
    gl_PointSize = 1.0;
}