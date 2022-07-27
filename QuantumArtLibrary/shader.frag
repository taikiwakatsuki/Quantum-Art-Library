#version 400 core

in vec4 outColor;
out vec4 outFragmentColor;

void main(void) {
    outFragmentColor = outColor;
}