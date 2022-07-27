from OpenGL.GL import *
import numpy as np

def create_program(vertex_shader_src, fragment_shader_src):
    vertex_shader = glCreateShader(GL_VERTEX_SHADER)
    glShaderSource(vertex_shader, vertex_shader_src)
    glCompileShader(vertex_shader)
    result = glGetShaderiv(vertex_shader, GL_COMPILE_STATUS)
    if not result:
        err_str = glGetShaderInfoLog(vertex_shader).decode('utf-8')
        raise RuntimeError(err_str)

    fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
    glShaderSource(fragment_shader, fragment_shader_src)
    glCompileShader(fragment_shader)
    result = glGetShaderiv(fragment_shader, GL_COMPILE_STATUS)
    if not result:
        err_str = glGetShaderInfoLog(fragment_shader).decode('utf-8')
        raise RuntimeError(err_str)

    program = glCreateProgram()
    glAttachShader(program, vertex_shader)
    glDeleteShader(vertex_shader)
    glAttachShader(program, fragment_shader)
    glDeleteShader(fragment_shader)
    glLinkProgram(program)
    result = glGetProgramiv(program, GL_LINK_STATUS)
    if not result:
        err_str = glGetProgramInfoLog(program).decode('utf-8')
        raise RuntimeError(err_str)

    return program

def create_vbo(vertex):
    if not isinstance(vertex, np.ndarray):
        vertex = np.array(vertex, dtype=np.float32)
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertex.nbytes, vertex, GL_STATIC_DRAW)
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    return vbo

def create_ibo(vert_index):
    if not isinstance(vert_index, np.ndarray):
        vert_index = np.array(vert_index, dtype=np.uint32)
    ibo = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, vert_index.nbytes, vert_index, GL_STATIC_DRAW)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
    return ibo
