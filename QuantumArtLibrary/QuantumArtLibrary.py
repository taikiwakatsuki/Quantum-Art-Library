import numpy as np
import matplotlib.pyplot as plt
from .gl_util import *
from PIL import Image
import progressbar
import time
from OpenGL.GL import *
import glfw
import cv2
import os


# parameter
Å = 1.8897261246257702
extent = 40 * Å
femtoseconds = 4.134137333518212 * 10.
simulation_time = 0.4 * femtoseconds
m_e = 1.0
hbar = 1.0
potcol = 0.15


program = None
pos_vbo = None
color_vbo = None
pos_loc = -1
color_loc = -1


def road_shader(vertex_shader, fragment_shader):
    global vertex_shader_src, fragment_shader_src

    if (vertex_shader != None) and (fragment_shader != None):
        with open(vertex_shader, "r", encoding="utf-8") as v:
            vertex_shader_src = v.read()

        with open(fragment_shader, "r", encoding="utf-8") as f:
            fragment_shader_src = f.read()
    elif (vertex_shader == None) and (fragment_shader == None):
        vertex_shader_src = """
        #version 400 core

        in vec2 position;
        in vec4 color;
        out vec4 outColor;

        void main(void) {
            outColor = color;
            gl_Position = vec4(position, 0.1, 1.0);
            gl_PointSize = 1.0;
        }
        """.strip()

        fragment_shader_src = """
        #version 400 core

        in vec4 outColor;
        out vec4 outFragmentColor;

        void main(void) {
            outFragmentColor = outColor;
        }
        """.strip()


def simulation(image="DoubleSlit", n=400, step=300, p=[0, -15], v=[0, 80], method="split-step", white=1, range=[0.1, 0.9]):
    global N, N_range, total_frames, gray_range, wb, init_pos, velocity

    N = n
    N_range = N + 2
    total_frames = step
    gray_range = range
    wb = white
    init_pos = p
    velocity = v
    img = image_adjustment(image)

    if method == "split-step":
        return splitstep(img, N_range=N_range, total_frames=total_frames)
    elif method == "split-step-cupy":
        return splitstepcupy(img, N_range=N_range, total_frames=total_frames)


def particle():
    x = np.linspace(-extent / 2, extent / 2, N_range)
    y = np.linspace(-extent / 2, extent / 2, N_range)
    x, y = np.meshgrid(x, y)

    return x, y


def wavefunction():
    particle_x, particle_y = particle()
    σ = 1.0 * Å
    vx0 = velocity[0] * Å / femtoseconds
    vy0 = velocity[1] * Å / femtoseconds

    return np.exp(-1 / (4 * σ ** 2) * ((particle_x - init_pos[0] * Å) ** 2 + (particle_y - init_pos[1] * Å) ** 2)) / np.sqrt(2 * np.pi * σ ** 2) * np.exp(vx0 * particle_x * 1j + vy0 * particle_y * 1j)


def compute_momentum_space():
    px = np.linspace(
        -np.pi * N_range // 2 / (extent / 2) * hbar,
        np.pi * N_range // 2 / (extent / 2) * hbar,
        N_range,
    )
    py = np.linspace(
        -np.pi * N_range // 2 / (extent / 2) * hbar,
        np.pi * N_range // 2 / (extent / 2) * hbar,
        N_range,
    )
    px, py = np.meshgrid(px, py)

    return px ** 2 + py ** 2


def potential(image, frame=None):
    global potential_pos

    particle_x, particle_y = particle()
    
    # Image Import potential
    if str(image) != "DoubleSlit":
        potential_value = 20
        min = image.min()
        max = image.max()
        if wb == 1:
            potpos = np.abs(((image - min) / (max - min)))
        else:
            potpos = np.abs(1 - ((image - min) / (max - min)))

        potpos = np.where(potpos < gray_range[0], gray_range[0], np.where(potpos > gray_range[1], gray_range[1], potpos))
        min = potpos.min()
        max = potpos.max()
        potpos = np.abs((potpos - min) / (max - min))
        potential_pos = potential_value * potpos
    else:
        # Double Slit
        potential_value = 5
        b = 2.0* Å # slits separation
        a = 0.5* Å # slits width
        d = 0.5* Å # slits depth
        potential_pos = np.where( ((particle_x < - b/2 - a) | (particle_x > b/2 + a) | ((particle_x > -b/2) & (particle_x < b/2))) & ((particle_y < d/2) & (particle_y > -d/2) ), potential_value, 0)

    # Non potential
    # self.p_alpha = 0
    # potential_value = 0
    # potential_pos = np.zeros((N, N))
    # potpos = potential_pos

    return potential_pos


def splitstep(image, N_range, total_frames):
    dt = simulation_time / total_frames
    Nt = int(np.round(dt / (simulation_time / 10000)))
    simulation_dt = dt / Nt
    p2 = np.array(compute_momentum_space())

    Ψ = np.zeros((total_frames + 1, * ([N_range] * 2)), dtype=np.complex128)
    Ψ[0] = np.array(wavefunction())

    m = m_e

    Ur = -0.5j*(simulation_dt / hbar)
    Uk = np.exp(-0.5j*(simulation_dt / (m * hbar)) * p2)

    t0 = time.time()
    bar = progressbar.ProgressBar()
    frame = 0
    for i in bar(range(total_frames)):
        Ur_V = np.exp(Ur * np.array(potential(image, frame)))
        tmp = np.copy(Ψ[i])
        for j in range(Nt):
            c = np.fft.fftshift(np.fft.fftn(Ur_V * tmp))
            tmp = Ur_V * np.fft.ifftn(np.fft.ifftshift(Uk * c))
        tmp[0] = 0
        tmp[-1] = 0
        tmp[:][0] = 0
        tmp[:][-1] = 0
        Ψ[i+1] = tmp
        frame += 1
    print(time.time() - t0)

    simulation_Ψ = Ψ
    simulation_Ψmax = np.amax(np.abs(Ψ))

    return simulation_Ψ / simulation_Ψmax


def splitstepcupy(image, N_range, total_frames):
    import cupy as cp

    dt = simulation_time / total_frames
    Nt = int(cp.round(dt / (simulation_time / 10000)))
    simulation_dt = dt / Nt
    p2 = cp.array(compute_momentum_space())

    Ψ = cp.zeros((total_frames + 1, * ([N_range]) * 2), dtype=cp.complex128)
    Ψ[0] = cp.array(wavefunction())

    m = m_e

    Ur = -0.5j*(simulation_dt / hbar)
    Uk = cp.exp(-0.5j*(simulation_dt / (m * hbar)) * p2)

    t0 = time.time()
    bar = progressbar.ProgressBar()
    frame = 0
    for i in bar(range(total_frames)):
        Ur_V = cp.exp(Ur * cp.array(potential(image, frame)))
        tmp = cp.copy(Ψ[i])
        for j in range(Nt):
            c = cp.fft.fftshift(cp.fft.fftn(Ur_V * tmp))
            tmp = Ur_V * np.fft.ifftn(cp.fft.ifftshift(Uk * c))
        tmp[0] = 0
        tmp[-1] = 0
        tmp[:][0] = 0
        tmp[:][-1] = 0
        Ψ[i+1] = tmp
        frame += 1
    print(time.time() - t0)

    simulation_Ψ = Ψ.get()
    simulation_Ψmax = np.amax(np.abs(simulation_Ψ))

    return simulation_Ψ / simulation_Ψmax


def complex_to_rgba(Z: np.ndarray, max_val: float = 1.0) -> np.ndarray:

    # cmap = plt.cm.twilight
    # cmap = plt.cm.cividis
    cmap = plt.cm.pink
    data = np.abs(Z)**2
    norm = plt.Normalize(data.min(), data.max())
    rgb_map = cmap(norm(data))
    rgb_map = rgb_map[:, :, :3]

    abs_z = np.abs(Z)/ max_val
    abs_z = np.where(abs_z > 1.0, 1.0 ,abs_z)
    return np.concatenate((rgb_map, abs_z.reshape((*abs_z.shape, 1))), axis=(abs_z.ndim))


def init_draw(window, width, height, simulation):
    global program, pos_vbo, color_vbo, pos_loc, color_loc, vert_pos, vert_color

    result = complex_to_rgba(simulation[0])
    
    # position
    pos_x = np.linspace(-1, 1, N)
    pos_y = np.linspace(-1, 1, N)
    vert_pos = np.stack(np.meshgrid(pos_x, pos_y), axis=-1).reshape((N)**2, 2)
    vert_pos = np.concatenate([vert_pos, vert_pos])
    vert_pos = np.array(vert_pos, dtype=np.float32)

    # color
    result = result[1:N+1, 1:N+1]
    vert_color = result.reshape(N**2, 4)
    potential_rgb = np.array([[potcol, potcol, potcol]]*(N**2))
    potential_alpha = potential_pos[1:N+1, 1:N+1]
    potential_color = np.block([potential_rgb, potential_alpha.reshape(N**2, 1)])
    vert_color = np.concatenate([vert_color, potential_color])
    vert_color = np.array(vert_color, dtype=np.float32)

    # OpenGL
    program = create_program(vertex_shader_src, fragment_shader_src)
    pos_loc = glGetAttribLocation(program, "position")
    color_loc = glGetAttribLocation(program, "color")
    pos_vbo = create_vbo(vert_pos)
    color_vbo = create_vbo(vert_color)


def update(simulation, frame):
    global program, pos_vbo, color_vbo, pos_loc, color_loc, vert_pos, vert_color

    # color adjustment
    result = complex_to_rgba(simulation[frame])

    # color
    result = result[1:N+1, 1:N+1]
    vert_color = result.reshape(N**2, 4)
    potential_rgb = np.array([[potcol, potcol, potcol]]*(N**2))
    potential_alpha = potential_pos[1:N+1, 1:N+1]
    potential_color = np.block([potential_rgb, potential_alpha.reshape(N**2, 1)])
    vert_color = np.concatenate([vert_color, potential_color])
    vert_color = np.array(vert_color, dtype=np.float32)

    # OpenGL
    program = create_program(vertex_shader_src, fragment_shader_src)
    color_loc = glGetAttribLocation(program, "color")
    color_vbo = create_vbo(vert_color)


def draw():
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glUseProgram(program)
    glEnableVertexAttribArray(pos_loc)
    glBindBuffer(GL_ARRAY_BUFFER, pos_vbo)
    glVertexAttribPointer(pos_loc, 2, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(color_loc)
    glBindBuffer(GL_ARRAY_BUFFER, color_vbo)
    glVertexAttribPointer(color_loc, 4, GL_FLOAT, GL_FALSE, 0, None)
    glDrawArrays(GL_POINTS, 0, N**2*2)
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glUseProgram(0)


def image_directory():
    global image_dir

    image_dir = "images/image1"
    i = 2 
    if os.path.exists(image_dir):
        while True:
            new_name = "images/" + "image" + str(i)
            if not os.path.exists(new_name):
                os.makedirs(new_name)
                image_dir = new_name
                return
            i += 1
    else:
        os.makedirs(image_dir)


def save(frame):
    size_x, size_y, width, height = glGetDoublev(GL_VIEWPORT)
    width, height = int(width), int(height)
    glPixelStorei(GL_PACK_ALIGNMENT, 1)
    data = glReadPixels(size_x, size_y, width, height, GL_RGB, GL_UNSIGNED_BYTE)
    image = Image.frombytes("RGB", (width, height), data)
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    image.save(image_dir + "/" + str(frame).zfill(4) + ".png")


def image_adjustment(image):
    if str(image) != "DoubleSlit":
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        img = cv2.flip(img, 0)
        img2 = cv2.resize(img, (N_range, N_range))
        img_color = np.empty((N_range, N_range))
        img_color[:] = img2
        image = img_color

    return image


def movie_rename(file_name):
    i = 2
    if os.path.exists("movies/" + file_name + ".mp4"):
        while True:
            new_name = "movie" + str(i)
            if not os.path.exists("movies/" + new_name + ".mp4"):
                return new_name
            i += 1
    else:
        return file_name


def movie(fps):
    dir = "movies"
    if not os.path.exists(dir):
        os.makedirs(dir)

    file_name = movie_rename("movie1")

    encoder = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    video = cv2.VideoWriter(dir + "/" + file_name + ".mp4", encoder, fps, (N, N))

    for i in range(0, total_frames):
        img = cv2.imread(image_dir + "/%04d.png" % i)
        video.write(img)
    
    video.release()


def drawing(simulation, vertex_shader=None, fragment_shader=None):
    road_shader(vertex_shader, fragment_shader)
    image_directory()

    SCREEN_WIDTH = N
    SCREEN_HEIGHT = N

    if not glfw.init():
        return
    
    window = glfw.create_window(SCREEN_WIDTH, SCREEN_HEIGHT, "Quantum Art", None, None)
    if not window:
        glfw.terminate()
        print("Failed to create window")
        return
    
    glfw.make_context_current(window)

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 0)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    init_draw(window, SCREEN_WIDTH, SCREEN_HEIGHT, simulation)

    frame = 0

    while (not glfw.window_should_close(window)) and (total_frames > frame):
        glClearColor(0, 0, 0, 0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        draw()
        save(frame)
        frame += 1
        update(simulation, frame)

        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.destroy_window(window)
    glfw.terminate()

    movie(30)
