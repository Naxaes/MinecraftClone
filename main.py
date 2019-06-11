import numpy
import os
from pyglet.gl import *
from pyglet.window import Window, mouse, key
from ctypes import cast, pointer, POINTER, byref, sizeof, create_string_buffer, c_char, c_float, c_uint
from mathematics import Vector2D, Vector3D, cos, sin, tan, pi, radians
from collections import namedtuple

# ALL ANGLES ARE IN RADIANS!

window = Window(width=480, height=480)
key_state = []

Block = namedtuple('Block', 'transformation')
Light = namedtuple('Light', 'transformation, color, attenuation')
AABB  = namedtuple('AABB',  'position, size')


class Shader:

    bound = None  # This is okay if we assume we're only going to need one OpenGL context.

    def __init__(self, sources, attributes, uniforms):
        self.id, self.uniform = create_shader_program(
            (bytes(x, 'utf-8') for x in sources),
            (bytes(x, 'utf-8') for x in attributes),
            (bytes(x, 'utf-8') for x in uniforms)
        )

    def enable(self):
        glUseProgram(self.id)
        Shader.bound = self  # Just for safety.

    @staticmethod
    def disable():
        glUseProgram(0)
        Shader.bound = None

    def load_uniform_matrix(self, name, data):
        assert Shader.bound is self, "Must bind this shader before being able to load uniform."
        name = bytes(name, 'utf-8')
        glUniformMatrix4fv(self.uniform[name], 1, GL_TRUE, data.ctypes.data_as(POINTER(GLfloat)))

    def load_uniform_floats(self, name, data):
        assert Shader.bound is self, "Must bind this shader before being able to load uniform."
        name = bytes(name, 'utf-8')
        if isinstance(data, (float, int)):
            glUniform1f(self.uniform[name], data)
        else:
            functions = glUniform2f, glUniform3f, glUniform4f
            functions[len(data)-2](self.uniform[name], *data)


class VBO:

    def __init__(self, id_, dimension):
        self.id = id_
        self.dimension = dimension
        self.type = GL_FLOAT

    def enable(self, index):
        glBindBuffer(GL_ARRAY_BUFFER, self.id)
        glEnableVertexAttribArray(index)
        glVertexAttribPointer(index, self.dimension, self.type, GL_FALSE, 0, 0)


class Model:

    bound = None  # This is okay if we assume we're only going to need one OpenGL context.

    def __init__(self, vbos, indexed_vbo, count):
        self.vbos = vbos
        self.index_vbo = indexed_vbo
        self.count = count

    def enable(self):
        for index, vbo in enumerate(self.vbos):
            vbo.enable(index=index)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.index_vbo)
        Model.bound = self  # Safety.

    def disable(self):
        for index in range(len(self.vbos)):
            glDisableVertexAttribArray(index)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
        Model.bound = None

    def render(self):
        assert Model.bound is self, "Model isn't bound."
        glDrawElements(GL_TRIANGLES, self.count, GL_UNSIGNED_INT, 0)


class Transform:

    def __init__(self, location=(0, 0, 0), rotation=(0, 0, 0), scale=(1, 1, 1)):
        self.location = Vector3D(*location)
        self.rotation = Vector3D(*rotation)
        self.scale    = Vector3D(*scale)

    def matrix(self):
        return create_transformation_matrix(
            *self.location, *self.rotation, *self.scale
        )


class Font:

    def __init__(self, path):
        self.characters = {}
        texture_path = None

        for line in open(path):
            if line.startswith('page '):
                for statement in line.split(' '):
                    if statement.startswith('file'):
                        texture_path = statement.split('=')[-1].replace('"', '', 2).replace('\n', '')
            elif line.startswith('common '):
                pass
            elif line.startswith('char '):
                character_attributes = {}
                for statement in line.split(' '):
                    if '=' in statement:
                        attribute, value = statement.split('=')
                        character_attributes[attribute] = int(value)

                        if 'id=' in statement:
                            self.characters[int(value)] = character_attributes

        assert texture_path, 'Could not find the texture!'
        folder_path = os.path.split(path)[0]
        self.texture = load_texture(os.path.join(folder_path, texture_path))

    def create_text_quad(self, text, anchor_center=False):

        positions = []
        texture_coordinates = []
        indices = []

        cursor_x, cursor_y = 0, 0
        index = 0

        height = self.texture.height

        for character in text:
            info = self.characters[ord(character)]
            tx, ty = info['x'], info['y']
            tw, th = info['width'], info['height']
            x,  y  = cursor_x + info['xoffset'], cursor_y - info['yoffset']

            v = [
                x     , y,        # topleft
                x     , y - th,   # bottomleft
                x + tw, y - th,   # bottomright
                x + tw, y,        # topright
            ]
            t = [
                tx     , height - ty,        # topleft
                tx     , height - (ty + th), # bottomleft
                tx + tw, height - (ty + th), # bottomright
                tx + tw, height - ty         # topright
            ]
            i = [index, index + 1, index + 3, index + 3, index + 1, index + 2]

            positions.extend(v)
            texture_coordinates.extend(t)
            indices.extend(i)

            index += 4

            cursor_x += info['xadvance']

        # Normalize
        max_value = max((self.texture.height, self.texture.width))

        if anchor_center:
            width = cursor_x
            offset = (width / 2) / max_value
            positions = [i / max_value - offset for i in positions]
        else:
            positions = [i / max_value for i in positions]

        texture_coordinates = [i / max_value for i in texture_coordinates]

        return positions, texture_coordinates, indices


class Camera:

    def __init__(self, position, fov=90,
                 maximum_render_distance=1000.0, minimum_render_distance=0.1):

        self.position  = Vector3D(*position)
        self.fov = fov
        self.maximum_render_distance = maximum_render_distance
        self.minimum_render_distance = minimum_render_distance

        self.right = Vector3D(1, 0, 0)
        self.up    = Vector3D(0, 1, 0)
        self.front = Vector3D(0, 0, -1)

        # self.function = function

    def rotate(self, yaw=0.0, pitch=0.0, roll=0.0):
        pass

    def move(self, forward=0, right=0, up=0):
        self.position += (self.right * right + self.up * up + self.front * forward)

    def update(self, *args, **kwargs):
        # if self.function:
        pass

    def transformation_matrix(self):
        return create_transformation_matrix(*self.position, 0, 0, 0, 1, 1, 1)

    def perspective_matrix(self):
        return create_perspective_matrix(self.fov, window.width / window.height, self.minimum_render_distance, self.maximum_render_distance)


def follow(target, distance, offset=(0, 0, 0)):
    def func(camera):
        camera.position = target.position + Vector3D(*offset) - target.direction * distance
        camera.rotation = target.rotation
    return func


def create_transformation_matrix(x, y, z, rx, ry, rz, sx, sy, sz):
    # TODO optimize by creating the transformation matrix directly.
    translation = numpy.array(
        ((1, 0, 0, x),
         (0, 1, 0, y),
         (0, 0, 1, z),
         (0, 0, 0, 1)), dtype=GLfloat
    )

    rotation_x = numpy.array(
        ((1, 0, 0, 0),
         (0, cos(rx), -sin(rx), 0),
         (0, sin(rx), cos(rx), 0),
         (0, 0, 0, 1)), dtype=GLfloat
    )

    rotation_y = numpy.array(
        ((cos(ry), 0, sin(ry), 0),
         (0, 1, 0, 0),
         (-sin(ry), 0, cos(ry), 0),
         (0, 0, 0, 1)), dtype=GLfloat
    )

    rotation_z = numpy.array(
        ((cos(rz), -sin(rz), 0, 0),
         (sin(rz), cos(rz), 0, 0),
         (0, 0, 1, 0),
         (0, 0, 0, 1)), dtype=GLfloat
    )

    scale = numpy.array(
        ((sx, 0, 0, 0),
         (0, sy, 0, 0),
         (0, 0, sz, 0),
         (0, 0, 0, 1)), dtype=GLfloat
    )

    return translation @ rotation_x @ rotation_y @ rotation_z @ scale


def create_orthographic_matrix(left, right, bottom, top, near, far):
    a = 2 * near
    b = right - left
    c = top - bottom
    d = far - near

    return numpy.array(
        (
            (a / b, 0, (right + left) / b, 0),
            (0, a / c, (top + bottom) / c, 0),
            (0, 0, -(far + near) / d, -(2 * d) / d),
            (0, 0, -1, 0)
        ), dtype=GLfloat
    )


def create_perspective_matrix(fov, aspect_ratio, near, far):
    # TODO optimize by creating the transformation matrix directly.
    top = near * tan((pi / 180) * (fov / 2))
    bottom = -top
    right = top * aspect_ratio
    left = -right

    return create_orthographic_matrix(left, right, bottom, top, near, far)


def create_shader_program(sources, attributes, uniforms):
    shader_handles = []
    for i, source in enumerate(sources):
        handle = glCreateShader(GL_VERTEX_SHADER if i == 0 else GL_FRAGMENT_SHADER)
        glShaderSource(handle, 1, cast(pointer(pointer(create_string_buffer(source))), POINTER(POINTER(c_char))), None)
        glCompileShader(handle)
        shader_handles.append(handle)

    # Create attributes.
    attribute_mapping = []
    for attribute in attributes:
        attribute_mapping.append(create_string_buffer(attribute))

    try:
        # Create program.
        program_handle = glCreateProgram()
        glAttachShader(program_handle, shader_handles[0])
        glAttachShader(program_handle, shader_handles[1])
        for index, name in enumerate(attributes):  # CHANGED
            glBindAttribLocation(program_handle, index, name)
        glLinkProgram(program_handle)
        glValidateProgram(program_handle)
        glUseProgram(program_handle)
    except pyglet.gl.GLException:
        # Print errors.
        status = GLint()
        glGetShaderiv(shader_handles[0], GL_INFO_LOG_LENGTH, byref(status))
        output = create_string_buffer(status.value)
        glGetShaderInfoLog(shader_handles[0], status, None, output)
        print(output.value.decode('utf-8'))
        status = GLint()
        glGetShaderiv(shader_handles[1], GL_INFO_LOG_LENGTH, byref(status))
        output = create_string_buffer(status.value)
        glGetShaderInfoLog(shader_handles[1], status, None, output)
        print(output.value.decode('utf-8'))
        status = GLint()
        glGetProgramiv(program_handle, GL_INFO_LOG_LENGTH,
                       byref(status))  # Getting the number of char in info log to 'status'
        output = create_string_buffer(status.value)  # status.value)
        glGetProgramInfoLog(program_handle, status, None, output)
        print(output.value.decode('utf-8'))

    # Get uniform location.
    uniform_mapping = {}
    for uniform in uniforms:
        name = create_string_buffer(uniform)
        location = glGetUniformLocation(program_handle, cast(pointer(name), POINTER(c_char)))
        uniform_mapping[uniform] = location

    return program_handle, uniform_mapping


def create_model(vertices, texture_coordinates, normals, indices):
    # Create and bind vbo.
    vbo = c_uint()
    glGenBuffers(1, vbo)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, len(vertices) * sizeof(c_float), (c_float * len(vertices))(*vertices), GL_STATIC_DRAW)

    # Associate vertex attribute 0 (position) with the bound vbo above.
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0)

    glBindBuffer(GL_ARRAY_BUFFER, 0)

    # Create and bind vbo. CHANGED (REMOVED color vbo and replaced it with texture)
    texture_coordinates_vbo = c_uint()
    glGenBuffers(1, texture_coordinates_vbo)
    glBindBuffer(GL_ARRAY_BUFFER, texture_coordinates_vbo)
    glBufferData(GL_ARRAY_BUFFER, len(texture_coordinates) * sizeof(c_float),
                 (c_float * len(texture_coordinates))(*texture_coordinates), GL_STATIC_DRAW)

    # Associate vertex attribute 2 with the bound vbo (our texture_coordinates vbo) above.
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, 0)  # CHANGED (to 2 instead of 3).

    glBindBuffer(GL_ARRAY_BUFFER, 0)

    # Create and bind vbo.
    normal_vbo = c_uint()
    glGenBuffers(1, normal_vbo)
    glBindBuffer(GL_ARRAY_BUFFER, normal_vbo)
    glBufferData(GL_ARRAY_BUFFER, len(normals) * sizeof(c_float), (c_float * len(vertices))(*vertices), GL_STATIC_DRAW)

    # Associate vertex attribute 0 (position) with the bound vbo above.
    glEnableVertexAttribArray(2)
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, 0)

    glBindBuffer(GL_ARRAY_BUFFER, 0)

    # Create and bind indexed vbo.
    indexed_vbo = c_uint()
    glGenBuffers(1, indexed_vbo)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexed_vbo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, len(indices) * sizeof(c_uint), (c_uint * len(indices))(*indices),
                 GL_STATIC_DRAW)

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

    # CHANGED
    vbos = VBO(vbo, 3), VBO(texture_coordinates_vbo, 2), VBO(normal_vbo, 3)
    return Model(vbos=vbos, indexed_vbo=indexed_vbo, count=len(indices))

    # return vbo, texture_coordinates_vbo, normal_vbo, indexed_vbo, len(indices)


def create_cube():
    vertices = [
        -0.5,  0.5, -0.5, -0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5, -0.5,  # Top.
        -0.5, -0.5, -0.5,  0.5, -0.5, -0.5,  0.5, -0.5,  0.5, -0.5, -0.5,  0.5,  # Bottom.
        -0.5, -0.5, -0.5, -0.5, -0.5,  0.5, -0.5,  0.5,  0.5, -0.5,  0.5, -0.5,  # Left.
         0.5, -0.5,  0.5,  0.5, -0.5, -0.5,  0.5,  0.5, -0.5,  0.5,  0.5,  0.5,  # Right.
        -0.5, -0.5,  0.5,  0.5, -0.5,  0.5,  0.5,  0.5,  0.5, -0.5,  0.5,  0.5,  # Front.
         0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5,  0.5, -0.5,  0.5,  0.5, -0.5,  # Back.
    ]
    
    texture_coordinates = [
        0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0,  # Top.
        0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0,  # Bottom.
        0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0,  # Left.
        0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0,  # Right.
        0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0,  # Front.
        0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0,  # Back.
    ]

    normals = [
         0.0,  1.0,  0.0,  0.0,  1.0,  0.0,  0.0,  1.0,  0.0,  0.0,  1.0,  0.0,  # Top.
         0.0, -1.0,  0.0,  0.0, -1.0,  0.0,  0.0, -1.0,  0.0,  0.0, -1.0,  0.0,  # Bottom.
        -1.0,  0.0,  0.0, -1.0,  0.0,  0.0, -1.0,  0.0,  0.0, -1.0,  0.0,  0.0,  # Left.
         1.0,  0.0,  0.0,  1.0,  0.0,  0.0,  1.0,  0.0,  0.0,  1.0,  0.0,  0.0,  # Right.
         0.0,  0.0,  1.0,  0.0,  0.0,  1.0,  0.0,  0.0,  1.0,  0.0,  0.0,  1.0,  # Front.
         0.0,  0.0, -1.0,  0.0,  0.0, -1.0,  0.0,  0.0, -1.0,  0.0,  0.0, -1.0,  # Back.
    ]

    indices = [
        0,  1,  3,  3,  1,  2,
        4,  5,  7,  7,  5,  6,
        8,  9,  11, 11, 9,  10,
        12, 13, 15, 15, 13, 14,
        16, 17, 19, 19, 17, 18,
        20, 21, 23, 23, 21, 22
    ]

    return create_model(vertices, texture_coordinates, normals, indices)


def create_2D_object(vertices, texture_coordinates, indices):
    # Create and bind vbo.
    vbo = c_uint()
    glGenBuffers(1, vbo)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, len(vertices) * sizeof(c_float), (c_float * len(vertices))(*vertices), GL_STATIC_DRAW)

    # Associate vertex attribute 0 (position) with the bound vbo above.
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0)

    glBindBuffer(GL_ARRAY_BUFFER, 0)

    # Create and bind vbo.
    texture_coordinates_vbo = c_uint()
    glGenBuffers(1, texture_coordinates_vbo)
    glBindBuffer(GL_ARRAY_BUFFER, texture_coordinates_vbo)
    glBufferData(GL_ARRAY_BUFFER, len(texture_coordinates) * sizeof(c_float),
                 (c_float * len(texture_coordinates))(*texture_coordinates), GL_STATIC_DRAW)

    # Associate vertex attribute 2 with the bound vbo (our texture_coordinates vbo) above.
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, 0)

    glBindBuffer(GL_ARRAY_BUFFER, 0)

    # Create and bind indexed vbo.
    indexed_vbo = c_uint()
    glGenBuffers(1, indexed_vbo)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexed_vbo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, len(indices) * sizeof(c_uint), (c_uint * len(indices))(*indices),
                 GL_STATIC_DRAW)

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

    vbos = VBO(vbo, 2), VBO(texture_coordinates_vbo, 2)
    return Model(vbos=vbos, indexed_vbo=indexed_vbo, count=len(indices))


def load_model(path):
    vertices = []
    normals = []
    texture_coordinates = []

    with open(path) as text_file:  # TODO Is it streamed? Does it matter?

        line = text_file.readline()

        while True:

            if line.startswith('v '):
                _, x, y, z = line.split()
                vertices.append((float(x), float(y), float(z)))
            elif line.startswith('vt '):
                _, u, t = line.split()
                texture_coordinates.append((float(u), float(t)))
            elif line.startswith('vn '):
                _, x, y, z = line.split()
                normals.append((float(x), float(y), float(z)))
            elif line.startswith('f ') or line == '':
                break

            line = text_file.readline()

        indices = []
        unique_data = []
        sorted_vertices = []
        sorted_normals = []
        sorted_texture_coordinates = []
        index = 0

        while line != '':
            if not line.startswith('f'):
                line = text_file.readline()
                continue

            _, *faces = line.split(' ')

            for face in faces:
                vertex = face.split('/')
                if vertex not in unique_data:
                    unique_data.append(vertex)
                    indices.append(index)
                    vertex_index, texture_index, normal_index = vertex
                    vertex_index, texture_index, normal_index = int(vertex_index), int(texture_index), int(normal_index)
                    sorted_vertices.extend(vertices[vertex_index - 1])
                    sorted_normals.extend(normals[normal_index - 1])
                    sorted_texture_coordinates.extend(texture_coordinates[texture_index - 1])
                    index += 1
                else:
                    indices.append(unique_data.index(vertex))

            line = text_file.readline()

    return create_model(sorted_vertices, sorted_texture_coordinates, sorted_normals, indices)


def load_texture(path, min_filter=GL_LINEAR, max_filter=GL_LINEAR, wrap_s=GL_CLAMP_TO_EDGE, wrap_t=GL_CLAMP_TO_EDGE):
    texture = pyglet.image.load(path).get_texture()  # DIMENSIONS MUST BE POWER OF 2.
    glBindTexture(GL_TEXTURE_2D, texture.id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, min_filter)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, max_filter)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrap_s)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrap_t)
    glBindTexture(GL_TEXTURE_2D, 0)
    return texture


@window.event
def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
    entity = all_entities[entity_selected]

    if buttons == mouse.LEFT:
        entity.transformation.location[0] += dx / 250
        entity.transformation.location[1] += dy / 250
    elif buttons == mouse.MIDDLE:  # Scroll button.
        entity.transformation.scale[0] += dx / 250
        entity.transformation.scale[1] += dy / 250
    elif buttons == mouse.RIGHT:
        entity.transformation.rotation[1] += dx / 250
        entity.transformation.rotation[0] -= dy / 250


@window.event
def on_mouse_scroll(x, y, scroll_x, scroll_y):
    entity = all_entities[entity_selected]
    entity.transformation.location[2] -= scroll_y / 10
    entity.transformation.location[0] += scroll_x / 10


@window.event
def on_key_press(symbol, modifiers):
    global entity_selected
    if symbol == key.LEFT:
        entity_selected = (entity_selected - 1) % len(all_entities)
    elif symbol == key.RIGHT:
        entity_selected = (entity_selected + 1) % len(all_entities)

    if symbol in (key.A, key.D, key.LSHIFT, key.SPACE, key.W, key.S, key.Q, key.E):
        key_state.append(symbol)


@window.event
def on_key_release(symbol, modifier):
    if symbol in (key.A, key.D, key.LSHIFT, key.SPACE, key.W, key.S, key.Q, key.E):
        key_state.remove(symbol)


@window.event
def on_draw():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_CULL_FACE)
    glCullFace(GL_BACK)

    # Setup light shader
    light_program.enable()
    light_program.load_uniform_matrix(name='perspective', data=camera.transformation_matrix())
    light_program.load_uniform_matrix(name='view', data=camera.transformation_matrix())


    # Render lights
    model = models[CUBE]
    model.enable()

    for light in lights:
        light_program.load_uniform_matrix(name='transformation', data=light.transformation.matrix())
        light_program.load_uniform_floats(name='color', data=light.color)

        model.render()


    # Setup block shader
    program.enable()
    program.load_uniform_matrix(name='perspective', data=camera.perspective_matrix())
    program.load_uniform_matrix(name='view', data=camera.transformation_matrix())

    # Upload lights to block shader
    for i, light in enumerate(lights):
        program.load_uniform_floats(name='light[{}].position'.format(i),  data=light.transformation.location)
        program.load_uniform_floats(name='light[{}].color'.format(i),     data=light.color)
        program.load_uniform_floats(name='light[{}].constant'.format(i),  data=light.attenuation[0])
        program.load_uniform_floats(name='light[{}].linear'.format(i),    data=light.attenuation[1])
        program.load_uniform_floats(name='light[{}].quadratic'.format(i), data=light.attenuation[2])

    glActiveTexture(GL_TEXTURE0)

    # Render blocks
    # Make bindings for texture.
    spritesheet = textures[SPRITE_SHEET_TEXTURE]
    glBindTexture(GL_TEXTURE_2D, spritesheet.id)

    model = models[CUBE]
    model.enable()

    for i, block in enumerate(blocks):
        # Prepare entities of specific model and texture, and draw.
        texture_index = i % 16                 # 4 x 4 texture atlas
        column = (texture_index % 4)  / 4
        row    = (texture_index // 4) / 4

        program.load_uniform_matrix(name='transformation', data=block.transformation.matrix())
        program.load_uniform_floats(name='texture_offset', data=(column, row))
        model.render()


    # Render text
    glDisable(GL_DEPTH_TEST)
    glDisable(GL_CULL_FACE)

    text = create_2D_object(
        *font_arial.create_text_quad(
            "FPS: {:.2f}".format(float(pyglet.clock.get_fps())),
            anchor_center=True)
    )

    simple_2D_program.enable()
    simple_2D_program.load_uniform_matrix(name='perspective', data=camera.perspective_matrix())
    simple_2D_program.load_uniform_matrix(name='view', data=camera.transformation_matrix())

    simple_2D_program.load_uniform_matrix(name='transformation', data=text_transform.matrix())
    simple_2D_program.load_uniform_floats(name='color', data=(255, 255, 255))

    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, font_arial.texture.id)

    text.enable()
    text.render()


def update(dt):

    speed = 10 * dt

    for symbol in key_state:

        if symbol == key.A:
            camera.move(right=-speed)
        elif symbol == key.D:
            camera.move(right=speed)
        elif symbol == key.LSHIFT:
            camera.move(up=-speed)
        elif symbol == key.SPACE:
            camera.move(up=speed)
        elif symbol == key.W:
            camera.move(forward=speed)
        elif symbol == key.S:
            camera.move(forward=-speed)

        elif symbol == key.Q:
            camera.rotate(yaw=-speed)
        elif symbol == key.E:
            camera.rotate(yaw=speed)


if __name__ == '__main__':

    # Create shaders.
    block_shader_sources = [
        """
        #version 120
    
        uniform mat4 transformation;
        uniform mat4 perspective;
        uniform mat4 view;
            
        uniform vec2 texture_offset;
        
        attribute vec3 position;
        attribute vec3 normal;
        attribute vec2 texture_coordinate;
            
        varying vec3 out_position;
        varying vec3 out_normal;
        varying vec2 out_texture_coordinate;
    
        void main()
        {
            // Vector should have 1.0 as w-component so the transformation matrix affects it properly, while directions
            // should have 0.0 as w-component so the transformation matrix doesn't affect it's location.
            // Since position is a vector, it should have 1.0 as w-component.
            // Since normal is a direction, it should have 0.0 as w-component.
    
            vec4 full_position = vec4(position, 1.0);
            vec4 full_normal   = vec4(normal, 0.0);
    
            vec4 world_position = transformation * full_position;
            vec4 world_normal   = transformation * full_normal;
    
            out_position = vec3(world_position);
            out_normal   = normalize(vec3(world_normal));
            out_texture_coordinate = (texture_coordinate / 4) + texture_offset;
    
            gl_Position =  perspective * view * world_position;
        }
    
        """,
        """
        #version 120
    
        /* 
        Naming conventions:
            * vector    - A vector that might not be normalized.
            * direction - A vector that must be normalized.
    
            All vectors/directions are in relation to the object vertex if nothing else is specified. 
            For example: 
                'normal' is the normal of the vertex.
                'vector_to_light' is a non-normalized vector pointing from the vertex to the light.
                'direction_camera_to_light' is a normalized vector pointing from the camera to the light.
        */
    
        struct Light {
            vec3  position;
            vec3  color;
    
            float constant;
            float linear;
            float quadratic;
        };
    
        const int NUM_LIGHTS = 1;
    
        uniform sampler2D texture;
        uniform Light light[NUM_LIGHTS];
        uniform mat4  view;
            
        varying vec3 out_position;
        varying vec3 out_normal;
        varying vec2 out_texture_coordinate;
    
        void main()
        {
            vec3 camera_position = -view[3].xyz;
            vec4 light_color = vec4(0.0, 0.0, 0.0, 0.0);
            float attenuation = 0.0;
    
            for (int i = 0; i < NUM_LIGHTS; i++) {
                vec3 vector_to_light = light[i].position - out_position;
    
                vec3  direction_to_light = normalize(vector_to_light);
                float distance_to_light  = length(vector_to_light);
    
                float angle_normal_and_light = dot(out_normal, direction_to_light);
                float diffuse_factor = clamp(angle_normal_and_light, 0.0, 1.0);        // or max(angle_normal_and_light, 1.0)
                attenuation += 1.0 / (
                    light[i].constant + light[i].linear * distance_to_light + light[i].quadratic * distance_to_light * distance_to_light
                    );
    
                light_color += vec4(light[i].color * diffuse_factor, 1.0);
            }
    
            vec4 color = texture2D(texture, out_texture_coordinate);
    
            vec4 ambient = color * 0.2;
            vec4 diffuse = color * light_color * attenuation;
    
            gl_FragColor = ambient + diffuse;
        }
    
        """
    ]
    light_shader_sources = [
        """
        #version 120
    
        uniform mat4 transformation;
        uniform mat4 perspective;
        uniform mat4 view;
        uniform vec3 color;
    
        attribute vec3 position;
    
        void main()
        {    
            gl_Position =  perspective * view * transformation * vec4(position, 1.0);
        }
    
        """,
        """
        #version 120
    
        uniform mat4 transformation;
        uniform mat4 perspective;
        uniform mat4 view;
        uniform vec3 color;
    
        void main()
        {    
            gl_FragColor = vec4(color, 1.0);
        }
    
        """
    ]
    text_shader_sources  = [
        """
        #version 120

        uniform mat4 transformation;
        uniform mat4 perspective;
        uniform mat4 view;

        attribute vec2 position;
        attribute vec2 texture_coordinate;

        varying vec2 out_texture_coordinate;

        void main()
        {    
            gl_Position =  /* perspective * view */ transformation * vec4(position, 0.0, 1.0);
            out_texture_coordinate = texture_coordinate;
        }

        """,
        """
        #version 120

        varying vec2 out_texture_coordinate;

        uniform vec3 color;
        uniform sampler2D font_texture;

        void main()
        {    
            gl_FragColor = texture2D(font_texture, out_texture_coordinate).a * vec4(color, 1.0);
        }

        """
    ]


    program = Shader(
        sources=block_shader_sources,
        attributes=['position', 'texture_coordinate', 'normal'],
        uniforms=['transformation', 'perspective', 'view', 'texture_offset',
                *['light[{}].{}'.format(i, attribute) for attribute in
                 ['position', 'color', 'intensity', 'constant', 'linear', 'quadratic'] for i in range(4)]
        ]
    )

    light_program = Shader(
        sources=light_shader_sources,
        attributes=['position'],
        uniforms=['transformation', 'perspective', 'view', 'color']
    )

    simple_2D_program = Shader(
        sources=text_shader_sources,
        attributes=['position'],
        uniforms=['transformation', 'perspective', 'view', 'color']
    )


    font_arial = Font('fonts/arial.fnt')
    text_transform = Transform(location=(-1, 1, 0), rotation=(0, 0, 0), scale=(0.6, 0.6, 0.6))

    CUBE   = 0
    SPHERE = 1
    models = [create_cube()]

    SPRITE_SHEET_TEXTURE = 0
    textures = [load_texture('texture.png', min_filter=GL_NEAREST, max_filter=GL_NEAREST)]

    blocks = [
        Block(Transform(location=(x, 0, z))) for x in range(-5, 5, 1) for z in range(-5, 5, 1)
    ]
    lights = [Light(transformation=Transform((0, 5, 0)), color=(0.5, 1.0, 1.0), attenuation=(1.0, 0.009, 0.032))]

    entity_selected = 0
    all_entities = [*blocks, *lights]

    camera = Camera(position=(0, 1, 0))

    pyglet.clock.schedule(update)
    pyglet.app.run()
