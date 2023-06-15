import numpy as np


class Particle:
    def __init__(
        self,
        x_t0: np.float = 0,
        y_t0: np.float = 0,
        v_x: np.float = 0,
        v_y: np.float = 0,
        mass: np.float = 1,
    ):
        self.x = x_t0
        self.y = y_t0
        self.x_trail = np.array(x_t0)
        self.y_trail = np.array(y_t0)
        self.v_x = v_x
        self.v_y = v_y
        self.mass = mass


class Object:
    def __init__(
        self, x: np.float = 0, y: np.float = 0, mass: np.float = 1,
    ):
        self.x = x
        self.y = y
        self.mass = mass


def updateParticle(
    particle: Particle,
    objects: list,
    G: np.float,
    dt: np.float,
    r_modifier=1.0,
    c=0.0,
    power=2,
    gravityCutoff=10_000,
    cos_sin: "cos or sin or None" = "cos",
    yGrid=False,
):
    def getAcceleration(
        particle: Particle, object: Object, G: np.float, dt: np.float,
    ):
        # x distance
        dx = object.x - particle.x
        # y distance
        dy = object.y - particle.y

        # distance squared
        rsq = dx ** 2 + dy ** 2
        # distance
        r = np.sqrt(rsq) * r_modifier

        if rsq < 1e-1000:
            # return a_x = 0, a_y = 0
            return 0, 0
        else:
            # start calculating acceleration
            a = G * object.mass / rsq
            if cos_sin == "cos":
                a *= np.cos(r - c / dt) ** (power)
            elif cos_sin == "sin":
                a *= np.sin(r - c / dt) ** (power)
            a_x = a * dx / r
            a_y = a * dy / r
            return a_x, a_y

    a_x = a_y = 0
    if particle.y < gravityCutoff:
        for object in objects:
            da_x, da_y = getAcceleration(particle, object, G, dt)
            a_x += da_x
            a_y += da_y

        dv_y = a_y * dt
        dv_x = a_x * dt
        particle.v_y += dv_y
        particle.v_x += dv_x
    particle.y_trail = np.append(particle.y_trail, particle.y)
    particle.x_trail = np.append(particle.x_trail, particle.x)
    if (yGrid == True) & (particle.v_y != 0.0):
        dt = abs(1 / particle.v_y)
        particle.y += np.round(particle.v_y * dt, 0)
    else:
        particle.y += particle.v_y * dt
    particle.x += particle.v_x * dt


# TODO : rewrite method
def f(x, y, edges, mult=1, v=10, trig="sin", power=2, wave_length=1):
    sum_ = 0
    for edge in edges:
        x_ = x + edge
        y_ = y
        newton_grav = mult * (1 / (x_ ** 2 + y_ ** 2))
        if trig == "sin":
            trig_ = np.sin(wave_length * (np.sqrt(x_ ** 2 + y_ ** 2) - v)) ** power
        elif trig == "cos":
            trig_ = np.cos(wave_length * (np.sqrt(x_ ** 2 + y_ ** 2) - v)) ** power
        else:
            trig_ = 1
        sum_ += newton_grav * trig_ ** 1

    return sum_
