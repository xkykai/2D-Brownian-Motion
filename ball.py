
"""
A ball module for 2D rigid disc collision simulation.
The balls collide elastically with one another and the wall of the container.

Xin Kai Lee 10/3/2020
"""
import numpy as np
from copy import deepcopy


class Ball:
    """
    Ball Class. This is used as the ball in a 2D rigid disc collision simulation.

    Attributes:
        mass (float): The mass of the ball.
        radius (float): The radius of the ball.
        pos (numpy.ndarray of float): The position of the ball.
        vel (numpy.ndarray of float): The velocity of the ball.
        count (int): The number of collisions the ball experienced.
    """

    time = 0

    def __init__(
        self,
        mass=1,
        radius=1,
        pos=np.array([0.0, 0.0]),
        vel=np.array([0.0, 0.0]),
        count=0,
    ):
        self._mass = mass
        self._radius = radius
        self._pos = pos
        self._vel = vel
        self._count = count
        self._dp = 0

    def __repr__(self):
        return f"Ball radius = {self._radius}, position = {self._pos}, velocity = {self._vel}, mass = {self._mass}"

    def __str__(self):
        return f"Ball, r = {self._radius}, pos = {self._pos}, vel = {self._vel}, mass = {self._mass}"

    def mass(self):
        """
        Returns:
            (float): The mass of the ball.
        """
        return self._mass

    def radius(self):
        """
        Returns:
            (float): The radius of the ball.
        """
        return self._radius

    def pos(self):
        """
        Returns:
            (numpy.ndarray of float): The position array of the ball.
        """
        return self._pos

    def vel(self):
        """
        Returns:
            (numpy.ndarray of float): The velocity array of the ball.
        """
        return self._vel

    def set_mass(self, mass):
        """
        Sets the mass of the ball.

        Parameters:
            mass (float): The mass of the ball.
        
        """
        self._mass = mass

    def set_radius(self, radius):
        """
        Sets the radius of the ball.

        Parameters:
            radius (float): The radius of the ball.
        
        """
        self._radius = radius

    def set_pos(self, pos):
        """
        Sets the position of the ball.

        Parameters:
            pos (numpy.ndarray of float): The position of the ball.
        
        """
        self._pos = np.array(pos)

    def set_vel(self, vel):
        """
        Sets the velocity of the ball.

        Parameters:
            vel (numpy.ndarray of float): The velocity of the ball.
        """
        self._vel = np.array(vel)

    def copy(self):
        """
        Performs a deepcopy on the ball.

        Returns:
            (Ball): The same Ball object with itself.
        """
        return deepcopy(self)

    def time_to_collision(self, other):
        """
        Calculates the next collision time between balls or with the container. If they don't collide, np.inf is returned.

        Parameters:
            other (Ball): The other ball/container to be collided with.
        
        Returns:
            dt (float): The next collision time.
        """
        r = self._pos - other._pos
        v = self._vel - other._vel

        a = magsquare_vector(v)

        if a == 0:  # If both balls move in same direction
            return np.inf

        else:  # Conditions for collision with balls or container
            if isinstance(other, Container):
                R = self._radius - other._radius
            else:
                R = self._radius + other._radius

            b = 2 * np.dot(v, r)
            c = magsquare_vector(r) - R ** 2

            if not isinstance(other, Container):  # Floating point errors
                if np.abs(c) <= 10e-13:
                    return np.inf

            discriminant = b ** 2 - 4 * a * c

            if discriminant < 0:  # Complex dt
                return np.inf
            elif discriminant == 0:
                dt = -b / (2 * a)
                if dt <= 0:
                    return np.inf
                else:
                    return dt
            elif discriminant > 0:
                dt1 = (-b + np.sqrt(discriminant)) / (2 * a)
                dt2 = (-b - np.sqrt(discriminant)) / (2 * a)
                if isinstance(other, Container):
                    return np.amax(np.array([dt1, dt2]))
                else:
                    if dt1 <= 0 and dt2 <= 0:
                        return np.inf
                    elif dt1 > 0 and dt2 > 0:
                        return np.amin(np.array([dt1, dt2]))
                    elif dt1 < 0:
                        return dt2
                    elif dt2 < 0:
                        return dt1

    def collide(self, other):
        """
        Calculates the change in velocity of balls after a collision.

        Arguments:
            other (Ball): The other ball which is collided.
        """

        # Transforming velocity to centre-of-mass frame
        r = self._pos - other._pos
        u1_parallel = projection(self._vel, r)  # parallel to line of c.o.m
        u1_perpendicular = self._vel - u1_parallel

        if isinstance(other, Container):  # Colliding with a container
            vel_i = self._vel
            v1_parallel = -u1_parallel
            v2_parallel = np.zeros(2)
            v1_perpendicular = u1_perpendicular
            self.set_vel(v1_parallel + v1_perpendicular)
            other.set_vel(np.zeros(2))
            self._count += 1
            other._count = -1  # Collision count for container is always -1
            vel_f = self._vel
            dv = vel_f - vel_i
            self._dp = dv * self._mass

        else:  # Colliding with another ball
            m1 = self._mass
            m2 = other._mass

            u2_parallel = projection(other._vel, r)
            u2_perpendicular = other._vel - u2_parallel

            u1_parallel_translated = u1_parallel - u2_parallel

            v1_parallel_translated = (m1 - m2) / (m1 + m2) * u1_parallel_translated

            v1_parallel = v1_parallel_translated + u2_parallel
            v2_parallel = u1_parallel - u2_parallel + v1_parallel
            v1_perpendicular = u1_perpendicular
            v2_perpendicular = u2_perpendicular
            self.set_vel(v1_parallel + v1_perpendicular)
            other.set_vel(v2_parallel + v2_perpendicular)
            self._count += 1
            other._count += 1

    def move(self, dt):
        """
        Moving the ball to another time.

        Arguments:
            dt (float): The time from now to move the ball to.
        """
        self.set_pos(self._pos + dt * self._vel)


class Container(Ball):
    """
    This is a Container class. The container is used to enclose the balls in
    the 2D rigid disc collision.

    Arguments:
            radius (float): The radius of the container.
            mass (float): The mass of the container.
    """

    def __init__(self, radius=10, mass=100):
        super().__init__()
        self._radius = radius
        self._mass = mass
        self._count = -1  # The collision count is always -1


def magsquare_vector(vector):
    """
    Calculates the magnitude squared of a vector.

    Arguments:
        vector (numpy.ndarray of float): Vector.
    
    Returns:
        (float): The magnitude squared of the vector.
    """
    return np.dot(vector, vector)


def mag_vector(vector):
    """
    Calculates the magnitude of a vector.

    Arguments:
        vector (numpy.ndarray of float): Vector.
    
    Returns:
        (float): The magnitude of the vector.
    """
    return np.sqrt(magsquare_vector(vector))


def projection(a, b):
    """
    Calculates the projection vector of a on b.

    Arguments:
        a (numpy.ndarray of float): Vector to find projection of.
        b (numpy.ndarray of float): Vector to be projected on.
    
    Returns:
        (numpy.ndarray of float): The projection vector of a on b.
    """
    magsquare_b = magsquare_vector(b)
    return np.vdot(a, b) / magsquare_b * b


def rejection(a, b):  # writing rejection vector of a on b
    """
    Calculates the rejection vector of a on b.

    Arguments:
        a (numpy.ndarray of float): Vector to find rejection of.
        b (numpy.ndarray of float): Vector to be rejected on.
    
    Returns:
        (numpy.ndarray of float): The rejection vector of a on b.
    """
    return a - projection(a, b)



