"""
An Event class used to contain information about the impending collision
between 2 balls or between a ball and a container.

Xin Kai Lee 11/3/2020
"""


class Event(tuple):
    """
    A tuple of 5 elements (ball_A, ball_B, count_A, count_B, dt).

    Parameters:
        ball_A (int): The first ball in impending collision.
        ball_B (int): The second ball in impending collision.
        count_A (int): The number of collisions the first ball 
            encountered prior to this impending collision calculation.
        count_B (int): The number of collisions the second ball 
            encountered prior to this impending collision calculation.
        dt (float): The global time this collision will happen on.
    """

    def ball_A(self):
        """
        Returns:
            (int): The index of first ball in impending collision.
        """
        return self[0]

    def ball_B(self):
        """
        Returns:
            (int): The index of second ball in impending collision.
        """
        return self[1]

    def count_A(self):
        """
        Returns:
            (int): The number of collisions the first ball 
                encountered prior to this impending collision calculation.
        """
        return self[2]

    def count_B(self):
        """
        Returns:
            (int): The number of collisions the second ball 
                encountered prior to this impending collision calculation.
        """
        return self[3]

    def dt(self):
        """
        Returns:
            (float): The global time this collision will happen on.
        """
        return self[4]

    def pair(self):
        """
        Returns:
            (list of int): A list of the two balls/ ball and container
                involved in collision.
        """
        return [self[0], self[1]]
