import cv2
import numpy as np

from environment import Environment, visualize_rollout


class ThreeSpring(Environment):

    WORLD_SIZE = 2.

    def __init__(self, masses, elastic_cstes, spring_length=2., q=None, p=None):
        """Constructor for spring system

        Args:
            mass (list of float): Spring masses (of length 2) (kg)
            elastic_cst (list of float): Spring elastic constant (of length 3) (kg/s^2)
            q ([float], optional): Generalized position in 1-D space: Position (m). Defaults to None
            p ([float], optional): Generalized momentum in 1-D space: Linear momentum (kg*m/s). Defaults to None
        """
        self.masses = masses
        self.elastic_cstes = elastic_cstes
        self.spring_length = spring_length
        super().__init__(q=q, p=p)

    def set(self, q, p):
        """Sets initial conditions for spring system

        Args:
            q ([float]): Generalized position in 1-D space: Position (m)
            p ([float]): Generalized momentum in 1-D space: Linear momentum (kg*m/s)

        Raises:
            ValueError: If p and q are not in 1-D space
        """
        if q is None or p is None:
            return
        if q.shape[0] != 2 or p.shape[0] != 2:
            raise ValueError(
                "q and p do not refer to the two objects of the system")
        self.q = q
        self.p = p

    def get_world_size(self):
        """Return world size for correctly render the environment.
        """
        return self.WORLD_SIZE

    def get_max_noise_std(self):
        """Return maximum noise std that keeps the environment stable."""
        return 0.1

    def get_default_radius_bounds(self):
        """Returns:
            radius_bounds (tuple): (min, max) radius bounds for the environment.
        """
        return (0.1, 1.0)

    def _dynamics(self, t, states):
        """Defines system dynamics

        Args:
            t (float): Time parameter of the dynamic equations.
            states ([float]) Phase states at time t

        Returns:
            equations ([float]): Movement equations of the physical system
        """
        # dynamics of the three-spring-two-masses oscillator
        return [states[2] / self.masses[0], states[3] / self.masses[1], - self.elastic_cstes[0]*(states[0] + self.spring_length / 2) + self.elastic_cstes[1] * (states[1] - states[0] - self.spring_length), - self.elastic_cstes[2] * (states[1] - self.spring_length / 2) - self.elastic_cstes[1] * (states[1] - states[0] - self.spring_length)]

    def _draw(self, res=32, color=True):
        """Returns array of the environment evolution

        Args:
            res (int): Image resolution (images are square).
            color (bool): True if RGB, false if grayscale.

        Returns:
            vid (np.ndarray): Rendered rollout as a sequence of images
        """
        q = self._rollout
        length = len(q[0])
        vid = np.zeros((length, res, res, 3), dtype='float')
        ball_color = self._default_ball_colors
        space_res = 2.*self.get_world_size()/res
        for t in range(length):
            vid[t] = cv2.circle(vid[t], self._world_to_pixels(q[0][t], 0, res),
                                int(self.masses[0]/space_res), ball_color[0], -1)

            vid[t] = cv2.circle(vid[t], self._world_to_pixels(q[1][t], 0, res),
                                int(self.masses[1]/space_res), ball_color[1], -1)
            vid[t] = cv2.blur(cv2.blur(vid[t], (2, 2)), (2, 2))
        vid += self._default_background_color
        vid[vid > 1.] = 1.
        if not color:
            vid = np.expand_dims(np.max(vid, axis=-1), -1)
        return vid

    def _sample_init_conditions(self, radius_bound):
        """Samples random initial conditions for the environment

        Args:
            radius_bound (float, float): Radius lower and upper bound of the phase state sampling.
                Optionally, it can be a string 'auto'. In that case, the value returned by
                get_default_radius_bounds() will be returned.
        """
        radius_lb, radius_ub = radius_bound
        radius = np.random.rand()*(radius_ub - radius_lb) + radius_lb
        states = np.random.rand(4) * 2. - 1
        # states = (states / np.sqrt((states**2).sum())) * radius
        states[0] = (-1 / 2 + (np.random.rand() * 2 - 1) / 3) * self.spring_length
        states[1] = (1 / 2 + (np.random.rand() * 2 - 1) / 3) * self.spring_length
        states[2] = np.sqrt(self.masses[0] / (self.elastic_cstes[0] + self.elastic_cstes[1])) / 10 * (2 * np.random.rand() - 1)
        states[3] = np.sqrt(self.masses[1] / (self.elastic_cstes[2] + self.elastic_cstes[1])) / 10 * (2 * np.random.rand() - 1)
        self.set(np.array([states[0], states[1]]), np.array([states[2], states[3]]))


# Sample code for sampling rollouts
if __name__ == "__main__":

    sp = ThreeSpring(masses=[.5, .3], elastic_cstes=[2, 3, 1], spring_length=2.)
    rolls = sp.sample_random_rollouts(number_of_frames=100,
                                      delta_time=0.1,
                                      number_of_rollouts=16,
                                      img_size=32,
                                      noise_level=0.,
                                      radius_bound=(.5, 1.4),
                                      color=True,
                                      seed=None)
    idx = np.random.randint(rolls.shape[0])
    visualize_rollout(rolls[idx])
