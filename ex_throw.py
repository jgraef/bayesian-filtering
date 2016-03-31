import matplotlib.pyplot as plt
from numpy import matrix

from kf import *


class TestSystem(System):
    def __init__(self, dt):
        self._A = matrix([[1, dt, 0.5 * dt],
                          [0,  1,       dt],
                          [0,  0,        1]])
        self._B = matrix([[0.2, 0.5 * dt, 0.25 * dt],
                          [0,        1,         1]])
        self._Q_omega = matrix([[1/50, 1/50,     0],
                                [   0, 1/50,     0],
                                [   0,    0,     0]])
        self._Q_nu = matrix([[5,   0],
                             [0, 2]])

    def A(self, n):
        return self._A

    def B(self, n):
        return self._B

    def Q_omega(self, n):
        return self._Q_omega

    def Q_nu(self, n):
        return self._Q_nu




if (__name__ == "__main__"):
    system = TestSystem(0.1)
    sim = SystemSimulation(system, matrix([[0], [5], [-0.981]]))
        
    kf = KalmanFilter(system, sim.x, system.Q_omega(1))

    plot = []

    for i in range(100):
        y = sim.observe()
        assert sim.n == kf.n
        xf, Pf, xp, Pp = kf.update(y)
        plot.append((sim.n - 1, sim.x[0].A1[0], xf[0].A1[0], Pf[0].A1[0]))
        sim.step()

    print("plotting")
    plt.plot(
    [p[0] for p in plot], [p[1] for p in plot], "b-",
    [p[0] for p in plot], [p[2] for p in plot], "r-",
    [p[0] for p in plot], [p[2] + 2 * p[3] for p in plot], "g-",
    [p[0] for p in plot], [p[2] - 2 * p[3] for p in plot], "g-"
    )
    plt.show()
