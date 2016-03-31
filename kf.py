from numpy import matrix, linalg, zeros
from numpy.random import multivariate_normal



class System:
    def A(self, n):
        raise NotImplementedError

    def B(self, n):
        raise NotImplementedError

    def Q_omega(self, n):
        raise NotImplementedError

    def Q_nu(self, n):
        raise NotImplementedError


    
class SystemSimulation:
    def __init__(self, system, x_0):
        self.n = 1
        self.system = system
        self.x = x_0

    def noise(self, Q):
        mean = zeros((Q.shape[0],))
        return matrix(multivariate_normal(mean, Q)).transpose()

    def observe(self):
        return self.system.B(self.n) * self.x + self.noise(self.system.Q_nu(self.n))

    def step(self):
        self.x = self.system.A(self.n) * self.x + self.noise(self.system.Q_omega(self.n))
        self.n += 1
        return self.x


class KalmanFilter:
    def __init__(self, system, x_predicted, P_predicted):
        self.n = 1
        self.system = system
        self.x_predicted = x_predicted
        self.P_predicted = P_predicted


    def update(self, y_n):
        B_n = self.system.B(self.n)
        B_nT = B_n.transpose()
        K = (B_n * self.P_predicted * B_nT + self.system.Q_nu(self.n))
        G_n = self.P_predicted * B_nT * linalg.inv(K)
        alpha_n = y_n - B_n * self.x_predicted
        x_filtered = self.x_predicted + G_n * alpha_n
        A_n = self.system.A(self.n)
        x_predicted = A_n * x_filtered
        P_filtered = self.P_predicted - G_n * B_n * self.P_predicted
        P_predicted = A_n * P_filtered * A_n.transpose() + self.system.Q_omega(self.n)

        self.x_predicted = x_predicted
        self.P_predicted = P_predicted
        self.n += 1

        return x_filtered, P_filtered, x_predicted, P_predicted



__all__ = [
    "System",
    "SystemSimulation",
    "KalmanFilter"
]
