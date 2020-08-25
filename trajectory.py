# import numpy as np
from autograd import jacobian
from autograd import numpy as np
import random
import RRT_3D


def _fig8(t, T):
    const = 1 / T
    # pd = np.array([2 * np.sin(const * 2 * np.pi * t) , np.sin(2 * const * 2 * np.pi * t) , 0])
    # vd = np.array([2 * const * 2 * np.pi * np.cos(const * 2 * np.pi * t),
    #                2 * const * 2 * np.pi * np.cos(2 * const * 2 * np.pi * t), 0])
    # ad = np.array([-2 * (const * 2 * np.pi) ** 2 * np.sin(const * 2 * np.pi * t),
    #                -(const * 2 * 2 * np.pi) ** 2 * np.sin(2 * const * 2 * np.pi * t), 0])
    # jd = np.array([-2 * (const * 2 * np.pi) ** 3 * np.cos(const * 2 * np.pi * t),
    #                -(const * 2 * 2 * np.pi) ** 3 * np.cos(2 * const * 2 * np.pi * t), 0])
    # sd = np.array([2 * (const * 2 * np.pi) ** 4 * np.sin(const * 2 * np.pi * t),
    #                (const * 2 * 2 * np.pi) ** 4 * np.sin(2 * const * 2 * np.pi * t), 0])
    pd = np.array([0, 0, 0])
    vd = np.array([0,
                   0, 0])
    ad = np.array([0,
                   0, 0])
    jd = np.array([0,
                   0, 0])
    sd = np.array([0,
                   0, 0])
    return pd, vd, ad, jd, sd


class RRT():
    def __init__(self):
        self.path, self.dist = RRT_3D.main()
        self.position = []
        print(len(self.path))

    def __call__(self, t):
        if t == "dist":
            return self.dist
        loop = int(t / 1e-2)
        jd = sd = np.zeros(3)
        if loop >= len(self.path):
            self.position.append(self.path[-1][0])
            return self.path[-1][0], self.path[-1][1], self.path[-1][2], jd, sd
            # self.position.append([10, 0, 0])
            # return [10, 0, 0], [0,0,0], [0,0,0], jd, sd
        self.position.append(self.path[loop][0])
        # print(self.path[loop])
        return self.path[loop][0], self.path[loop][1], self.path[loop][2], jd, sd


class RRT_path_to_trajectory():
    def __init__(self):
        # initial condition from RRT
        self.x0 = np.zeros(2)
        # u is the control input, t is how long that control input is applied for, so
        # u = u[0] for 0 <= t < t[0]
        # u = u[1] for t[0] <= t < t[1]
        # ...
        self.path = {'u': np.random.randn(100), 't': np.random.rand(100)} # TODO: load path
        self.idx_last_node = 0
        self.t_last_node = 0. # = np.sum(path['t'][0:idx_internal])
        self.u_last_node = self.path['u'][0]
        self.t_internal = 0.
        self.x_internal = self.x0
        self.dt_internal = 0.033 # strange step size for demonstration
    def f_from_RRT(self, x, u, t):
        # TODO: incorporate same dynamics as the RRT
        return np.array([x[1], u])
    def step_from_RRT(self, x, u, t):
        # TODO: check that integration method is the same
        xdot = self.f_from_RRT(x,u,t)
        x += xdot * self.dt_internal
        return x, t + self.dt_internal
    def get_xd(self, t):
        while self.t_internal < t:
            self.x_internal, self.t_internal = self.step_from_RRT(self.x_internal, self.u_last_node, self.t_internal)
            if self.t_internal > self.t_last_node:
                self.idx_last_node += 1
                self.u_last_node = self.path['u'][self.idx_last_node]
                self.t_last_node += self.path['t'][self.idx_last_node]
        return self.x_internal.copy() # need copy because np.arrays are modified in place
    def __call__(self, t):
        return self.get_xd(t)

def fig8(T):
    ''' Returns a function that takes one argument, time t, and returns the corresponding position, velocity, acceleration, jerk, and snap for the corresponding figure-8 trajectory with period T. 
    Inputs - T: period of figure 8 trajectory
    '''

    def retfun(t): return _fig8(t, T)

    return retfun


def setpoint(pd):
    vd = ad = jd = sd = np.zeros(pd.shape)
    return pd, vd, ad, jd, sd


class random_setpoint_generator():
    def __init__(self, t0=0., T=2.0, bounds=None):
        if bounds is None:
            self.xmin = -5.0
            self.xmax = 15.0
            self.ymin = -10.0
            self.ymax = 10.0
            self.zmin = -10.0
            self.zmax = 10.0
        else:
            self.xmin = bounds['xmin']
            self.xmax = bounds['xmax']
            self.ymin = bounds['ymin']
            self.ymax = bounds['ymax']
            self.zmin = bounds['zmin']
            self.zmax = bounds['zmax']

        self.t_lastupdate = t0
        self.T = T
        self.pd = np.zeros(3)

    def __call__(self, t):
        if t > self.t_lastupdate + self.T:
            # self.pd = np.array([random.uniform(self.xmin, self.xmax), random.uniform(self.ymin, self.ymax),
            #                     random.uniform(self.zmin, self.zmax)])
            self.pd = np.array([0, 0, 0])
            self.t_lastupdate = t + self.T
        return setpoint(self.pd)


path = []
