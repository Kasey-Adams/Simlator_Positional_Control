''' You should add a docstring here '''

import copy
from warnings import warn
import random

import numpy as np
import rowan

import controller
import trajectory
from readparamfile import readparamfile, dotdict


def qmultiply(q1, q2):
    return np.concatenate((
        np.array([q1[0] * q2[0]]),  # w1w2
        q1[0] * q2[1:4] + q2[0] * q1[1:4] + np.cross(q1[1:4], q2[1:4])))


def qconjugate(q):
    return np.concatenate((q[0:1], -q[1:4]))


def qrotate(q, v):
    quat_v = np.concatenate((np.array([0]), v))
    return qmultiply(q, qmultiply(quat_v, qconjugate(q)))[1:]


def qexp(q):
    norm = np.linalg.norm(q[1:4])
    e = np.exp(q[0])
    result_w = e * np.cos(norm)
    if np.isclose(norm, 0):
        result_v = np.zeros(3)
    else:
        result_v = e * q[1:4] / norm * np.sin(norm)
    return np.concatenate((np.array([result_w]), result_v))


def qintegrate(q, v, dt):
    quat_v = np.concatenate((np.array([0]), v * dt / 2))
    return qmultiply(qexp(quat_v), q)


def qnormalize(q):
    return q / np.linalg.norm(q)


class Quadrotor():
    def __init__(self, paramsfile='params-quad-default.txt'):
        self.params = readparamfile(paramsfile)
        arm = 0.707106781 * .3
        self.arm = arm
        t2t = 0.006  # thrust-to-torque ratio
        self.t2t = t2t
        self.mass = 2.4  # kg
        h = 0.15
        arm_length = 0.3  # m
        self.B0 = np.array([
            [1, 1, 1, 1],
            [-arm, -arm, arm, arm],
            [-arm, arm, arm, -arm],
            [-t2t, t2t, -t2t, t2t]
        ])
        self.J = np.array([(1 / 12) * self.mass * ((arm_length * 2) ** 2 + h ** 2),
                           (1 / 12) * self.mass * ((arm_length * 2) ** 2 + h ** 2),
                           (1 / 12) * self.mass * (2 * (arm_length * 2) ** 2)])
        self.t_start = 0.0
        self.t_stop = 60  # seconds
        self.dt = 1e-2  # integration step size
        self.times = np.arange(self.t_start, self.t_stop, self.dt)
        self.ave_dt = self.times[1] - self.times[0]

    def update_motor_speed(self, u, dt, Z=None):
        # Hidden motor dynamics discrete update
        if Z is None:
            Z = u
        else:
            alpha_m = 1 - np.exp(-self.params['w_m'] * dt)
            Z = alpha_m * u + (1 - alpha_m) * Z

        return np.maximum(np.minimum(Z, self.params['motor_max_speed']), self.params['motor_min_speed'])

    def f(self, X, Z, A, t, logentry=None):
        # x = X[0]
        # y = X[1]
        # z = X[2]
        # xdot = X[3]
        # ydot = X[4]
        # zdot = X[5]
        # qw = X[6]
        # qx = X[7]
        # qy = X[8]
        # qz = X[9]
        # wx = X[10]
        # wy = X[11]
        # wz = X[12]
        a = Z ** 2
        # a = A
        J = self.J
        dxdt = np.zeros(13)
        q = X[6:10]
        q = rowan.normalize(q)
        omega = X[10:]
        B0 = self.B0 * self.params['C_T']
        # B0 = self.B0
        eta = np.dot(B0, a)
        f_u = np.array([0, 0, eta[0]])
        tau_u = np.array([eta[1], eta[2], eta[3]])

        dxdt[0:3] = X[3:6]
        # dxdt[3:6] = -9.81 + rowan.rotate(q, f_u) / self.params['m']
        dxdt[3:6] = np.array([0., 0., -9.81]) + rowan.rotate(q, f_u) / self.params['m']
        Vinf = self.params['Vwind_mean'] - np.linalg.norm(X[3:6])
        Vinf_B = rowan.rotate(rowan.inverse(q), Vinf)
        Vz_B = np.array([0.0, 0.0, Vinf_B[2]])
        Vs_B = Vinf_B - Vz_B
        alpha = np.arcsin(np.linalg.norm(Vz_B) / np.linalg.norm(Vinf_B))
        n = np.sqrt(
            np.multiply(a, self.B0[0, :]) / (self.params['C_T'] * self.params['rho'] * self.params['D'] ** 4))
        Fs_B = (Vs_B / np.linalg.norm(Vs_B)) * self.params['C_s'] * self.params['rho'] * sum(n ** self.params['k1']) * (
            np.linalg.norm(Vinf) ** (2 - self.params['k1'])) * (self.params['D'] ** (2 + self.params['k1'])) * (
            (np.pi / 2) ** 2 - alpha ** 2) * (alpha + self.params['k2'])
        #Fs_B = [0,0,0]
        dxdt[3:6] += rowan.rotate(q, Fs_B) / self.mass
        qnew = rowan.calculus.integrate(q, omega, self.ave_dt)
        if qnew[0] < 0:
            qnew = -qnew

        dxdt[6:10] = (qnew - q) / self.ave_dt
        dxdt[10:] = 1 / J * (np.cross(J * omega, omega) + tau_u)
        dxdt[10:] += 1 / J * np.cross(np.array([0.0, 0.0, self.params['D'] / 4]), Fs_B)
        euler_o = rowan.to_euler(q, 'xyz')
        if logentry is not None:
            logentry['f_u'] = f_u
            logentry['tau'] = tau_u
            logentry['euler_o'] = euler_o

        return dxdt.reshape((len(dxdt), 1))

    def step(self, X, u, t, dt, Z=None, A=None, logentry=None):

        Z = self.update_motor_speed(Z=Z, u=u, dt=dt)
        dsdt = self.f(X, Z, A, t, logentry=logentry)
        # Z = self.update_motor_speed(Z=Z, u=u, dt=dt)
        sp1 = np.squeeze(np.reshape(X, (len(X), 1)) + dsdt * dt)
        # Zero mean, Gaussian noise model
        if logentry is not None:
            pass
        return (sp1, t + dt, Z)

    def runiter(self, controller):
        X = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        Z = np.array([0.0, 0.0, 0.0, 0.0])

        t = self.params['t_start']  # simulation time
        t_posctrl = -0.0  # time of next position control update
        t_attctrl = -0.0  # time of next attitude control update
        t_readout = -0.0
        logentry = {}
        RRT = trajectory.RRT()
        logentry['path'] = RRT.path
        while t < self.params['t_stop']:
            if t >= t_posctrl:
                pd, vd, ad, jd, sd = RRT(t)
                T_r, th_r, w_d, alpha_d, u, A = controller.policy(X=X, pd=pd, vd=vd, ad=ad, jd=jd, sd=sd,
                                                                  logentry=logentry, time=t)
                t_posctrl += controller.params['dt_posctrl']
            # if t >= t_attctrl:
            #     u = controller.attitude(X, T_r=T_r, th_r=th_r, w_d=w_d, alpha_d=alpha_d, logentry=logentry)
            #     t_attctrl += controller.params['dt_attctrl']
            X, t, Z = self.step(X=X, u=u, t=t, dt=self.params['dt'], Z=Z, A=A, logentry=logentry)
            if t >= t_readout:
                logentry['t'] = t
                logentry['X'] = X
                logentry['Z'] = Z
                logentry['T_r'] = T_r
                logentry['th_r'] = th_r
                logentry['w_d'] = w_d
                logentry['alpha_d'] = alpha_d
                logentry['pd'] = pd
                logentry['vd'] = vd
                logentry['ad'] = ad
                logentry['jd'] = jd
                logentry['sd'] = sd
                #logentry['pos'] = RRT.position
                #logentry['dist'] = RRT(dist)
                yield copy.copy(logentry)
                t_readout += self.params['dt_readout']

    def run(self, controller=controller.Baseline(), trajectory=trajectory):
        # # Use zip to switch output array from indexing of time, (X, t, log), to indexing of (X, t, log), time
        # log = zip(*self.runiter(trajectory=trajectory, controller=controller))
        log = list(self.runiter(controller=controller))
        # Concatenate entries of the log dictionary into single np.array, with first dimension corresponding to each time step recorded
        log2 = dotdict({k: np.array([logentry[k] for logentry in log]) for k in log[0]})
        return log2

# class QuadrotorWithSideForce
#     def __init__(self, *args, sideforcemodel='force and torque', **kwargs):
#         super().__init__(*args, **kwargs)
#         self.sideforcemodel = sideforcemodel
#         self.t_wind_update = np.nan
#
#     def get_wind_velocity(self, t):
#         if t != self.t_wind_update:
#             self.wind_velocity = self.params.Vwind_mean + np.dot(self.params.Vwind_cov, np.random.normal(size=2))
#             # TODO: implement wind as band limited Gaussian process
#         return self.wind_velocity
#
#     def f(self, X, Z, t, logentry=None):
#         Xdot = super().f(X, Z, t, logentry)
#
#         th = X[2]
#
#         # Side force model
#         R_world_to_body = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])
#         Vwind = self.get_wind_velocity(t)  # velocity of wind in world frame
#         Vinf = Vwind - X[3:5]  # velocity of air relative to quadrotor in world frame orientation
#         Vinf_B = np.dot(R_world_to_body, Vinf)  # use _B suffix to denote vector is in body frame
#         Vy_B = np.array([0., Vinf_B[1]])  # y-body-axis aligned velocity
#         Vs_B = np.array([Vinf_B[0], 0.])  # cross wind
#         if np.linalg.norm(Vs_B) > 1e-4:
#             aoa = np.arcsin(np.linalg.norm(Vy_B) / np.linalg.norm(Vinf_B))  # angle of attack
#             n = np.sqrt(Z / self.params.C_T)  # Propeller rotation speed
#             Fs_B = (Vs_B / np.linalg.norm(Vs_B)) * self.params.C_s * self.params.rho * sum(n ** self.params.k1) * (
#                         np.linalg.norm(Vinf) ** (2 - self.params.k1)) * (self.params.D ** (2 + self.params.k1)) * (
#                                (np.pi / 2) ** 2 - aoa ** 2) * (aoa + self.params.k2)
#             # print(aoa, n, self.params.g, F, Fs_B)
#         else:
#             Fs_B = np.array([0., 0.])
#
#         Fs = np.dot(R_world_to_body.transpose(), Fs_B)
#         tau_s = - Fs_B[0] * (self.params.D / 4)
#
#         if self.sideforcemodel == 'force and torque':
#             pass
#         elif self.sideforcemodel == 'force only':
#             tau_s = 0. * tau_s
#         elif self.sideforcemodel == 'torque only':
#             Fs = 0. * Fs
#
#         Xdot[3:5] += Fs / self.params.m
#         Xdot[5] += tau_s / self.params.J
#
#         if logentry is not None:
#             logentry['Fs'] = np.linalg.norm(Fs)
#             logentry['tau_s'] = tau_s
#             logentry['Vwind'] = Vwind
#
#         return Xdot
