from warnings import warn

import rowan
import numpy as np

# import parambaseline as params
# from planarquadsim import Quadrotor
from readparamfile import readparamfile


class Controller():
    def __init__(self):
        self.params = readparamfile('params-quad-default.txt')

    def position(self, X, pd=np.zeros(2), vd=np.zeros(2), ad=np.zeros(3), jd=np.zeros(3), sd=np.zeros(3),
                 logentry=None):
        raise NotImplementedError

    def attitude(self, X, T_r=0., th_r=0., w_d=0., alpha_d=0., logentry=None):
        raise NotImplementedError


class Baseline(Controller):
    def __init__(self, ctrlparamfile="params-baseline.txt"):
        self.params = readparamfile(filename='params-quad-default.txt')
        self.params = readparamfile(filename=ctrlparamfile, params=self.params)
        self.int_x_error = 0.
        self.int_y_error = 0.
        self.int_z_error = 0.

        self.mass = 2.4  # kg

        h = 0.15
        arm_length = 0.3  # m
        self.J = np.array([(1 / 12) * self.mass * ((arm_length * 2) ** 2 + h ** 2),
                           (1 / 12) * self.mass * ((arm_length * 2) ** 2 + h ** 2),
                           (1 / 12) * self.mass * (2 * (arm_length * 2) ** 2)])

        # Note: we assume here that our control is forces
        arm = 0.707106781 * 0.3
        self.arm = arm
        t2t = 0.006  # thrust-to-torque ratio
        self.t2t = t2t
        self.B0 = np.array([
            [1, 1, 1, 1],
            [-arm, -arm, arm, arm],
            [-arm, arm, arm, -arm],
            [-t2t, t2t, -t2t, t2t]
        ])
        self.g = 9.81  # not signed

        # PID parameters
        # self.K_i = 0.25  # eigs[0] * eigs[1] * eigs[2]
        # self.K_p = 1.0  # eigs[0] * eigs[1] + eigs[1] * eigs[2] + eigs[2] * eigs[0]
        # self.K_d = 3.0  # sum(eigs)

        self.Att_p = 400
        self.Att_d = 150
        self.time = 0.0
        self.int_p_e = np.zeros(3)

    def policy(self, X, pd=np.zeros(3), vd=np.zeros(3), ad=np.zeros(3), jd=np.zeros(3), sd=np.zeros(3),
                 logentry=None, time=None, K_i = 0, K_p = 0, K_d = 0):
        #     print('baseline: X = ', X)
        p_e = pd-X[:3]
        v_e = vd-X[3:6]
        print(X)
        int_p_e = self.integrate_error(p_e, time)
        F_d = (self.K_i * int_p_e + self.K_p * p_e + self.K_d * v_e) * self.mass  # TODO: add integral term
        F_d[2] += self.g * self.mass
        T_d = np.linalg.norm(F_d)
        yaw_d = 0
        roll_d = np.arcsin((F_d[0] * np.sin(yaw_d) - F_d[1] * np.cos(yaw_d)) / T_d)
        pitch_d = np.arctan((F_d[0] * np.cos(yaw_d) + F_d[1] * np.sin(yaw_d)) / F_d[2])
        euler_d = np.array([roll_d, pitch_d, yaw_d])
        q = rowan.normalize(X[6:10])
        euler = rowan.to_euler(q, 'xyz')

        att_e = -(euler - euler_d)
        th_r = rowan.normalize(rowan.from_euler(att_e[0], att_e[1], att_e[2], 'xyz'))
        th_d = rowan.normalize(rowan.from_euler(euler_d[0], euler_d[1], euler_d[2], 'xyz'))
        att_v_e = -X[10:]

        torque = self.Att_p * att_e + self.Att_d * att_v_e
        torque[0] *= self.J[0]
        torque[1] *= self.J[1]
        torque[2] *= self.J[2]
        Jomega = np.array([self.J[0] * X[10], self.J[1] * X[11], self.J[2] * X[12]])
        torque -= np.cross(Jomega, X[10:])

        yawpart = -0.25 * torque[2] / self.t2t
        rollpart = 0.25 / self.arm * torque[0] #/ self.t2t
        pitchpart = 0.25 / self.arm * torque[1] #/ self.t2t
        thrustpart = 0.25 * T_d
        motorForce = np.array([
            thrustpart - rollpart - pitchpart + yawpart,
            thrustpart - rollpart + pitchpart - yawpart,
            thrustpart + rollpart + pitchpart + yawpart,
            thrustpart + rollpart - pitchpart - yawpart
        ])
        Fc = motorForce/self.params['C_T']
        omega = np.sqrt(
            np.maximum(Fc, self.params['motor_min_speed']))  # Ensure each prop has a positive thrust
        omega = np.minimum(omega, self.params['motor_max_speed'])  # Maximum rotation speed
        #     print('baseline: omega = ', omega)
        #     print('T', T_r)
        #     print('tau',tau_r)

        # x_error = pd[0] - X[0]
        # y_error = pd[1] - X[1]
        # z_error = pd[2] - X[2]
        #
        # vx_error = vd[0] - X[3]
        # vy_error = vd[1] - X[4]
        # vz_error = vd[2] - X[5]
        #
        # #     print('baseline: x_error = ', x_error)
        # #     print('baseline: y_error = ', y_error)
        #
        # ax_d = ad[0]
        # ay_d = ad[1]
        # az_d = ad[2]
        #
        # self.int_x_error += self.params.dt_posctrl * x_error
        # self.int_y_error += self.params.dt_posctrl * y_error
        # self.int_z_error += self.params.dt_posctrl * z_error
        #
        # ax_r = self.params['K_p_x'] * x_error + self.params['K_d_x'] * vx_error + self.params[
        #     'K_i_x'] * self.int_x_error + ax_d
        # ay_r = self.params['K_p_y'] * y_error + self.params['K_d_y'] * vy_error + self.params[
        #     'K_i_y'] * self.int_y_error + ay_d
        # az_r = self.params['K_p_z'] * z_error + self.params['K_d_z'] * vz_error + self.params[
        #     'K_i_z'] * self.int_z_error + az_d
        # #     print('baseline: x PDI terms:',K_p_x * x_error, K_d_x * vx_error, K_i_x * intx_error, ax_d)
        # #     print('baseline: y PDI terms:',K_p_y * y_error, K_d_y * vy_error, K_i_y * inty_error, ay_d)
        #
        # jx_d = jd[0]
        # jy_d = jd[1]
        # jz_d = jd[2]
        # #     jx_r = K_p_x * vx_error + K_d_x * (ax_d - ax_r) + K_i_x * x_error + jx_d
        # #     jy_r = K_p_y * vy_error + K_d_y * (ay_d - ay_r) + K_i_y * y_error + jy_d
        #
        # sx_d = sd[0]
        # sy_d = sd[1]
        # sz_d = sd[2]
        # #     sx_r = K_p_x * (ax_d - ax_r) + K_d_x * (jx_d - jx_r) + K_i_x * vx_error + sx_d
        # #     sy_r = K_p_y * (ay_d - ay_r) + K_d_y * (jy_d - jy_r) + K_i_y * vy_error + sy_d
        #
        # #     print('baseline: ay_r = ', ay_r)
        #
        # Fx_r = ax_r * self.params['m']
        # Fy_r = (ay_r + self.params['g']) * self.params['m']
        # Fz_r = az_r * self.params['m']
        #
        # Fy_r = np.maximum(Fy_r, 0.10 * self.params['g'] * self.params['m'])
        #
        #
        # #     print('baseline: Fx_r, Fy_r = ', Fx_r, Fy_r)
        #
        # T_r = np.minimum(np.sqrt(Fy_r ** 2 + Fx_r ** 2 + Fz_r ** 2), 125.)  # / np.cos(X[2])
        # if T_r > 124.:
        #     warn('thrust gets too high')
        #     # print('T_r = ', T_r)
        # T_d = self.params['m'] * np.sqrt(ax_d ** 2 + (ay_d + self.params['g']) ** 2 + az_d ** 2)
        #
        # # th = X[2]
        # th_r =
        # th_d =
        #
        # mat = np.array([[-np.sin(th_d), -T_d * np.cos(th_d)],  #TODO: use th_d, T_d
        #                 [np.cos(th_d), -T_d * np.sin(th_d)]])
        # b = self.params['m'] * np.array([jx_d, jy_d, jz_d])#
        # v = np.linalg.solve(mat, b)
        # T_d_dot = v[0]
        # w_d = v[1]
        # # w = X[5]
        #
        # mat = np.array([[-np.sin(th_d), -T_d * np.cos(th_d)],
        #                 [np.cos(th_d), -T_d * np.sin(th_d)]])
        # b = self.params['m'] * np.array([sx_d, sy_d]) - np.array(
        #     [-2. * T_d_dot * np.cos(th_d) * w_d + T_d * np.sin(th_d) * (w_d ** 2),
        #      -2. * T_d_dot * np.sin(th_d) * w_d - T_d * np.cos(th_d) * (w_d ** 2)])
        #
        # v = np.linalg.solve(mat, b)
        # # T_d_ddot = v[0]
        # alpha_d = v[1]
        # #     print('baseline_pos: Tdot, w_d, alpha_d', Tdot, w_d, alpha_d)
        w_d = 0
        alpha_d = 0
        if logentry is not None:
            logentry['th_d'] = th_d
            logentry['th_r'] = th_r
            logentry['T_d'] = T_d
            logentry['euler'] = euler
            logentry['att_e'] = att_e
            # logentry['Fx_r'] = Fx_r
            # logentry['Fy_r'] = Fy_r
            # logentry['Fz_r'] = Fz_r
            # logentry['int_x_error'] = self.int_x_error
            # logentry['int_y_error'] = self.int_y_error
            # logentry['int_y_error'] = self.int_z_error
        return T_d, th_d, w_d, alpha_d, omega, motorForce

    # def attitude(self, X, T_r=0., th_r=0., w_d=0., alpha_d=0., logentry=None, time=None):
    #     p_e = -X[:3]
    #     v_e = -X[3:6]
    #
    #     int_p_e = self.integrate_error(p_e, time)
    #     F_d = (self.K_i * int_p_e + self.K_p * p_e + self.K_d * v_e) * self.mass  # TODO: add integral term
    #     F_d[2] += self.g * self.mass
    #     T_d = np.linalg.norm(F_d)
    #     yaw_d = 0
    #     roll_d = np.arcsin((F_d[0] * np.sin(yaw_d) - F_d[1] * np.cos(yaw_d)) / T_d)
    #     pitch_d = np.arctan((F_d[0] * np.cos(yaw_d) + F_d[1] * np.sin(yaw_d)) / F_d[2])
    #     euler_d = np.array([roll_d, pitch_d, yaw_d])
    #     euler = rowan.to_euler(rowan.normalize(X[6:10]), 'xyz')
    #     att_e = -(euler - euler_d)
    #     att_v_e = -X[10:]
    #     torque = self.Att_p * att_e + self.Att_d * att_v_e
    #     torque[0] *= self.J[0]
    #     torque[1] *= self.J[1]
    #     torque[2] *= self.J[2]
    #     Jomega = np.array([self.J[0] * X[10], self.J[1] * X[11], self.J[2] * X[12]])
    #     torque -= np.cross(Jomega, X[10:])
    #
    #     yawpart = -0.25 * torque[2] / self.t2t
    #     rollpart = 0.25 / self.arm * torque[0]
    #     pitchpart = 0.25 / self.arm * torque[1]
    #     thrustpart = 0.25 * T_d
    #
    #     motorForce = np.array([
    #         thrustpart - rollpart - pitchpart + yawpart,
    #         thrustpart - rollpart + pitchpart - yawpart,
    #         thrustpart + rollpart + pitchpart + yawpart,
    #         thrustpart + rollpart - pitchpart - yawpart
    #     ])
    #     omega = np.sqrt(
    #         np.maximum(motorForce, self.params['motor_min_speed']))  # Ensure each prop has a positive thrust
    #     omega = np.minimum(omega, self.params['motor_max_speed'])  # Maximum rotation speed
    #     #     print('baseline: omega = ', omega)
    #     #     print('T', T_r)
    #     #     print('tau',tau_r)
    #
    #     if logentry is not None:
    #         pass
    #     return omega

    def integrate_error(self, p_e, time):
        if not self.time:
            dt = 0.0
            self.time = time
            return np.zeros(3)
        else:
            dt = time - self.time
            self.time = time
            self.int_p_e += dt * p_e
            return self.int_p_e


class BaselineNoFdFwd(Baseline):
    def position(self, X, pd=np.zeros(3), vd=np.zeros(3), ad=np.zeros(3), jd=np.zeros(3), sd=np.zeros(3),
                 logentry=None):
        T_r, th_r, _, _ = super().position(X, pd=pd, vd=vd, ad=ad, jd=jd, sd=sd, logentry=logentry)
        return T_r, th_r, 0., 0.
