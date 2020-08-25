import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import math
import random
import numpy as np
from scipy.integrate import solve_ivp
import time


class Environment:
    def __init__(self):

        # obstacle array of [x,y,r] arrays defining obstacles in teh simulation
        self.obstacle = obstacles

        # goal region
        self.xg = goal[0]
        self.yg = goal[1]
        self.zg = goal[2]
        self.eg = eg

        # simulation boundaries
        self.xmin = xmin
        self.ymin = ymin
        self.zmin = zmin
        self.xmax = xmax
        self.ymax = ymax
        self.zmax = zmax

    # check if point is in obstacle
    def in_obstacle(self, x, y, z):
        obs = False
        for i in range(0, len(self.obstacle)):
            cx = self.obstacle[i][0]  # x coordinate of obstacle center
            cy = self.obstacle[i][1]  # y coordinate of obstacle center
            cz = self.obstacle[i][2]  # z coordinate of obstacle center
            cr = self.obstacle[i][3]  # radius of obstacle
            acd = np.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2)
            if acd <= cr + radius:
                obs = True
        return obs

    # checks if point is in goal
    def in_goal(self, x, y, z):
        dpg = np.sqrt(
            (x - self.xg) ** 2 + (y - self.yg) ** 2 + (z - self.zg) ** 2)  # distance from point to goal center
        if dpg < self.eg:  # if distance is less than goal radius end
            return True
        return False

    # check if point is in simulation bounds
    def in_bounds(self, x, y, z):
        if x < self.xmin + radius or x > self.xmax - radius or y < self.ymin + radius or y > self.ymax - radius or z < self.zmin + radius or z > self.zmax - radius:
            return False
        return True


class RRT:
    def __init__(self):
        # starting node
        nstart = initial_position

        # tracks state of the object
        self.state = [[], [], [], [], [], [], [], [], [], [], [], []]
        self.parent = [0]
        for i in range(0, len(self.state)):
            self.state[i] = [nstart[i]]

        self.time = [0]
        # first node is the only node whose parent is itself

        self.dmax = dmax
        self.best_dist = 10

        self.d_track = []

        self.d_track.append(self.dmax)

        self.goalstate = 0
        self.path = []

        self.state_follow = [[[0, 0, 0], [0, 0, 0], [0, 0, 0]]]  # [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        self.full_state_follow = []
        self.path_state = []
        self.full_path_state = []

        self.out = True
        self.goal_found = False

        self.acc_varx = acc_varx
        self.acc_vary = acc_vary
        self.acc_varz = acc_varz


    # check the distance between two nodes
    def distance_between(self, n1, n2):
        d = np.sqrt((self.state[0][n1] - self.state[0][n2]) ** 2 + (self.state[2][n1] - self.state[2][n2]) ** 2 + (
                self.state[4][n1] - self.state[4][n2]) ** 2)
        return d

    # expand a random node and test if its valid, connect to nearest node if it is
    def expand(self):
        x = np.zeros(12)
        in_obs = True

        # randomize a node until it is in a valid configuration
        while in_obs is True:
            # if self.goal_found:
            #     x[0] = goal[0]
            #     x[1] = goal[3]
            #     x[2] = goal[1]
            #     x[3] = goal[4]
            #     x[4] = goal[2]
            #     x[5] = goal[5]
            #     x[7] = goal[6]
            #     x[9] = goal[7]
            #     x[11] = goal[8]
            # else:
            x[0] = np.random.randn() * self.dmax + E.xg  # x
            x[1] = np.random.randn() * vel_var  # dxdt
            x[2] = np.random.randn() * self.dmax + E.yg  # y
            x[3] = np.random.randn() * vel_var  # dydt
            x[4] = np.random.randn() * self.dmax + E.zg  # z
            x[5] = np.random.randn() * vel_var  # dzdt
            x[7] = np.random.randn() * 0  # dyawdt
            x[9] = np.random.randn() * vel_var  # dpitchdt
            x[11] = np.random.randn() * vel_var  # drolldt
            x[6] = np.random.rand() * 2 * np.pi  # yaw
            x[8] = np.random.rand() * 2 * np.pi  # pitch
            x[10] = np.random.rand() * 2 * np.pi  # roll
            if E.in_obstacle(x[0], x[2], x[4]) is False and E.in_bounds(x[0], x[2], x[4]) is True:
                in_obs = False
            # print(in_obs)
            # print(x[1],x[3],x[5])
        dt = np.random.rand()
        n = self.number_of_nodes()

        # adds new node
        self.add_node(n, x)

        if E.in_goal(G.state[0][-1], G.state[2][-1], G.state[4][-1]):
            if not self.out:
                self.acc_varx *= -10
                self.acc_varx *= -10
                self.acc_varx *= -10
            self.out = True
        else:
            self.out = False
            self.acc_varx = np.sign(goal[0] - G.state[0][-1]) * .5
            self.acc_vary = np.sign(goal[1] - G.state[2][-1]) * .5
            self.acc_varz = np.sign(goal[2] - G.state[4][-1]) * .5

        # checks nearest node and gets parameters of that node
        n_nearest = self.near(n)
        n_nearest = int(n_nearest)
        x_nearest = []
        x_append = []
        for i in range(0, 12):
            x_nearest.append(self.state[i][n_nearest])
        nearest_parent = self.parent[n_nearest]
        t_nearest = self.time[nearest_parent]
        # steers the robot to the new node
        (x_new, u, track, col) = self.steer(x_nearest, x, t_nearest, t_nearest + dt)

        # removes the theoretical node
        self.remove_node(n)

        # if there was no sample free of collission, go to next loop
        if col is True:
            return

        # add the node with sampled dynamics
        else:
            self.add_node(n, x_new)
            self.add_edge(n_nearest, n)
            self.time.insert(n, t_nearest + dt)
            x_check = np.ma.array(x_new, mask=False)
            x_check.mask[6] = True
            x_check.mask[8] = True
            x_check.mask[10] = True
            if np.linalg.norm(x_check.compressed() - goal) < self.dmax:
                self.dmax = np.linalg.norm(x_check.compressed() - goal)
            if np.sqrt((x_check[0] - goal[0]) ** 2 + (x_check[2] - goal[1]) ** 2 + (
                    x_check[4] - goal[2]) ** 2):
                self.best_dist = np.sqrt((x_check[0] - goal[0]) ** 2 + (x_check[2] - goal[1]) ** 2 + (
                        x_check[4] - goal[2]) ** 2)
            # if np.sqrt((G.state[0][-1] - G.state[0][0]) ** 2 + (G.state[2][-1] - G.state[2][0]) ** 2 + (
            #         G.state[4][-1] - G.state[4][0]) ** 2) > np.sqrt((G.state[0][-1] - goal[0]) ** 2 + (
            #         G.state[2][-1] - goal[2]) ** 2 + (G.state[4][-1] - goal[4]) ** 2):
            self.d_track.append(np.linalg.norm(x_check.compressed() - goal))
            self.state_follow.append([[x_new[0], x_new[2], x_new[4]], [x_new[1], x_new[3], x_new[5]],
                                      [u[0] * np.cos(x[6]) * self.acc_varx, u[1] * np.sin(x[6]) * self.acc_vary, u[2] * np.sin(x[8]) * self.acc_varz]])
            self.full_state_follow.append(
                [[x_nearest], [x], [x[1], u[0] * np.cos(x[6]) * (goal[0] - x[0]) * vel_var, x[3],
                                    u[1] * np.sin(x[6]) * (goal[1] - x[2]) * vel_var,
                                    x[5], u[2] * np.sin(x[8]) * (goal[2] - x[4]) * vel_var, x[7], u[3],
                                    x[9], u[4], x[11], u[5]], [t_nearest, t_nearest + dt]])
            # else:
            # for i in range (0, steps):
            #     x_append.append([[track[i][0], track[i][2], track[i][4]], [track[i][1], track[i][3], track[i][5]],
            #                           [u[0] * np.cos(track[i][6]), u[1] * np.sin(track[i][6]), u[2] * np.sin(track[i][8])]])
            # self.state_follow.append(x_append)
            # print(self.state_follow)
            # find the nearest node

    def near(self, n):
        dmin = self.distance_between(0, n)
        nearest = 0
        for i in range(1, n):
            if self.distance_between(i, n) < dmin:
                dmin = self.distance_between(i, n)
                nearest = i
        return nearest

    # add node at position n with state x
    def add_node(self, n, x):
        for i in range(0, 12):
            self.state[i].insert(n, x[i])

    # remove node at position n
    def remove_node(self, n):
        for i in range(0, 12):
            self.state[i].pop(n)

    # connect two nodes
    def add_edge(self, parent, child):
        self.parent.append(parent)

    # remove the edge between two nodes
    def remove_edge(self, n):
        self.parent.pop(n)

    # gets the number of nodes in the tree
    def number_of_nodes(self):
        return len(self.state[0])

    # steers the robot towards point x1 from point x0 by integrating dynamics over t0 to tf
    def steer(self, x0, x1, t0, tf):
        n_samples = 50  # points to sample
        u_candidates = []
        x_candidates = []
        x_track = []
        col_list = []
        for i in range(0, n_samples):
            u_candidates.append([0, 0, 0, 0, 0, 0])  # assume nearest is starting node
            x_candidates.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            x_track.append(np.zeros((12, steps)))
            col_list.append(False)
        x_free = []
        u_free = []
        track_free = []
        for i in range(0, n_samples):
            u_candidates[i] = self.sample_u()  # randomly sample
            x_candidates[i], col_list[i], x_track[i] = self.propegate_dynamics(x0, u_candidates[i], t0,
                                                                               tf)  # create path from sample
        for i in range(0, len(col_list)):

            # if no collision is found on the path, append it to available states
            if col_list[i] is False:
                x_free.append(x_candidates[i])
                u_free.append(u_candidates[i])
                track_free.append(x_track[i])
        if x_free == []:
            return None, None, None, True  # if no free states, go to next loop
        else:
            # take path ending closest to desired node
            nearest = 0
            dist = np.sqrt((x_free[0][0] - x1[0]) ** 2 + (x_free[0][2] - x1[2]) ** 2 + (x_free[0][4] - x1[4]) ** 2)
            for i in range(1, len(x_free)):
                dist1 = np.sqrt((x_free[i][0] - x1[0]) ** 2 + (x_free[i][2] - x1[2]) ** 2 + (x_free[i][4] - x1[4]) ** 2)
                if dist1 < dist:
                    dist = dist1
                    nearest = i
            x_new = x_free[nearest]
            u_new = u_free[nearest]
            track_new = track_free[nearest]
            # [u_new[0] * np.cos(x0[6]), u_new[1] * np.cos(x0[6]), u_new[2] * np.sin(x0[8])]])
            return x_new, u_new, track_new, False

    def sample_u(self):
        # random sampling
        # if np.sqrt((G.state[0][-1] - G.state[0][0]) ** 2 + (G.state[2][-1] - G.state[2][0]) ** 2 + (G.state[4][-1] - G.state[4][0]) ** 2) > np.sqrt((G.state[0][-1] - goal[0]) ** 2 + (G.state[2][-1] - goal[2]) ** 2 + (G.state[4][-1] - goal[4]) ** 2):
        #     u = np.zeros(6)
        #     u[0] = -np.sqrt(np.random.rand() ** 2) * np.sign(G.state[1][-1])
        #     u[1] = -np.sqrt(np.random.rand() ** 2) * np.sign(G.state[3][-1])
        #     u[2] = -np.sqrt(np.random.rand() ** 2) * np.sign(G.state[5][-1])
        #     u[3] = np.random.randn()
        #     u[4] = np.random.randn()
        #     u[5] = np.random.randn()
        # else:
        u = np.zeros(6)
        u[0] = u[1] = u[2] = np.random.rand()
        # u[1] = np.random.rand()
        # u[2] = np.random.rand()
        u[3] = u[4] = u[5] = np.random.randn()
        # u[4] = np.random.randn()
        # u[5] = np.random.randn()
        return u

    def propegate_dynamics(self, x0, u, t0, tf):
        # create path by integrating dynamics
        def get_xdot(t, x):
            # dynamics from random sample
            xdot = np.array([x[1],
                             u[0] * np.cos(x[6]) * self.acc_varx,
                             x[3],
                             u[1] * np.sin(x[6]) * self.acc_vary,
                             x[5],
                             u[2] * np.sin(x[8]) * self.acc_varz,
                             x[7],
                             u[3],
                             x[9],
                             u[4],
                             x[11],
                             u[5]])
            return xdot

        tsteps = []
        for i in range(0, steps):
            tsteps.append((tf - t0) * i / steps + t0)
        sol = solve_ivp(get_xdot, [t0, tf], x0, t_eval=tsteps)  # initial value problem
        xout = sol.y
        xnew = []

        # final point on path
        for i in range(0, 12):
            xnew.append(xout[i][-1])
        collision = False

        # check if any points along path lie in the obstacle
        for i in range(0, steps):
            if E.in_obstacle(xout[0][i], xout[2][i], xout[4][i]) is True or E.in_bounds(xout[0][i], xout[2][i],
                                                                                        xout[4][i]) is False:
                collision = True
                # print(i)
                break
            # if E.in_goal(xout[0][i], xout[2][i]):
            #     for j in range(0, 6):
            #         xnew[j] = xout[j][i]
            #         break

        return xnew, collision, xout


# Global Variables
radius = .5  # radius of bot

# node limit
nmax = 5000

# integration steps
steps = 6

# goal region
initial_position = np.zeros(12)
goal = np.array(
    [10.0, 0.0, 0.0, 1.0, 0, 0, 0, 0, 0])  # desired final position [0,2,4], velocity[1,3,5], angular velocity[6,7,8]
eg = 1  # radius of goal region

# simulation boundaries
xmin = -5
xmax = 15
ymin = -10
ymax = 10
zmin = -10
zmax = 10

# initial step size
dmax = 10

# variance
vel_var = .5
acc_varx = .5
acc_vary = .5
acc_varz = .5

# obstacles
obstacles = []

# create an RRT tree with a start node
G = RRT()

# environment instance
E = Environment()


def goalpath(goalstate):
    # draw path to goal if it exists
    current = goalstate
    # print(current)
    parent = G.parent[current]
    while parent:
        G.path.append(current)
        current = parent
        parent = G.parent[current]
    print("The best path was found in %s iterations." % goalstate)
    G.path.append(0)
    G.path.reverse()
    G.path_state.append([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    # for i in range(1, len(G.path) - 2):
    #     if np.linalg.norm(
    #             G.state_follow[G.path[i + 1]][-1][0][0] - G.state_follow[G.path[i]][-1][0][0]) > 1:
    #         raise ValueError
    #     for j in range(0, steps):
    #         G.path_state.append(
    #             [G.state_follow[G.path[i]][j][0], G.state_follow[G.path[i]][j][1], [0, 0, 0]])
    for i in range(1, len(G.path) - 2):
        # if np.linalg.norm(
        #         G.state_follow[G.path[i + 1]][0][0] - G.state_follow[G.path[i]][0][0]) > 1:
        #     raise ValueError
        G.path_state.append(
            [G.state_follow[G.path[i]][0], G.state_follow[G.path[i]][1], G.state_follow[G.path[i]][2]])
        G.full_path_state.append(G.full_state_follow[G.path[i]])
    G.path_state.append(
        [[goal[0], goal[1], goal[2]], [goal[3], goal[4], goal[5]], [0, 0, 0]])
        #np.save('path_file',G.full_path_state)


        #np.savetxt("path.csv", file_state, delimiter=",")



def main():
    goalstate = 0  # + 1
    best_dist = 10
    for i in range(0, nmax):
        G.expand()
        if i % 100 == 0:
            print(i)
            print(goalstate)
            print(best_dist)
        if G.best_dist < best_dist:
            best_dist = G.best_dist
            goalstate = G.number_of_nodes() - 1
            G.goalstate = goalstate
        if E.in_goal(G.state[0][-1], G.state[2][-1], G.state[4][-1]):
            goal_found = True
        if E.in_goal(G.state[0][-1], G.state[2][-1], G.state[4][-1]) and G.dmax < .25:
            goalstate = G.number_of_nodes() - 1
            G.goalstate = goalstate
            break
    goalpath(goalstate)
    return G.path_state, G.d_track


# run main when RRT is called
if __name__ == '__main__':
    main()
