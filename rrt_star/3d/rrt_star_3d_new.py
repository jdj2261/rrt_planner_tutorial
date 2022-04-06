import sys
import math
import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt

from dataclasses import dataclass

class Environment:
    """
    Environment (Map, Obstacles)
    """
    def __init__(
        self, 
        x_min, 
        y_min, 
        z_min,
        x_max, 
        y_max,
        z_max
    ):
        self.x_min = x_min
        self.y_min = y_min
        self.z_min = z_min
        self.x_max = x_max
        self.y_max = y_max
        self.z_max = z_max
        self.obstacles = []

    def add_obstacle(self, obj):
        self.obstacles.extend(obj)

@dataclass
class NodeData:
    COST = 'cost'
    POINT = 'point'

class RRTStar(NodeData):
    """
    RRT path planning
    """

    def __init__(
        self, 
        env,
        start, 
        goal,
        delta_distance=0.5,
        epsilon=0.1,
        max_iter=3000,
        gamma_RRT_star=300, # At least gamma_RRT > delta_distance,
        dimension=2
    ):
        self.env = env
        self.start = start
        self.goal  = goal
        self.delta_dis = delta_distance
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.gamma_RRTs = gamma_RRT_star
        self.d = dimension
        self.tree = self._create_tree()
        
        self.goal_node = None
        self.paths = []

    def _create_tree(self):
        tree = nx.DiGraph()
        tree.add_node(0)
        tree.update(
            nodes=[(0, {NodeData.COST: 0,
                        NodeData.POINT: (self.start)})])
        return tree

    def generate_path(self):
        
        for i in range(self.max_iter):
            x_rand = self.sample_free()
            nearest_node, x_nearest = self.nearest(x_rand)
            x_new = self.steer(x_nearest, x_rand)

            if i % 100 == 0:
                print(f"iter : {i}")

            if self.collision_free(x_nearest, x_new):
                near_nodes = self.near(x_new)

                # V <-- V U {x_new}
                new_node = self.tree.number_of_nodes()
                self.tree.add_node(new_node)

                c_min = self.tree.nodes[nearest_node][NodeData.COST] + self.get_distance(x_nearest, x_new)
                min_node = nearest_node

                for near_node in near_nodes:
                    x_near = self.tree.nodes[near_node][NodeData.POINT]
                    near_cost = self.tree.nodes[near_node][NodeData.COST]
                    if self.collision_free(x_near, x_new):
                        if (near_cost + self.get_distance(x_near, x_new)) < c_min:
                            c_min = near_cost + self.get_distance(x_near, x_new)
                            min_node = near_node
                self.tree.update(nodes=[(new_node, {NodeData.COST: c_min,
                                                     NodeData.POINT: x_new})])
                self.tree.add_edge(min_node, new_node)

                new_cost = self.tree.nodes[new_node][NodeData.COST]
                x_new = self.tree.nodes[new_node][NodeData.POINT]

                for near_node in near_nodes:
                    x_near = self.tree.nodes[near_node][NodeData.POINT]
                    near_cost = self.tree.nodes[near_node][NodeData.COST]
                    
                    if self.collision_free(x_near, x_new):
                        if (new_cost + self.get_distance(x_near, x_new)) < near_cost:
                            parent_node = [node for node in self.tree.predecessors(near_node)][0]
                            self.tree.remove_edge(parent_node, near_node)
                            self.tree.add_edge(new_node, near_node)
                            print("rewire")
                
                if self.reach_to_goal(x_new):
                    self.goal_node = new_node

    def sample_free(self):
        if np.random.random() > self.epsilon:
            rand_point =  np.array([np.random.uniform(self.env.x_min, self.env.x_max),
                              np.random.uniform(self.env.y_min, self.env.y_max),
                              np.random.uniform(self.env.z_min, self.env.z_max)]) 
        else:
            rand_point = self.goal
        return rand_point

    def nearest(self, x_rand):
        distances = [self.get_distance(self.tree.nodes[node][NodeData.POINT], x_rand) for node in self.tree.nodes]
        nearest_node = np.argmin(distances)
        nearest_point = self.tree.nodes[nearest_node][NodeData.POINT]
        return nearest_node, nearest_point

    def get_distance(self, p1, p2):
        return np.linalg.norm(p2-p1)

    def steer(self, x_nearest, x_rand):
        if np.equal(x_nearest, x_rand).all():
            return x_nearest

        vector = x_rand - x_nearest
        dist = self.get_distance(x_rand, x_nearest)
        step = min(self.delta_dis, dist)
        unit_vector = vector / dist
        new_point = x_nearest + unit_vector * step
        return new_point

    def near(self, x_new):
        card_V = len(self.tree.nodes) + 1
        r = self.gamma_RRTs * ((math.log(card_V) / card_V) ** (1/self.d))
        search_radius = min(r, self.gamma_RRTs)
        distances = [self.get_distance(self.tree.nodes[node][NodeData.POINT], x_new) for node in self.tree.nodes]

        near_nodes = []
        for node, dist in enumerate(distances):
            if self.collision_free(self.tree.nodes[node][NodeData.POINT], x_new):
                if dist <= search_radius:
                    near_nodes.append(node)

        return near_nodes

    def reach_to_goal(self, x_new):
        dist = self.get_distance(x_new, self.goal)
        if dist <= 0.5:
            return True
        return False

    def get_rrt_path(self, goal_node=None):
        path = [self.goal]
        if goal_node is None:
            goal_node = self.goal_node

        parent_node = [node for node in self.tree.predecessors(goal_node)][0]
        while parent_node:
            path.append(self.tree.nodes[parent_node][self.POINT])
            parent_node = [node for node in self.tree.predecessors(parent_node)][0]
        path.append(self.start)
        path.reverse()
        return path
        
    def collision_free(self, pointA, pointB):
        for (obs_x, obs_y, obs_z, obs_r) in self.env.obstacles:
            if self.is_inside_circle(obs_x, obs_y, obs_z, obs_r, pointB):
                return False
            if self.is_intersect_circle(obs_x, obs_y, obs_z, obs_r, pointA, pointB):
                return False
        return True

    def is_inside_circle(self, x, y, z, r, point):
        obs_point = np.array([x, y, z])
        distances = self.get_distance(point, obs_point)
        if distances <= r + self.delta_dis:
            return True
        return False

    def is_intersect_circle(self, x, y, z, r, pointA, pointB):
        vectorAB = pointB - pointA
        distanceAB = self.get_distance(pointB, pointA)
        if distanceAB == 0:
            return False

        pointC = np.array([x, y, z])
        vectorAC = pointC - pointA
        proj = np.dot(vectorAC, vectorAB) / distanceAB
        proj = np.clip(proj, 0, 1)

        pointD = pointA + proj * vectorAB
        distancCD = self.get_distance(pointD, pointC)
        
        if distancCD <= r + self.delta_dis:
            return True
        return False

    def get_rrt_tree(self):
        trees = []
        for edge in self.tree.edges:
            from_node = self.tree.nodes[edge[0]][NodeData.POINT]
            goal_node = self.tree.nodes[edge[1]][NodeData.POINT]
            trees.append((from_node, goal_node))
        return trees

def init_3d_figure(name=None):
    """
    Initializes 3d figure
    """
    fig = plt.figure(name)
    ax = fig.add_subplot(111, projection='3d')
    return fig, ax

def plot_sphere(ax=None, radius=1.0, p=np.zeros(3), n_steps=20, alpha=1.0, color="k"):
    """
    Plot sphere
    """
    phi, theta = np.mgrid[0.0:np.pi:n_steps * 1j, 0.0:2.0 * np.pi:n_steps * 1j]
    x = p[0] + radius * np.sin(phi) * np.cos(theta)
    y = p[1] + radius * np.sin(phi) * np.sin(theta)
    z = p[2] + radius * np.cos(phi)
    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0)
if __name__ == "__main__":

    fig, ax = init_3d_figure("RRT STAR")
    env = Environment(x_min=-20, y_min=-20, z_min=-20, x_max=20, y_max=20, z_max=20)

    spheres = []
    radius = 5
    for i in range(20):
        x = random.choice([i for i in range(-10, 10)])
        y = random.choice([i for i in range(-10, 10)])
        z = random.choice([i for i in range(-10, 10)])
        spheres.append((x, y, z, radius))
    env.add_obstacle(spheres)

    start_point = np.array([-20, 20, -20])
    goal_point  = np.array([20, -20, 20])

    planner = RRTStar( env, 
                       start=start_point, 
                       goal=goal_point, 
                       delta_distance=5,
                       gamma_RRT_star=100,
                       epsilon=0.2, 
                       max_iter=1000)

    planner.generate_path()
    path = planner.get_rrt_path()
    tree = planner.get_rrt_tree()

    ax.scatter([x for (x, y, z) in path], [y for (x, y, z) in path], [z for (x, y, z) in path], s=10, c='r')
    ax.plot([x for (x, y, z) in path], [y for (x, y, z) in path], [z for (x, y, z) in path], '-b', linewidth=2,)
    ax.text(path[0][0], path[0][1], path[0][2], 'Start', verticalalignment='bottom', horizontalalignment='center', size="20")
    ax.text(path[-1][0], path[-1][1], path[-1][2],'Goal', verticalalignment='bottom', horizontalalignment='center', size="20")
    
    # # Plot
    for sp_x, sp_y, sp_z, sp_r in spheres:
        sp_radius = sp_r
        sp_pos = np.array([sp_x, sp_y, sp_z])
        plot_sphere(ax, radius=radius, p=sp_pos, alpha=0.2, color="k")
    plt.show()