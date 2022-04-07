import sys
import math
import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt

from dataclasses import dataclass
from networkx.drawing.nx_agraph import graphviz_layout

class Environment:
    """
    Environment (Map, Obstacles)
    """
    def __init__(
        self, 
        x_min, 
        y_min, 
        x_max, 
        y_max
    ):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
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

        self.search_radius = gamma_RRT_star
        
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
        last_plt = None
        plt.plot(self.start[0],self.start[1],'*g',self.goal[0], self.goal[1],'*r', markersize=12)
        plt.title("RRT Star with Obstacles")
        plt.xlabel("X-Position")
        plt.ylabel("Y-Position")
        plt.xlim(self.env.x_min-5, self.env.x_max+5)
        plt.ylim(self.env.y_min-5, self.env.y_max+5)

        for circle in self.env.obstacles:
            plot_circle(circle[0], circle[1], circle[2])

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

                plt.plot([self.tree.nodes[nearest_node][NodeData.POINT][0],x_new[0]], [self.tree.nodes[nearest_node][NodeData.POINT][1],x_new[1]], 'r--', linewidth=0.5,)
                plt.scatter(x_new[0], x_new[1], s=10, c = 'b')

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

                plt.plot([self.tree.nodes[min_node][NodeData.POINT][0], x_new[0]], [self.tree.nodes[min_node][NodeData.POINT][1],x_new[1]], 'k', linewidth=0.5,)
                plot_circle(x_new[0], x_new[1], self.search_radius, "k--", linewidth=0.1)
                plt.pause(0.001)

                if self.reach_to_goal(x_new):
                    self.goal_node = new_node
                    path = self.get_rrt_path()
                    self.paths.append(path)
                    if last_plt is None:
                        pass
                    else:
                        l = last_plt.pop(0)
                        l.remove()
                    plt_path = plt.plot([x for (x, y) in path], [y for (x, y) in path], 'g', linewidth=2,)
                    init_path = plt.plot([x for (x, y) in self.paths[0]], [y for (x, y) in self.paths[0]], 'r', linewidth=1,)
                    last_plt = plt_path

        plt.plot([x for (x, y) in path], [y for (x, y) in path], '-b', linewidth=4,)
        plt.show(block=False)
        plt.pause(1)
        plt.close()

    def sample_free(self):
        if np.random.random() > self.epsilon:
            rand_point = np.array([np.random.uniform(self.env.x_min, self.env.x_max),
                              np.random.uniform(self.env.y_min, self.env.y_max)]) 
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

    def near(self, x_rand):
        card_V = len(self.tree.nodes) + 1
        r = self.gamma_RRTs * ((math.log(card_V) / card_V) ** (1/self.d))
        self.search_radius = min(r, self.gamma_RRTs)
        distances = [self.get_distance(self.tree.nodes[node][NodeData.POINT], x_rand) for node in self.tree.nodes]

        near_nodes = []
        for node, dist in enumerate(distances):
            if self.collision_free(self.tree.nodes[node][NodeData.POINT], x_rand):
                if dist <= self.search_radius:
                    near_nodes.append(node)

        return near_nodes

    def reach_to_goal(self, x_new):
        dist = self.get_distance(x_new, self.goal)
        if dist <= 0.5:
            return True
        return False

    def get_rrt_path(self):
        path = [self.goal]
        parent_node = [node for node in planner.tree.predecessors(self.goal_node)][0]
        while parent_node:
            path.append(planner.tree.nodes[parent_node][planner.POINT])
            parent_node = [node for node in planner.tree.predecessors(parent_node)][0]
        path.append(self.start)
        path.reverse()
        return path
        
    def collision_free(self, pointA, pointB):
        m = 0
        b = 0
        if pointA[0] == pointB[0]:
            x = pointA[0]
        elif pointA[1] == pointB[1]:
            y = pointA[0]
        else:
            m = (pointB[1]-pointA[1]) / (pointB[0]-pointA[0])
            b = pointA[1] - (m*pointA[0])

        for (obs_x, obs_y, obs_r) in self.env.obstacles:
            if self.is_inside_circle(obs_x, obs_y, obs_r, pointB):
                return False
            if pointA[0] == pointB[0]:
                d = abs(pointB[0] - obs_x)
            elif pointA[1] == pointB[1]:
                d = abs(pointB[1] - obs_y)
            else:
                d = abs(m*obs_x - obs_y + b) / np.sqrt(m**2 + 1)
            # print(d)
            if d < obs_r:
                return False
        return True

    def is_inside_circle(self, x, y, r, point):
        obs_point = np.array([x, y])
        distances = self.get_distance(point, obs_point)
        if distances <= r**2:
            return True
        return False

    def is_intersect_circle(self, x, y, r, pointA, pointB):
        vectorAB = pointB - pointA
        distanceAB = self.get_distance(pointB, pointA)
        if distanceAB == 0:
            return False

        pointC = np.array([x, y])
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


def plot_circle(x, y, size, color="-b", linewidth=1):
    deg = list(range(0, 360, 5))
    deg.append(0)
    xl = [x + size * np.cos(np.deg2rad(d)) for d in deg]
    yl = [y + size * np.sin(np.deg2rad(d)) for d in deg]
    plt.plot(xl, yl, color, linewidth=linewidth,)


if __name__ == "__main__":

    if len(sys.argv) > 1:
        K = int(sys.argv[1])
        print(K)
    else:
        K = 300
        
    env = Environment(x_min=-20, y_min=-20, x_max=20, y_max=20)

    circles = []
    radius = 3
    for i in range(10):
        x = random.choice([i for i in range(-10, 10)])
        y = random.choice([i for i in range(-10, 10)])
        circles.append((x, y, radius))
    env.add_obstacle(circles)

    start_point = np.array([-20, 20])
    goal_point  = np.array([20, -20])

    planner = RRTStar( env, 
                       start=start_point, 
                       goal=goal_point, 
                       delta_distance=5,
                       gamma_RRT_star=30,
                       epsilon=0.2, 
                       max_iter=1000)

    planner.generate_path()
    path = planner.get_rrt_path()
    tree = planner.get_rrt_tree()

    
    plt.scatter([x for (x, y) in path], [y for (x, y) in path], s=55, c = 'b')
    plt.plot([x for (x, y) in path], [y for (x, y) in path], '-b', linewidth=4,)
    plt.text(path[0][0], path[0][1], 'Start', verticalalignment='bottom', horizontalalignment='center', size="20")
    plt.text(path[-1][0], path[-1][1], 'Goal', verticalalignment='bottom', horizontalalignment='center', size="20")
    

    # # Plot
    for circle in circles:
        plot_circle(circle[0], circle[1], circle[2])
    
    
    for vertex in tree:
        plt.plot([x for (x, y) in vertex],[y for (x, y) in vertex], 'k', linewidth=1,)
    plt.show()
        # plt.plot([x for (x, y) in planner.paths[0]],[y for (x, y) in planner.paths[0]], 'k', linewidth=1,)

    # if path is None:
    #     print("cannot create path")
    # else:
    #     plt.scatter([x for (x, y) in path], [y for (x, y) in path], s=55, c = 'b')
    #     plt.plot([x for (x, y) in path], [y for (x, y) in path], '-b', linewidth=4,)
    #     plt.plot([x for (x, y) in planner.paths[0]], [y for (x, y) in planner.paths[0]], '-r', linewidth=2,)
    #     plt.text(path[0][0], path[0][1], 'Start', verticalalignment='bottom', horizontalalignment='center', size="20")
    #     plt.text(path[-1][0], path[-1][1], 'Goal', verticalalignment='bottom', horizontalalignment='center', size="20")
    # plt.show()