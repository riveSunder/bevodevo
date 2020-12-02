import numpy as np
import matplotlib.pyplot as plt
import gym

import time

class OptimizationEnv(gym.Env):

    def __init__(self, fn_name="Rastrigin", agent_mode=False, shuffle=False):

        self.max_steps = 32

        if "astrigi" in fn_name:
            self.fn_name = "rastrigin"
        elif "osenbro" in fn_name:
            self.fn_name = "rosenbrock"
        elif "atyas" in fn_name:
            self.fn_name = "matyas"
        elif "immmel" in fn_name:
            self.fn_name= "himmelblau"
        else:
            self.fn_name = "quadratic"

        self.fn_dict = {"rastrigin": [ lambda x,y:\
                    20 + (x**2 + y**2) - 10 * (np.cos(2*np.pi * x) + np.cos(2*np.pi*y)),\
                    -5.12, 5.12, -5.12, 5.12],\
                "rosenbrock": [lambda x,y:\
                    (1-x)**2 + 100 * (y - x**2)**2,\
                    -4.0, 4.0, -4.0, 4.0],\
                "matyas": [lambda x,y:\
                    0.26 * (x**2 + y**2) - 0.48 * x * y,\
                    -5.12, 5.12, -5.12, 5.12],\
                "himmelblau": [lambda x,y:\
                    (x**2 + y - 11)**2 + (x + y**2 - 7)**2,
                    -6.0, 6.0, -6.0, 6.0],
                "quadratic": [lambda x,y:\
                    x**2 + y**2,\
                    -5.12, 5.12, -5.12, 5.12]\
                }

        self.shuffle = shuffle
        self.action_space = self.my_action_space()
        self.observation_space = self.my_observation_space()

        self.reset_actions()

    class my_action_space():

        def __init__(self):

            self.range = [(-1.0, 1.0), (-1.0, 1.0)]
            self.shape = (2,)

        def sample(self):

            return 2 * (np.random.random((2)) - 0.5) 

        def sample_many(self, n = 64):

            return [  2 * (np.random.random((2)) - 0.5) for ii in range(n)] 

    class my_observation_space():

        def __init__(self):

            self.range = (-1e5, 0.0)
            self.max = 0.0
            self.min = -1.e-5
            self.shape = (1,)

        def sample(self):

            return self.min * (np.random.random((1))) 

        def sample_many(self, n = 64):

            return [  self.min * (np.random.random((1))) for ii in range(n)] 

    def parse_coords(self, coords):

        #min_x, max_x = self.fn_dict[self.fn_name][1], self.fn_dict[self.fn_name][2]
        #min_y, max_y = self.fn_dict[self.fn_name][3], self.fn_dict[self.fn_name][4]

        #coords = [np.tanh(elem) for elem in coords]
        #coord_x = (max_x - min_x)/2 * (coords[0] ) + (max_x + min_x) / 2
        #coord_y = (max_y - min_y)/2 * (coords[1] ) + (max_y + min_y) / 2

        coord_x, coord_y = coords[0]+self.offset[0], coords[1]+self.offset[1]

        return [coord_x, coord_y]

    def reset(self):

        self.steps = 0
        
        if self.shuffle:
            self.fn_name = np.random.choice(list(self.fn_dict.keys()), \
                    p=[1/len(self.fn_dict.keys()) for ii in range(len(self.fn_dict.keys()))])

        self.offset = self.action_space.sample()
        coord_x, coord_y = self.parse_coords([0.0, 0.0])

        reward = -self.fn_dict[self.fn_name][0](coord_x, coord_y)
        obs = np.append(reward, np.array([0.0,0.0]))
        
        self.reset_actions()
        self.action.append([coord_x, coord_y])

        return obs

    def reset_actions(self):
        self.action = []

    def step(self, action):

        if type(action) == list and action[0].shape[0] == 2:
            multi = True
        else: 
            multi = False

        info = {}
        if multi:
            info["num_actions"] =  len(action)
            reward = []
            obs = []
            for coords in action:
                [coord_x, coord_y] = self.parse_coords(coords)
                reward.append( - self.fn_dict[self.fn_name][0](coord_x, coord_y))
                self.action.append([coord_x, coord_y])

                obs.append(np.append(reward, coords))


        else:
            [coord_x, coord_y] = self.parse_coords(action)
            reward = - self.fn_dict[self.fn_name][0](coord_x, coord_y)
            self.action.append([coord_x, coord_y])

            obs = np.append(reward, action)

        info["avg_reward"] = np.mean(reward)

        self.steps += 1
        if self.steps >= self.max_steps:
            done = True
        else:
            done = False


            
        return obs, reward, done, info


    def render(self, save_it=False, tag=0):

        x = np.linspace(self.fn_dict[self.fn_name][1],self.fn_dict[self.fn_name][2],256)
        y = np.linspace(self.fn_dict[self.fn_name][3],self.fn_dict[self.fn_name][4],256)

        xx, yy = np.meshgrid(x,y)
        
        plt.figure(figsize=(6,6))


        plt.imshow(- self.fn_dict[self.fn_name][0](xx, -yy), extent=(self.fn_dict[self.fn_name][1],\
                self.fn_dict[self.fn_name][2], \
                self.fn_dict[self.fn_name][3], \
                self.fn_dict[self.fn_name][4]),\
                cmap="plasma")

        my_contour = - self.fn_dict[self.fn_name][0](xx,yy)
                
        plt.contour(xx,yy, my_contour,\
                levels = [-elem for elem in \
                np.logspace(np.log10(-np.min(my_contour)), np.log10(1e-3 + -np.max(my_contour)), 8)],\
                cmap="twilight")

        for coords in self.action:
            plt.plot(coords[0], coords[1], "o", mfc=[0.85, 0.5, 0.0], markeredgecolor=[0.0,0.0,0.0])

        plt.axis([self.fn_dict[self.fn_name][1], self.fn_dict[self.fn_name][2],\
                self.fn_dict[self.fn_name][3], self.fn_dict[self.fn_name][4]])
        if save_it:
            plt.savefig("./results/indirect/cmaes_step{}_{}.png".format(tag, self.fn_name))
        else:
            plt.show()

        plt.close()

if __name__ == "__main__":

    # run tests
    
    
    print("OK")
