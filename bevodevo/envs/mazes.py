import numpy as np
import matplotlib.pyplot as plt
import time

import gym

class MazeEnv(gym.Env):

    def __init__(self, visible_rewards=False, use_colors=False,  num_envs=1):

        # TODO: vectorized environments
        self.num_envs = num_envs

        self.visible_rewards = visible_rewards
        self.use_colors = use_colors

        self.action_space = self.my_action_space(self.num_envs)
        self.observation_space = self.my_observation_space(\
                visible_rewards=self.visible_rewards,\
                use_colors=self.use_colors, num_envs=self.num_envs)

        self.max_steps = 100

        self.action_effects = np.array(\
                [[-1, 0],\
                [0, -1],\
                [1, 0],\
                [0, 1],\
                [0, 0]])

    class my_action_space():

        def __init__(self, num_envs = 1):

            self.num_envs = num_envs
            self.shape = (self.num_envs,4)
            self.visible_rewards = True

        def sample(self):
            
            return  np.argmax((np.random.random(self.shape)), axis=1)

    class my_observation_space():
        """
        All maze environments return a Von Neumann neighborhood observation
        channels 0 to 2 inclusive are a 3 element 'color' w/ values 0.0-1.0
        channels 3 and 4 are either 1. or 0. and indicate the presence 
            of a wall or reward, respectively.
        """

        def __init__(self, visible_rewards=True, use_colors=False, num_envs=1):

            self.visible_rewards = visible_rewards
            self.colored_squares = use_colors
            self.num_envs = num_envs

            self.max = 1.0
            self.min = 0.0
            self.shape = (self.num_envs,4,5)

        def sample(self):
            obs = np.random.random(self.shape)

            # last two channels indicate presence of wall or reward
            obs[:,:,3] = 1.0 * (obs[:,:,3] > 0.5)
            obs[:,:,4] = 1.0 * (self.visible_rewards and (obs[:,:,4] > 0.05))
            # no rewards in walls
            obs[:,:,4] = obs[:,:,4] * (1.0 - obs[:,:,3])
            
            obs[:,:,0:3] *= float(self.colored_squares)

            return obs

    def generate_maze(self):
        self.maze = np.zeros((self.num_envs, 7, 7, 6))

    def get_observation(self):

        obs = np.zeros(self.observation_space.shape)
        for ii in range(self.num_envs):
            
            # get info about Von Neumann neighborhood
            # corresponding to WASD key order ^, <-, v, ->

            obs_index = 0
            for offset in [[-1, 0], [0,-1], [1,0], [0,1]]:

                check_x = int(self.agent_location[ii,0] + offset[0])
                check_y = int(self.agent_location[ii,1] + offset[1])

                obs[ii,obs_index,:] = self.maze[ii, check_x, check_y, 0:5]
                obs_index += 1

        return obs

    def reset(self):

        self.step = 0
        self.generate_maze()
        obs = self.get_observation()

        return obs

    def move_agents(self):

        self.maze[:,:,:,5] *= 0
        for ii in range(self.num_envs):
            self.maze[ii, int(self.agent_location[ii,0]),\
                    int(self.agent_location[ii,1]), 5] += 1.0

    def step(self, action):

        assert self.reward_location is not None, \
                "env.reset() must be called before env.step()"
        
        # TODO: vectorize wall checks and movement

        collisions = np.zeros((self.num_envs,1))

        for ii in range(self.num_envs):

            movement = self.action_effects[action[ii]]
            move_to = self.agent_location[ii] + movement

            if self.maze[ii, int(move_to[0]), int(move_to[1]), 3]:
                collisions[ii] = 0.1
            else:
                self.agent_location[ii, :] = move_to

        self.move_agents()

        obs = self.get_observation()

        reward = [10.0 * (elem1 == elem2).all()\
                for elem1, elem2 in \
                zip(self.reward_location, self.agent_location)]

        reward = np.array(reward).reshape(self.num_envs,1)
        reward -= collisions + 0.01

        self.steps += 1
        done = self.steps >= self.max_steps

        info = {}

        return obs, reward, done, info

    def render(self, ii=0):
        print(env.maze[ii,:,:,3] \
                + env.maze[ii,:,:,4] * 10.\
                + env.maze[ii,:,:,5] * 4.0)


class TMazeEnv(MazeEnv):

    def __init__(self, num_envs=1):
        super(TMazeEnv, self).__init__(num_envs)

        self.left_probability = 0.5

        self.reward_location = None
        self.agent_location = None


    def generate_maze(self):

        self.maze = np.zeros((self.num_envs, 7, 7, 6))

        # add walls and carve out maze
        self.maze[:,:,:,3] += 1.0
        maze_coordinates = [\
                [1,1],\
                [1,2],\
                [1,3],\
                [1,4],\
                [1,5],\
                [2,3],\
                [3,3],\
                [4,3],\
                [5,3]]
        for carve_maze in maze_coordinates:
            self.maze[:, carve_maze[0], carve_maze[1], 3] = 0.0

        for ii in range(self.num_envs):

            if self.reward_sides[ii]:
                self.maze[ii,1,1,4] = 1.0
            else:
                self.maze[ii,1,5,4] = 1.0

            self.maze[ii, int(self.agent_location[ii,0]),\
                    int(self.agent_location[ii,1]), 5] += 1.0

    def reset(self):

        self.steps = 0

        # agents start at the bottom of the T-maze
        self.agent_location = np.array([5, 3]) * np.ones((self.num_envs, 2))

        # rewards are placed to left or right
        self.reward_sides = np.random.randint(2, size=(self.num_envs,1))

        self.reward_location = [\
                elem * np.array([1,1]) + (1 - elem) * np.array([1,5])\
                for elem in self.reward_sides]

        self.generate_maze()
        
        obs = self.get_observation()

        return obs
    
class MorrisMazeEnv(MazeEnv):

    def __init__(self, visible_rewards=False, use_colors=True, num_envs=1):
        super(MorrisMazeEnv, self).__init__(\
                visible_rewards, use_colors, num_envs)

    def generate_maze(self):

        self.maze = np.zeros((self.num_envs, 11, 11, 6))
        self.maze[:,:,:,0:3] = np.random.random(\
                (self.num_envs, 11, 11, 3))

        # build walls
        for jj in [0, 10]:
            self.maze[:, jj, :, 3] = 1.0 
            self.maze[:, :, jj, 3] = 1.0 

        for ii in range(self.num_envs):

            self.maze[ii, int(self.reward_location[ii,0]),\
                    int(self.reward_location[ii,1]), 4] += 1.0

            self.maze[ii, int(self.agent_location[ii,0]),\
                    int(self.agent_location[ii,1]), 5] += 1.0

    def get_observation(self):

        obs = np.zeros(self.observation_space.shape)

        # get info about Von Neumann neighborhood
        # corresponding to WASD key order ^, <-, v, ->
        
        # in the Morriz maze, observations correspond to the wall colors
        # in a cross shape centered on the agent location
        obs[:, 0, 0:3] = self.maze[\
                :, 0, self.agent_location[:,1], 0:3]
        obs[:, 1, 0:3] = self.maze[\
                :, self.agent_location[:,0], 0, 0:3]
        obs[:, 2, 0:3] = self.maze[\
                :, -1, self.agent_location[:,1], 0:3]
        obs[:, 3, 0:3] = self.maze[\
                :, self.agent_location[:,0], -1, 0:3]

        obs[:, 0, 3:5] = self.maze[\
                :, self.agent_location[:,0]-1, self.agent_location[:,1], 3:5]
        obs[:, 1, 3:5] = self.maze[\
                :, self.agent_location[:,0], self.agent_location[:,1]-1, 3:5]
        obs[:, 2, 3:5] = self.maze[\
                :, self.agent_location[:,0]+1, self.agent_location[:,1], 3:5]
        obs[:, 3, 3:5] = self.maze[\
                :, self.agent_location[:,0], self.agent_location[:,1]+1, 3:5]

        return obs

    def reset(self):

        self.steps = 0

        # agents start at a random location
        self.agent_location = np.random.randint(\
                low=1, high=9, size=(self.num_envs, 2))

        # rewards are placed at a random location
        self.reward_location = np.random.randint(\
                low=1, high=9, size=(self.num_envs, 2))

        self.generate_maze()
        
        obs = self.get_observation()

        return obs

class NavigationMazeEnv(MazeEnv):

    def __init__(self, num_envs=1):
        super(NavigationMazeEnv, self).__init__(num_envs)

    def generate_maze(self):

        self.maze = np.zeros((self.num_envs, 11, 11, 6))
        if self.use_colors:
            self.maze[:,:,:,0:3] += np.random.random(\
                    (self.num_envs, 11, 11, 3))

        # build walls
        for jj in [0, 10]:
            self.maze[:, jj, :, 3] = 1.0 
            self.maze[:, :, jj, 3] = 1.0 

        # build inner walls/immovable blocks
        self.maze[:, 2:-1:2, 2:-1:2, 3] += 1.0
        

        for ii in range(self.num_envs):

            self.maze[ii, int(self.reward_location[ii,0]),\
                    int(self.reward_location[ii,1]), 4] += 1.0

            self.maze[ii, int(self.agent_location[ii,0]),\
                    int(self.agent_location[ii,1]), 5] += 1.0

    def reset(self):

        self.steps = 0

        # agents start at a random location
        self.agent_location = np.random.randint(\
                low=1, high=9, size=(self.num_envs, 2))

        # rewards are placed at a random location
        self.reward_location = np.random.randint(\
                low=1, high=9, size=(self.num_envs, 2))

        self.generate_maze()

        for ii in range(self.num_envs):
            # Move agents and rewards that are stuck in walls 

            location = self.agent_location[ii]
            if self.maze[ii, location[0], location[1], 3]:
                self.agent_location[ii, 0] -= 1

            reward_location = self.reward_location[ii]
            if self.maze[ii, reward_location[0], reward_location[1], 3]:
                self.reward_location[ii, 0] -= 1

        obs = self.get_observation()

        return obs


if __name__ =="__main__":

    # run tests

    for maze_env in [TMazeEnv]:#[MazeEnv, TMazeEnv, MorrisMazeEnv, NavigationMazeEnv]:

        env = maze_env()
        
        obs = env.observation_space.sample()
        action = env.action_space.sample()
        print("sampled obs and action shapes: ", obs.shape, action.shape)
        print("sampled obs and action: ", obs, action)

        obs = env.reset()

        obs, reward, done, info = env.step(action)

        print(env.maze[0,:,:,3] \
                + env.maze[0,:,:,4] * 10.\
                + env.maze[0,:,:,5] * 4.0)


    env = NavigationMazeEnv()
    obs = env.reset()
    
    for step in range(10):

        time.sleep(0.5)
        obs, reward, done, info = env.step(env.action_space.sample())
        env.render()
        print("reward: ", reward)
        print("obs: \n", obs)

    env = MorrisMazeEnv()
    obs = env.reset()
    
    for step in range(10):

        time.sleep(0.5)
        obs, reward, done, info = env.step(env.action_space.sample())
        env.render()
        print("reward: ", reward)
        print("obs: \n", obs)

    env = TMazeEnv()
    obs = env.reset()
    
    for action in [0, 0, 0, 0, 1, 1, 3, 3, 3, 3]:

        time.sleep(0.5)
        obs, reward, done, info = env.step(np.array([action]))
        env.render()
        print("reward: ", reward)
        print("obs: \n", obs)


    import pdb; pdb.set_trace()
    print("OK")
