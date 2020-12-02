# environments
from bevodevo.envs import landscapes

# policies
from bevodevo.policies import base, params

# algorithms
#from bevodevo.algos import 

# register envs
from gym.envs.registration import register
import bevodevo.envs

register(id="OptimizationEnv-v0", entry_point="bevodevo.envs.landscapes:OptimizationEnv")
register(id="TMazeEnv-v0", entry_point="bevodevo.envs.mazes:TMazeEnv")
register(id="MorrisMazeEnv-v0", entry_point="bevodevo.envs.mazes:MorrisMazeEnv")
register(id="NavigationMazeEnv-v0", entry_point="bevodevo.envs.mazes:NavigationMazeEnv")
