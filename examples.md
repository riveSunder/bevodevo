# Evo/Devo Examples


## CartPole

CartPole, and more generally inverted pendulum balance tasks, are relatively simple classic control problems. The version incorporated into OpenAI gym has a discrete action space. When using CartPole to test learning algorithms, solving the task doesn't tell you very much but failing does. Take a look at the learning curve for random search to get an idea why that is.

<div align="center">
<img src="./assets/rs_cartpole.png" width=60%>
<em>Random search applied to the CartPole problem with 3 different psuedorandom number generator seeds.</em>
</div> 

You can see in the figure above that my RandomSearch baseline solves CartPole in less than 5 seconds. The experimental settings can be replicated with the call to `bevodevo.train` below.

```
python -m bevodevo.train -a RandomSearch -n CartPole-v1 -g 2500 \
-w 40 -t 499.0 -pi MLPPolicy -p 80 -s 13 1337 42
```

<div align="center">
<img src="./assets/es_cartpole.png" width=60%>
<em>Simple Gaussian evolution strategies applied to CartPole.</em>
</div> 

```
python -m bevodevo.train -a ESPopulation -n CartPole-v1 -g 2500 \
-w 40 -t 499.0 -pi MLPPolicy -p 80 -s 13 1337 42
```

Using simple Gaussian evolution strategies also finds an answer quickly, but it doesn't really gain us anything.  

Watch a trained agent apply its solution policy is pretty exciting. 

<div align="center">
<img src="assets/es_cartpole.gif" width=60%>>
</div> 

## InvertedPendulumBulletEnv

A version of the pole-balancing problem with a continuous action space, implemented in PyBullet. 


<div align="center">
<img src="./assets/es_invertedpendulum.png" width=60%>
</div> 

## InvertedDoublePendulumBulletEnv

A slightly more difficult balancing task can be had by adding a joint to the pole to be balanced. 

<div align="center">
<img src="./assets/cmaes_inverteddoublependulum.png" width=60%>
</div> 

```
python -m bevodevo.train -a CMAESPopulation -n InvertedDoublePendulumBulletEnv-v0 -g 250 \
-w 40 -t 9200.0 -pi MLPPolicy -p 80 -s 13 1337 42
```

<div align="center">
<img src="./assets/cmaes_inverteddoublependulum.gif" width=60%>
</div> 
