# EXample 16: Monte Carlo tree search on ```Connect2```

In this example we will go over the <a href="https://en.wikipedia.org/wiki/Monte_Carlo_tree_search">Monte Carlo tree search algorithm</a> 
or MCTS for short.

----
**Remark**

A concise presentation of the algorithm can be found in the following video: https://www.youtube.com/watch?v=UXW2yZndl7U

The following videos from Udacity's <a href=" https://www.udacity.com/course/ud600">Reinforcement Learning </a> also explain
the MCTS algorithm 

1. <a href="https://www.youtube.com/watch?v=onBYsen2_eA&list=PL_W9hg3Zoi8dIQMkb19tj-lYwAajhg5YJ">Monte Carlo Tree Search p1</a>
2. <a href="https://www.youtube.com/watch?v=4rVXxSD2DIM&list=PL_W9hg3Zoi8dIQMkb19tj-lYwAajhg5YJ&index=2">Monte Carlo Tree Search p2</a>
3. <a href="https://www.youtube.com/watch?v=EGN1KAjtNS4&list=PL_W9hg3Zoi8dIQMkb19tj-lYwAajhg5YJ&index=3">Monte Carlo Tree Search p3</a>
4. <a href="https://www.youtube.com/watch?v=2SeEQMfM3Ug&list=PL_W9hg3Zoi8dIQMkb19tj-lYwAajhg5YJ&index=4">Monte Carlo Tree Search p4</a>


In this tutorial we will be applying MCTS in a toy environment namely Connect2. The following video is a nice discussion
on the application of MCTS on Connect2: <a href="https://www.youtube.com/watch?v=62nq4Zsn8vc">Alpha Zero and Monte Carlo Tree Search</a>.

----

Overall, the vanilla  MCTS algorithm has four steps

1. Tree tranversal 
2. Node expansion
3. Rollout or random simulation
4. Backward propagation

Let's go over each one of these steps.

### Tree traversal

At this step the agent select the root node from which it will recursively select the best paths.
This is done until a leaf node is reached. The selction of a node is typically done using the 
<a href="https://www.turing.com/kb/guide-on-upper-confidence-bound-algorithm-in-reinforced-learning">UCB</a> (or upper confidence bound) criterion
but you can use any criterion that is suitable for your application but make sure that the criterion you are using
balances exploration and exploitation. The UCB criterion is as follows



Where $N$ is the number of node visits
The UCB criterion limits the extent that each each branch will be explored.
This is done because as $N$ increases beacuse we visit the node more and more, 
the overall factor continuously decreases thus the expected reward from the current branch will also decrease.

A new node is created when we execute  the rollout step.



### Rollout step 

The MCTS starts with an empty tree data structure and iteratively will build a portion of the tree by running a number of simulations
Each simulation will add a single node to the tree. Since a simulation is the exploration step, the more simualtions we run, the beter
we expect the model  to perform. However, simulating indefinitetly is not possible for obvious reasons.

The MCTS algorithm implies that somehow we have a way of doing the simulation step. One way is if we know the transition dynamics of the 
environment or being able to sample from the environment.

### Backward propagation

During the backward step we update the statistics for every node in the search path. These statistics affect the UCB score and will
guide the next MCTS simulation.

Now that we have gone over the detaisl of the MCTS algorithm let's try to answer some practical questions. Fo example when does the algorithm stop?
There are various ways to address this question and the most common one is to simulate up until a specific number of iterations.

All in all, the MCTS algorithm is summarised below.

In this 

## Driver code


## References





