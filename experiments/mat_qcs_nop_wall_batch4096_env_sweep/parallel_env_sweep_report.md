# No. Parallel Environments

# vs 

# Episode Segment Length

This experiment sets out to test if MAT-QCS on the swarm-bots wall task benefits more from batches made up of a large amount of parallel environments or of long episode segments. We fix the batch size $B = 4096$ and iterate through combinations of parallel environments $P$ and episode segment lengths $T$ where $B = P \times T$. Concretely, the following combinations were used: 

* $32 \times 128$
* $128 \times 32$
* $512 \times 8$
* $1024 \times 4$
* $2048 \times 2$

For the last combination it's important to note that the next-observation-prediction auxiliary loss normally predicts up to 4 steps into the future. Since we have $T = 2$, we can only predict up to two steps.



## Results

![image-20260507172029410](C:\Users\Brn\AppData\Roaming\Typora\typora-user-images\image-20260507172029410.png)

![image-20260507172058381](C:\Users\Brn\AppData\Roaming\Typora\typora-user-images\image-20260507172058381.png)

Although we do not have many runs per category, a clear trend is visible. Higher $P$ leads to more stable and faster learning. $2048\times2$ learns slightly faster in the beginning compared to $1024\times4$ but ends up with a lower final return.

## Interpretation

While long episode segment lengths increase the accuracy of advantage and value estimates, MAT-QCS in the swarm-bots wall scenario appears to favor the diversity that comes from having many parallel environments. Having many environments also means fewer steps in the batch are correlated with each other. This likely reduces batch bias and thus leads to more stable learning.

## Next Steps 

This experiment suggests that MAT-QCS favors batches with high diversity. The obvious next step would be to increase the batch size $B$ to make room for higher $P$ e.g $2048\times4$. However, [MAPPO](https://arxiv.org/abs/2103.01955) has shown that using excessively large batch sizes might reduce sample efficiency, so we will have to test where the ceiling will be.