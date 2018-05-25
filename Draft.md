# Deep Learning Landscape
An (incomplete) overview of recent advances on the topic of Deep Learning Landscape. （我们用中文还是英文写？）

## 引言: 深度学习中的地貌(landscape)问题
* 深度学习中的地貌定义
* 重要性
* 理论分析难点
* 基本概念: 梯度(Jacobian), Hesian, 全局最小, 局部最小, 鞍点 (+图)
* 深度学习中的优化: (随机)梯度下降...

## 对于深度学习中的地貌的描述的进展
* linear network
* deep learning without poor local minima

* single layer network
* multi-layer network

In [8], the author studies the autoencoder with one hidden layer and showed the equivalence between the local minimum and the global minimum with a characterization of the form of global minimum points. 

In [^10], the author proves the existence of degenerate saddle points for deep linear neural networks with squared loss function and the fact that any local minimum is also a global minimum, with slightly weaker assumptions. The authors  simplify the proof proposed in [^11] and generlize the previous results with fewer assumptions.

## Hessian特性的问题 + 一些 empirical studies
In [^1], the authors study the highly non-convex loss function of a simple model of the fully-connected feed-forward neural network with the spherical spin glass model. The following empirical observations are impressive:
* For large-size networks, most local minima are equivalent and yield similar performance on a test set that is of similar nature.
* The probability of finding a "bad" (with a high loss) local minimum is non-zero for small-size networks and decreases quickly as the network size grows large.
* Struggling to find the global minimum on the training set (instead of any of the nuemrous good local ones) is not useful in practice and may lead to overfitting.

在[^1]中，作者提出以下经验的结果：
* 对于一个大型神经网络，大多数局部最优点在测试集上具有类似的泛化性能；
* 实际中，刻意地在训练集上寻找全局最优点可能会导致过拟合
## Sharp/flat minima -> generalization 问题

In [^3][^4], the authors argue that the flatness of minima of the loss function found by stochastic gradient-based methods will resulst  in good generalization performance. However, in [^5], the authors prove that sharp minima can generalize well when following several definitions of flatness.
## RMT分析深度学习地貌的相关工作


[^1]: Choromanska A, Henaff M, Mathieu M, et al. The Loss Surfaces of Multilayer Networks\[J\]. Eprint Arxiv, 2014:192-204.

[^3]: Hochreiter, Sepp and Schmidhuber, Jürgen. Flat minima. Neural Computation, 9(1):1–42, 1997.

[^4]: Keskar, Nitish Shirish, Mudigere, Dheevatsa, Nocedal, Jorge, Smelyanskiy, Mikhail, and Tang, Ping Tak Peter. On largebatch training for deep learning: Generalization gap and sharp minima. In ICLR’2017, arXiv:1609.04836, 2017.

[^5]: Dinh L, Pascanu R, Bengio S, et al. Sharp Minima Can Generalize For Deep Nets\[J\]. 2017.

[^8]: Baldi P. Linear learning: landscapes and algorithms\[M\]// Advances in neural information processing systems 1. Morgan Kaufmann Publishers Inc. 1988:65-72.

[^10]: Kawaguchi K. Deep Learning without Poor Local Minima\[J\]. 2016.

[^11]: Lu H, Kawaguchi K. Depth Creates No Bad Local Minima\[J\]. 2017.

