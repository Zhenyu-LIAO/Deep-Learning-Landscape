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

In [Neural networks and principal component analysis](https://github.com/Zhenyu-LIAO/Deep-Learning-Landscape/blob/master/references/Neural%20networks%20and%20principal%20component%20analysis-%20Learning%20from%20examples%20without%20local%20minima.pdf), the author studies the autoencoder with one hidden layer and showed the equivalence between the local minimum and the global minimum with a characterization of the form of global minimum points. 

早在1989年， 作者在 [Neural networks and principal component analysis](https://github.com/Zhenyu-LIAO/Deep-Learning-Landscape/blob/master/references/Neural%20networks%20and%20principal%20component%20analysis-%20Learning%20from%20examples%20without%20local%20minima.pdf)中研究了单隐层的线性自编码器中局部最优点和全局最优点的等价性。

In [Deep Learning without Poor Local Minima](https://github.com/Zhenyu-LIAO/Deep-Learning-Landscape/blob/master/references/Deep%20Learning%20without%20Poor%20Local%20Minima.pdf), the author proves the existence of degenerate saddle points for deep linear neural networks with squared loss function and the fact that any local minimum is also a global minimum, with slightly weaker assumptions. The authors  simplify the proof proposed in [Depth Creates No Bad Local Minima](https://github.com/Zhenyu-LIAO/Deep-Learning-Landscape/blob/master/references/Depth%20Creates%20No%20Bad%20Local%20Minima.pdf) and generalize the previous results with fewer assumptions.

NIPS 2016的一篇文章 [Deep Learning without Poor Local Minima](https://github.com/Zhenyu-LIAO/Deep-Learning-Landscape/blob/master/references/Deep%20Learning%20without%20Poor%20Local%20Minima.pdf) 证明了在满秩条件下，多层线性网络的所有局部极小是全局最小。对于多层非线性网络，作者在一个比较强的假设下：网络隐层中各结点的输出相互独立（假设了free？），得到了和多层线性网络一样的结论。[Depth Creates No Bad Local Minima](https://github.com/Zhenyu-LIAO/Deep-Learning-Landscape/blob/master/references/Depth%20Creates%20No%20Bad%20Local%20Minima.pdf) 中作者做了更弱的假设并且简化了证明。

## Hessian特性的问题 + 一些 empirical studies
* We can talk about the Gauss-Newton decomposition here.

In [The Loss Surfaces of Multilayer Networks](https://github.com/Zhenyu-LIAO/Deep-Learning-Landscape/blob/master/references/The%20Loss%20Surfaces%20of%20Multilayer%20Networks.pdf), the authors study the highly non-convex loss function of a simple model of the fully-connected feed-forward neural network with the spherical spin glass model. The following empirical observations are impressive:
* For large-size networks, most local minima are equivalent and yield similar performance on a test set that is of similar nature.
* The probability of finding a "bad" (with a high loss) local minimum is non-zero for small-size networks and decreases quickly as the network size grows large.
* Struggling to find the global minimum on the training set (instead of any of the nuemrous good local ones) is not useful in practice and may lead to overfitting.

在[1]中，作者提出以下经验的结果：
* 对于一个大型神经网络，大多数局部最优点在测试集上具有类似的泛化性能；
* 实际中，刻意地在训练集上寻找全局最优点可能会导致过拟合

In [Singularity of the Hessian in Deep Learning](https://github.com/Zhenyu-LIAO/Deep-Learning-Landscape/blob/master/references/Singularity%20of%20the%20Hessian%20in%20Deep%20Learning.pdf) and [Empirical Analysis of the Hessian of Over-Parametrized Neural Networks](https://github.com/Zhenyu-LIAO/Deep-Learning-Landscape/blob/master/references/Empirical%20Analysis%20of%20the%20Hessian%20of%20Over-Parametrized%20Neural%20Networks.pdf), the authors empirically show that the spectrum of the Hessian is composed of two parts: (1) the bulk centered near zero, (2) and outliers away from the bulk. The bulk depends on the number of parameters merely and indicates how over-parametized the network is. The outliers depend on the data distribution and indicate the complexity of the input data.


在[Singularity of the Hessian in Deep Learning](https://github.com/Zhenyu-LIAO/Deep-Learning-Landscape/blob/master/references/Singularity%20of%20the%20Hessian%20in%20Deep%20Learning.pdf) and [Empirical Analysis of the Hessian of Over-Parametrized Neural Networks](https://github.com/Zhenyu-LIAO/Deep-Learning-Landscape/blob/master/references/Empirical%20Analysis%20of%20the%20Hessian%20of%20Over-Parametrized%20Neural%20Networks.pdf) 中，作者经验性地指出，Hessian矩阵的谱可以由两部分组成：集中在0附件的“bulk”以及远离“bulk”的离群点。其中，“bulk”仅仅由网络结构中的冗余参数决定，而离群点依赖于数据本身的分布。


## Sharp/flat minima -> generalization 问题

In [^3][^4], the authors argue that the flatness of minima of the loss function found by stochastic gradient-based methods will resulst  in good generalization performance. However, in [^5], the authors prove that sharp minima can generalize well when following several definitions of flatness.

## RMT分析深度学习地貌的相关工作
In [The Loss Surfaces of Multilayer Networks](https://github.com/Zhenyu-LIAO/Deep-Learning-Landscape/blob/master/references/The%20Loss%20Surfaces%20of%20Multilayer%20Networks.pdf), the authors study the highly non-convex loss function of a simple model of the fully-connected feed-forward neural network with the spherical spin glass model.

在 [The Loss Surfaces of Multilayer Networks](https://github.com/Zhenyu-LIAO/Deep-Learning-Landscape/blob/master/references/The%20Loss%20Surfaces%20of%20Multilayer%20Networks.pdf) 中，作者使用随机矩阵中经典的 spherical spin glass model 对神经网络高度非凸的损失函数进行建模。


In [Geometry of neural network loss surfaces via random matrix theory](https://github.com/Zhenyu-LIAO/Deep-Learning-Landscape/blob/master/references/Geometry%20of%20neural%20network%20loss%20surfaces%20via%20random%20matrix%20theory.pdf), the authors introduce a new analytical framework for studying the Hessian matrix of a single-hidden-layer network without biases based on free probability and random matrix theory (RMT). A analysis formula is deduced for predicting the the index of critical points.

在 [Geometry of neural network loss surfaces via random matrix theory](https://github.com/Zhenyu-LIAO/Deep-Learning-Landscape/blob/master/references/Geometry%20of%20neural%20network%20loss%20surfaces%20via%20random%20matrix%20theory.pdf) 中，作者引入了一种新的分析工具 free probability， 将单隐层网络的hessian矩阵建模成随机矩阵相加，利用free probability中的R变换给出了负特征值的个数的解析表达式。

[1]: Choromanska A, Henaff M, Mathieu M, et al. The Loss Surfaces of Multilayer Networks\[J\]. Eprint Arxiv, 2014:192-204.

[3]: Hochreiter, Sepp and Schmidhuber, Jürgen. Flat minima. Neural Computation, 9(1):1–42, 1997.

[4]: Keskar, Nitish Shirish, Mudigere, Dheevatsa, Nocedal, Jorge, Smelyanskiy, Mikhail, and Tang, Ping Tak Peter. On largebatch training for deep learning: Generalization gap and sharp minima. In ICLR’2017, arXiv:1609.04836, 2017.

[5]: Dinh L, Pascanu R, Bengio S, et al. Sharp Minima Can Generalize For Deep Nets\[J\]. 2017.

[8]: Baldi P. Linear learning: landscapes and algorithms\[M\]// Advances in neural information processing systems 1. Morgan Kaufmann Publishers Inc. 1988:65-72.

[10]: Kawaguchi K. Deep Learning without Poor Local Minima\[J\]. 2016.

[11]: Lu H, Kawaguchi K. Depth Creates No Bad Local Minima\[J\]. 2017.

