# Deep Learning Landscape
An (incomplete) overview of recent advances on the topic of Deep Learning **Landscape**. 

关于深度学习中**地貌**问题的不完整的回顾.

[TOC]

## 引言: 深度学习中的地貌问题

Learning = Representation + Evaluation + Optimization.

(机器)学习 = 特征/表示 + (代价)评估 + 优化

来源: Domingos, Pedro. ”A few useful things to know about machine learning.” Communications of the ACM 55.10 (2012): 78-87.

自2010年以来深度学习(深度神经网络)再次迎来发展热潮的原因, 正是在新的硬件(GPU)和网络结构(RNN, CNN, ResNet等等)以及被不断提升的各种优化算法的基础上, 从事深度学习的科学家和工程师们获得了前所未有的成功. 

深度学习中的优化难点: 高维, 非凸(non-convex) + 针对这两点具体解释 难点是什么???

本文中, 我们主要着眼于深度学习问题中的优化部分. 更具体的来说, 对于一个任意的非凸损失函数(loss function), 找到其全局最小值(global minimum)往往是NP-complete的(参考[Some NP-complete problems in quadratic and nonlinear programming](https://link.springer.com/article/10.1007/BF02592948)). 非常不幸的是, 在[Training a 3-node neural network is NP-complete]()已经得到证明, 即使是训练非常简单神经网络事实上也是NP-complete的. 因此, 长期以来, **成功训练**一个神经网络一直被认为是非常困难, 甚至不可能的.

(下面将会提到的一些概念将在后文中正式定义)
然而, 随着我们对非凸问题的理解的不断深入, [When Are Nonconvex Problems Not Scary?]()一文的作者指出: 很多常见的非凸优化问题, 例如 phase retrieval, independent component analysis 以及 orthogonal tensor decomposition 等等, 都具有以下特点:

* 所有的局部最小值都是(等价的)全局最小值 (all local minima are also global)
* 在任何鞍点的"附近", 目标损失函数都具有一个具有**负数曲率**的(下降)方向(around any saddle point the objective function has a negative directional curvature), 因此有沿着这个方向继续下降(使目标损失函数的值继续减小)的可能, 进一步的, 这提供了一种"有效地寻找到**全局最小值**"的可能性.

因而是有希望实现有效的优化的.

因此, 我们非常渴望知道, 深度学习是否也具有和上面相同或者相似的性质? 通过怎样的优化方法我们可以有效的到达(我们"渴望"的)全局最小值.

事实上, 问题远远比这更加复杂. 在上文中, 我们一直在讨论的事实上是, 神经网络和深度学习中的**训练**的问题, 即, 针对于一组给定的*训练数据*, 我们如何通过有效的优化手段(算法)训练我们的神经网络, 使网络的输出能够成功**拟合**这部分训练数据, 通常情况下, 找到对应的目标损失函数的**全局最小值**. 然而, 成功拟合了训练数据并不保证神经网络就获得了优良的性能. 事实上, 在深度学习中, 我们真正关心的是所谓的**泛化性能**(generalization performance), 

to-do list

* 深度神经网络中的欠拟合(under-fitting)和过拟合(over-fitting): over-parametric NNs...
* 我们是否需要获得全局最小值? 是否**局部最小值**或者**鞍点**就可以保证很好的泛化性能[Are Saddles Good Enough for Deep Learning?]
* 如果是的, 具有怎样特征的局部最小值或者鞍点才能够获得良好的泛化性能?

### 深度学习中的地貌定义
### 重要性
### 理论分析难点
### 基本概念

梯度和Jacobian矩阵, Hesian矩阵, 损失地貌的全局最小, 局部最小和鞍点(saddle point)


Consider a smooth function ![](http://latex.codecogs.com/gif.latex?l:\\mathbb{R}^n\\rightarrow\\mathbb{R}). ![](http://latex.codecogs.com/gif.latex?x) is a critical point iff ![](http://latex.codecogs.com/gif.latex?\\nabla{l(x)}=0). The critical points are further classified by considering the Hessian ![](http://latex.codecogs.com/gif.latex?\\nabla^2l(x)=0) of ![](http://latex.codecogs.com/gif.latex?\\f) at ![](http://latex.codecogs.com/gif.latex?x) :
* If all eigenvalues of ![](http://latex.codecogs.com/gif.latex?\\nabla^2l(x)) are positive, critical point ![](http://latex.codecogs.com/gif.latex?x) is a local minimum;
* If all eigenvalues of ![](http://latex.codecogs.com/gif.latex?\\nabla^2l(x)) are negative, critical point ![](http://latex.codecogs.com/gif.latex?x) is a local maximum;
* If  eigenvalues of ![](http://latex.codecogs.com/gif.latex?\\nabla^2l(x)) are both positive and negative, critical point ![](http://latex.codecogs.com/gif.latex?x) is a min-max saddle point;
* If there are zero eigenvalues of ![](http://latex.codecogs.com/gif.latex?\\nabla^2l(x)) at a saddle point ![](http://latex.codecogs.com/gif.latex?x), ![](http://latex.codecogs.com/gif.latex?x) is called a degenerate saddle. 

对于一个光滑的损失函数 ![](http://latex.codecogs.com/gif.latex?l:\\mathbb{R}^n\\rightarrow\\mathbb{R}), ![](http://latex.codecogs.com/gif.latex?x) 是该损失函数的驻点(stationary point)当且仅当 ![](http://latex.codecogs.com/gif.latex?\\nabla{l(x)}=0). 驻点的类别由 Hessian矩阵 ![](http://latex.codecogs.com/gif.latex?\\nabla^2l(x)) 来判断：
* 如果 ![](http://latex.codecogs.com/gif.latex?\\nabla^2l(x) )的所有特征值为正, 驻点 ![](http://latex.codecogs.com/gif.latex?x) 是局部最小点;
* 如果 ![](http://latex.codecogs.com/gif.latex?\\nabla^2l(x))的所有特征值为负, 驻点 ![](http://latex.codecogs.com/gif.latex?x) 是局部最大点；
* 如果 ![](http://latex.codecogs.com/gif.latex?\\nabla^2l(x)) 特征值有正有负, 驻点 ![](http://latex.codecogs.com/gif.latex?x) 是鞍点;
* 如果对于一个鞍点 ![](http://latex.codecogs.com/gif.latex?x)， ![](http://latex.codecogs.com/gif.latex?\\nabla^2l(x)) 有零特征值， 则称 ![](http://latex.codecogs.com/gif.latex?x) 为退化的鞍点 (degenerate saddle point).

为了方便理解上面介绍的几个概念, 我们借助下面的图以及一些公式, 来解释为什么在深度学习的优化过程中会出现上面的几种情况.

首先需要注意的是, 当使用基于梯度的优化方法的时候

<img src='https://d.pr/i/kEUPNf+'>

<img src='https://d.pr/i/NKLDNT+'>

(图片来源: Ian Goodfellow [Deep Learning](http://www.deeplearningbook.org/) Section 4)

### 深度学习中的优化: (随机)梯度下降...

## 对于深度学习中的地貌的描述的进展
* linear network
* deep learning without poor local minima

* single layer network
* multi-layer network

In [Neural networks and principal component analysis](https://github.com/Zhenyu-LIAO/Deep-Learning-Landscape/blob/master/references/Neural%20networks%20and%20principal%20component%20analysis-%20Learning%20from%20examples%20without%20local%20minima.pdf), the author studies the autoencoder with one hidden layer and showed the equivalence between the local minimum and the global minimum with a characterization of the form of global minimum points. 

早在1989年， 作者在 [Neural networks and principal component analysis](https://github.com/Zhenyu-LIAO/Deep-Learning-Landscape/blob/master/references/Neural%20networks%20and%20principal%20component%20analysis-%20Learning%20from%20examples%20without%20local%20minima.pdf) 一文中中研究了单隐层的线性自编码器中局部最优点和全局最优点的等价性。

In [Deep Learning without Poor Local Minima](https://github.com/Zhenyu-LIAO/Deep-Learning-Landscape/blob/master/references/Deep%20Learning%20without%20Poor%20Local%20Minima.pdf), the author proves the existence of degenerate saddle points for deep linear neural networks with squared loss function and the fact that any local minimum is also a global minimum, with slightly weaker assumptions. The authors  simplify the proof proposed in [Depth Creates No Bad Local Minima](https://github.com/Zhenyu-LIAO/Deep-Learning-Landscape/blob/master/references/Depth%20Creates%20No%20Bad%20Local%20Minima.pdf) and generalize the previous results with fewer assumptions.

NIPS 2016的一篇文章 [Deep Learning without Poor Local Minima](https://github.com/Zhenyu-LIAO/Deep-Learning-Landscape/blob/master/references/Deep%20Learning%20without%20Poor%20Local%20Minima.pdf) 证明了在满秩条件下，多层线性网络的所有局部极小是全局最小。对于多层非线性网络，作者在一个比较强的假设下：网络隐层中各结点的输出相互独立（假设了free？），得到了和多层线性网络一样的结论。[Depth Creates No Bad Local Minima](https://github.com/Zhenyu-LIAO/Deep-Learning-Landscape/blob/master/references/Depth%20Creates%20No%20Bad%20Local%20Minima.pdf) 中作者做了更弱的假设并且简化了证明。

## Hessian特性的问题 + 一些 empirical studies
The Gauss-Newton decomposition of the Hessian matrix

Let  ![](http://latex.codecogs.com/gif.latex?l) donate the loss function and   ![](http://latex.codecogs.com/gif.latex?f) is the real-valued output of a network. Then the Hessian of the loss for a given example are given by  ![](http://latex.codecogs.com/gif.latex?\\nabla^2{l(w)}=l^{''}(f(w))\\nabla{f(w)}\\nabla{f(w)}^T+l^{'}(f(w))\\nabla^2{f(w)}). 

Hessian矩阵的高斯牛顿分解:
令 ![](http://latex.codecogs.com/gif.latex?l) 表示损失函数，![](http://latex.codecogs.com/gif.latex?f) 表示网络的输出。那么，损失函数的Hessian矩阵可分解为![](http://latex.codecogs.com/gif.latex?\\nabla^2{l(w)}=l^{''}(f(w))\\nabla{f(w)}\\nabla{f(w)}^T+l^{'}(f(w))\\nabla^2{f(w)})。

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

In [^3][^4], the authors argue that the flatness of minima of the loss function found by stochastic gradient-based methods will result  in good generalization performance. However, in [^5], the authors prove that sharp minima can generalize well when following several definitions of flatness.

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

