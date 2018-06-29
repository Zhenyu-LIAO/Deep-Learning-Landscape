

关于深度学习中**地貌**问题的不完整的回顾.


## 引言: 深度学习中的地貌问题

在新的硬件(GPU)和网络结构(RNN, CNN, ResNet等等)以及不断提升的各种优化算法的基础上, 从事深度学习的科学家和工程师们获得了前所未有的成功, 从而(再次)迎来了自2010年以来深度学习(深度神经网络)的一波新的发展热潮: 从计算机视觉到语音识别, 自然语言处理, 一项一项的惊人成就被不断的突破, 全新的变革好像无时无刻不在发生. 这一波深度学习的浪潮仿佛在这短短十年之间就席卷了千家万户, 给人们的工作和生活带来了无法忽视的重大影响. 这些影响不仅仅局限在科学研究的层面, 借助科技公司的强大推力, 每个人的生活其实早已发生了变化, 人们甚至已经习惯了相对"机器"来说, 人在某些领域望尘莫及的这一事实: 越来越多的棋手选择了计算机作为自己的陪练, 越来越多的人们已经对精确的搜索结果和私人化的广告推送习以为常, 智能助手已经以手机或者音响的形式走进了千家万户. 这一切看上去都是一片繁荣景象.

然而, 尽管深度学习在种种应用场景中无一例外取得了惊艳的表现, 我们对于它本身的"科学"的认知, 还仿佛蹒跚学步的孩子, 被这趟风驰电骋的技术的火车远远的甩在了身后. 当大众都沉醉在深度学习的勾勒出的美妙图景的时候, 很多人工智能的科研人员已经忧心忡忡的提出了下面这些疑惑:
> 深度神经网络是不是新的"炼金术"?
> 深度学习真的"学习"到了么?
> 深度学习这个黑盒子里面装的到底是什么?
> 深度神经网络的性能可以得到理论保证吗?

自2017年LeCun和Ali Rahimi针对"深度学习是不是炼金术"这一问题引发的辩论以来, 深度学习, 亦或者深度神经网络的理论工作已经吸引了越来越多的学者, 其中不乏各种大神级别的物理学家和数学家, 从各自的专业领域, 给大家带来对于深度学习的新的认识和理解. 在本系列中, 我们将尝试梳理这些和深度学习有关的理论结果, 具体来说, 我们将围绕"深度学习中的地貌(landscape)问题"来展开讨论.

### 深度学习的地貌: 是什么?

深度学习中的地貌问题, 事实上包含了几个相互关联但是又截然不同的研究方向. 事实上, 我们可以将深度学习看做一种广义的"曲线拟合"的问题: 我们希望, 能够找到一个好的模型, 尽可能准确的描绘给定的数据和对应目标的关系: 图像数据和"猫"或者"狗"的概念之间的关系, 股票数据和其涨跌之间的关系, 不同语言之间同义词或者近义词之间的联系. 而深度学习就给我们指明了这样一条做好"曲线拟合"的道路: 通过构建足够**深**的神经网络模型, 借助**海量**的数据和**基于梯度**(gradient-based)的优化方法, 我们往往能够获得一个很好的"曲线拟合"模型. 

再次审视深度学习的这几个关键点的时候, 下面几个问题就会自然而然的出现:

* 表达：深度神经网络是否真的能够拟合我们需要拟合的数据? 如果是, 我们需要多么**深**的网络才能做到这一点?
* 优化：目前被广泛使用的基于梯度的优化方法是有效的么? 它是不是总能帮我们找到我们渴望的"好的"拟合模型?
* 泛化：单纯的"数据拟合"真的足够了么? 事实上, 我们期望的是, 通过有限的训练数据, 获得一套所谓的"普适规则". 真正的目的是这套获得的规则也能够应用于更多更广的数据(即, 不仅仅局限于用来训练的那些数据, 因而并不是对训练数据的单纯**记忆(memorize)**). 我们将模型的这种"能够推广到更广泛数据"的能力称之为**泛化性能(generalization performance)**. 因此, 我们的问题事实上是: 神经网络的泛化能力到底如何呢?

这些问题, 都可以广义的被总结为深度学习中的地貌问题. 具体来说, 深度学习中的地貌问题研究的是, 深度神经网络**模型**以及对应的**优化方法**和其**性能**(在训练集上的表现, 或者其泛化性能)之间的关系. 从某种意义上讲, 它几乎涵盖了关于神经网络的绝大多数理论研究方向.
### 基本概念

我们首先介绍一些和地貌问题有关的基本概念：梯度和Jacobian矩阵, Hesian矩阵, 损失地貌的全局最小, 局部最小和鞍点(saddle point)

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

备注: 那么在深度学习的问题中, 人们一直在讨论深度神经网络中可能需要的问题是什么: 在认为局部最大点不存在的情况下(在一些特定的网络结构和损失函数的情形下, 这一点已经被证明了), 需要关注的就是局部最小点和鞍点. 

为了方便理解上面介绍的几个概念, 我们借助下面的图以及一些公式, 来解释为什么在深度学习的优化过程中会出现上面的几种情况.

首先需要注意的是, 当使用基于梯度的优化方法的时候

<img src='https://d.pr/i/kEUPNf+'>

<img src='https://d.pr/i/NKLDNT+'>

(图片来源: Ian Goodfellow [Deep Learning](http://www.deeplearningbook.org/) Section 4)
### 深度学习的地貌: 为什么?

深度学习地貌的科研意义是毋庸置疑的, 那么其如此火热的今天, 为什么学界迟迟还没有拿出一套令人信服的研究方案, 换言之, 这个问题本身难在哪里?

#### 难点1: 高维(high-dimensional)

深度学习问题的**高维**特点体现在很多方面, 比如:

* 高维数据: 在机器视觉, 自然语言处理, 语音识别等等深度学习获得重大突破的领域, 数据无一例外的有**海量**和**高维**两个特点. 海量意味着用以训练的数据数目非常多, 而高维则代表每个数据"很大": 例如, 知名的计算机视觉数据集 [MNIST](http://yann.lecun.com/exdb/mnist/) 和 [ImageNet](http://www.image-net.org/) 中图片数据的维度就分别是$28\times 28 = 784$和$256\times 256= 65536$(常用彩色). 事实上, 实验已经证明[ref?], 对于少量或者低维的数据, 深度神经网络相对于传统机器学习方法来说优势并不明显, 甚至在有些时候远远不如. 为什么高维问题会非常困难呢? 事实上, 机器学习中高维问题已经是一个老生常谈的话题了: 在传统统计学习(statistical learning)的框架下, 我们常常称之为**维度灾难(curse of dimensionality)**. 常见的两种理解如下:
    * 需要指数增长的训练数据数目: 在机器学习经典教材 Pattern Recognition and Machine Learning(PRML by Bishop) 一书中就提到的是这个角度
    * 高维空间中反直觉(counterintuitive): 例如不同数据点之间的欧氏距离几乎约等于一个常数, 无论远近(在分类问题中: 同类还是不同类), 这就导致了, 很多基于低维空间(3D)直觉的算法在高维数据上行不通(例如最邻近搜索:nearest neighbor search. 数据之间的距离都差不多, 而不是距离近的同一类, 远的是不同类). 亦或是高维空间的体积分布和我们熟悉的三维非常不同.

    To-do list:
    
    * [Cover's theorem](https://en.wikipedia.org/wiki/Cover%27s_theorem) 和(大概率)线性可分, 与核方法(kernel method)的联系
    * 流形学习(manifold learning) 信息的维度, 噪声的维度和数据的维度
    * https://zhuanlan.zhihu.com/p/27488363


* 高维(模型)参数: 导致的计算(搜索)量随维度的指数上升
然而, 需要格外注意的是, 由于深度神经网络的参数远远大于数据的维度和数目, 依据统计学习的理论, 这样的模型应该是没办法进行有效的训练的. 即由于模型复杂度太高, 一定会发生过拟合(overfitting)的现象, 从而导致模型的泛化性能不好. 然而, 现实生活中, 



#### 难点2:非凸(non-convex)


具体来说, 对于一个任意的非凸损失函数(loss function), 找到其全局最小值(global minimum)往往是NP-complete的(参考[Some NP-complete problems in quadratic and nonlinear programming](https://link.springer.com/article/10.1007/BF02592948)). 非常不幸的是, 在[Training a 3-node neural network is NP-complete](https://github.com/Zhenyu-LIAO/Deep-Learning-Landscape/blob/master/references/Training%20a%203-node%20neural%20network%20is%20NP-complete.pdf)已经得到证明, 即使是训练非常简单神经网络事实上也是NP-complete的. 因此, 长期以来, **成功训练**一个神经网络一直被认为是非常困难, 甚至不可能的.

(下面将会提到的一些概念将在后文中给出正式的定义)

然而, 随着我们对非凸问题的理解的不断深入, [When Are Nonconvex Problems Not Scary?](https://arxiv.org/abs/1510.06096)一文的作者指出: 很多常见的非凸优化问题, 例如 phase retrieval, independent component analysis 以及 orthogonal tensor decomposition等等, 都具有以下特点:

* 所有的局部最小值都是(等价的)全局最小值 (all local minima are also global)
* 在任何鞍点的"附近", 目标损失函数都具有一个具有**负数曲率**的(下降)方向(a negative directional curvature), 因此有沿着这个方向继续下降(使目标损失函数的值继续减小)的可能, 进一步的, 这提供了一种"有效地寻找到**全局最小值**"的可能性.

因而是有希望实现有效的优化的.

因此, 我们非常渴望知道, 深度学习是否也具有和上面相同或者相似的性质? 通过怎样的优化方法我们可以有效的到达(我们"渴望"的)全局最小值.

#### 难点3: 泛化性能

事实上, 问题远远比这更加复杂. 在上文中, 我们一直在讨论的事实上是, 神经网络和深度学习中的**训练**的问题, 即, 针对于一组给定的*训练数据*, 我们如何通过有效的优化手段(算法)训练我们的神经网络, 使网络的输出能够成功**拟合**这部分训练数据, 通常情况下, 找到对应的目标损失函数的**全局最小值**. 然而, 成功拟合了训练数据并不保证神经网络就获得了优良的性能. 事实上, 在深度学习中, 我们真正关心的是所谓的**泛化性能**(generalization performance). 然而, 由于神经网络其复杂的结构, 基于传统统计学习方法的, 对于泛化性能的估计往往比较悲观, 事实上, 对于深度模型泛化性能的度量标准或者相关表征的探索, 在持续地吸引着一批又一批的科研工作者投身其中.

* 深度神经网络中的欠拟合(under-fitting)和过拟合(over-fitting): over-parametric NNs...
* 我们是否需要获得全局最小值? 是否**局部最小值**或者**鞍点**就可以保证很好的泛化性能[Are Saddles Good Enough for Deep Learning?](https://arxiv.org/pdf/1706.02052.pdf)
* 如果是的, 具有怎样特征的局部最小值或者鞍点才能够获得良好的泛化性能?






### 几种尝试的方向

* 地貌的渐进分析: spin-glass模型, 
* 空间角度的分析: (分层的)莫尔斯理论(Stratified Morse Theory): 非常方便的分析一个流形(manifold)的拓扑的方法.

### 深度学习中的优化: (随机)梯度下降...



