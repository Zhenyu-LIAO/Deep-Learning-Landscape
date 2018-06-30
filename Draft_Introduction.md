
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

我们首先介绍一些和地貌问题有关的基本概念：梯度, Hesian矩阵, 损失地貌的全局最小, 局部最小和鞍点(saddle point)

对于一个光滑的损失函数 ![](http://latex.codecogs.com/gif.latex?l:\\mathbb{R}^n\\rightarrow\\mathbb{R}), ![](http://latex.codecogs.com/gif.latex?x) 是该损失函数的**驻点(stationary point)**当且仅当对应梯度 ![](http://latex.codecogs.com/gif.latex?\\nabla{l(x)}=0). 驻点的类别由 **Hessian矩阵** ![](http://latex.codecogs.com/gif.latex?\\nabla^2l(x)) 来判断：
* 如果 ![](http://latex.codecogs.com/gif.latex?\\nabla^2l(x) )的所有特征值为正, 驻点 ![](http://latex.codecogs.com/gif.latex?x) 是**局部最小点**;
* 如果 ![](http://latex.codecogs.com/gif.latex?\\nabla^2l(x))的所有特征值为负, 驻点 ![](http://latex.codecogs.com/gif.latex?x) 是**局部最大点**；
* 如果 ![](http://latex.codecogs.com/gif.latex?\\nabla^2l(x)) 特征值有正有负, 驻点 ![](http://latex.codecogs.com/gif.latex?x) 是**鞍点**;
* 如果对于一个鞍点 ![](http://latex.codecogs.com/gif.latex?x)， ![](http://latex.codecogs.com/gif.latex?\\nabla^2l(x)) 有零特征值， 则称 ![](http://latex.codecogs.com/gif.latex?x) 为**退化的鞍点** (degenerate saddle point).

在了解了上面定义的基础上, 我们正式给出一直谈到的**地貌**问题的定义:

>考虑权重矩阵 ![](http://latex.codecogs.com/gif.latex?W=(W_1,...,W_H)) 所构成的空间 ![](http://latex.codecogs.com/gif.latex?\\Xi), 空间中的每一点和一个(不同的)神经网络模型一一对应, 并且对应一个唯一的损失函数 ![](http://latex.codecogs.com/gif.latex?l(W)) 的值 (反之不然, 同一个损失函数值可能对应多个不同的![](http://latex.codecogs.com/gif.latex?W)). 深度学习地貌问题讨论的就是该空间中点和对应损失函数之间的关系以及不同的算法对于该空间的搜索能力(即, 不同的算法在空间中的运动轨迹).

另外, 对于光滑的损失函数 ![](http://latex.codecogs.com/gif.latex?l), Hessian矩阵是对称阵, 可以写成谱分解写成![](http://latex.codecogs.com/gif.latex?U\\Sigma{U^T}=\\sum{u_iu_i^T}\\sigma_i) 其中![](http://latex.codecogs.com/gif.latex?(\\sigma_i,u_i)) 为对应的特征值和特征向量.

显然, 在地貌问题中, Hessian矩阵扮演着一个非常重要的角色. 根据上面的定义, 其特征值直接决定了空间中对应点附近邻域中的损失函数 ![](http://latex.codecogs.com/gif.latex?l) 的值, 即:

* 如果该点(记为 ![](http://latex.codecogs.com/gif.latex?W))为局部最小值, 则附近邻域的损失函数值都**严格大于**其对应的损失函数 ![](http://latex.codecogs.com/gif.latex?l(W)). 对应的结果就是, 当算法在 ![](http://latex.codecogs.com/gif.latex?W) 点并尝试进行下一步移动是, 附近空间的每个方向都一定导致损失函数增大.
* 如该点为局部最大值, 则附近邻域的损失函数值都**严格小于**其对应的损失函数 ![](http://latex.codecogs.com/gif.latex?l(W)), 因此, 当算法在 ![](http://latex.codecogs.com/gif.latex?W) 点并尝试进行下一步移动是, 附近空间的每个方向都一定导致损失函数减小.
* 如果该点为鞍点, 则附近邻域的损失函数值都可能大于, 小于或者等于其对应的损失函数 ![](http://latex.codecogs.com/gif.latex?l(W)), 因此, 当算法在 ![](http://latex.codecogs.com/gif.latex?W) 点并尝试进行下一步移动是, 附近空间的存在一些方向可以使损失函数增大, 另一些方向使其减小, 还有一些方向则不会使之发生变化.

当![](http://latex.codecogs.com/gif.latex?W) 为一维时, 图示如下:

<img src='https://d.pr/i/kEUPNf+'>
(图片来源: Ian Goodfellow [Deep Learning](http://www.deeplearningbook.org/) Section 4)

可以为什么Hessian矩阵能够告诉我们这些信息呢? 下面, 我们再来审视一下Hessian矩阵的特征值到底意味着什么:

考虑一个驻点 ![](http://latex.codecogs.com/gif.latex?W^*),对于该点附近邻域的另一点![](http://latex.codecogs.com/gif.latex?W), 我们对损失函数![](http://latex.codecogs.com/gif.latex?l(W)) 进行泰勒展开可以得到: ![](http://latex.codecogs.com/gif.latex?l(W)\\approx{l(W^*)}+(W-W^*)^T\\nabla{l(W^{*})}+\\frac{1}{2}(W-W^*)^TH(W^*)(W-W^*)). 其中 ![](http://latex.codecogs.com/gif.latex?\\nabla{l(W^*)}) 为 ![](http://latex.codecogs.com/gif.latex?W^*) 对应的梯度, ![](http://latex.codecogs.com/gif.latex?H(W^*)) 为对应Hessian矩阵, ![](http://latex.codecogs.com/gif.latex?W-W^*) 就是我们从![](http://latex.codecogs.com/gif.latex?W^*) 移动到 ![](http://latex.codecogs.com/gif.latex?W) 的这一步(包含大小和方向).

根据定义我们有 ![](http://latex.codecogs.com/gif.latex?\\nablal(W^*)=0), 因此, ![](http://latex.codecogs.com/gif.latex?W) 和 ![](http://latex.codecogs.com/gif.latex?W^*) 对应损失函数的差别即为 ![](http://latex.codecogs.com/gif.latex?l(W)-l(W^*)\\approx\\frac{1}{2}(W-W^*)^TH(W^*)(W-W^*)). 通过对应Hessian矩阵的谱分解, 我们得到 ![](http://latex.codecogs.com/gif.latex?l(W)-l(W^*)\\approx\\frac{1}{2}(W-W^*)^TU_H\\Sigma_HU_H^T(W-W^*)=\\sum\\sigma_i\\frac{1}{2}(W-W^*)^Tu_iu_i^T(W-W^*)).

因此, 如果假设在驻点![](http://latex.codecogs.com/gif.latex?W^*) 我们有一个非常美好的线性空间(事实情况可能比这复杂的多), 我们可以研究这个线性空间的基(basis)和维度(dimension). 那么, 如果我们选择沿着特征值![](http://latex.codecogs.com/gif.latex?\\sigma_i>0) 对应的特征向量![](http://latex.codecogs.com/gif.latex?u_i) 的方向移动的话 (其他方向分量为0), 可以得到 ![](http://latex.codecogs.com/gif.latex?l(W)-l(W^*)>0). 也就是说, 当我们考虑驻点![](http://latex.codecogs.com/gif.latex?W^*) 附近空间的时候, 正的特征值对以特征向量的方向就是上升的方向. 反正, 负特征值对应的方向就是下降的方向, 而零特征值对应的方向则无法判断, 往往需要更加高阶的信息(更进一步的泰勒展开). 

**备注1**: 那么在深度学习的问题中, 人们一直在讨论训练深度神经网络中**最大的障碍**可能是什么: 在认为局部最大点不存在的情况下(在一些特定的网络结构和损失函数的情形下, 这一点已经被证明了), 需要关注的就是局部最小点和鞍点. 值得注意的是, **局部**最小值只描述了对应驻点附近邻域的**局部**情况: 我们并不了解更远的地方发生了什么, 是否有更小的值. 在所有局部最小值中, 我们将其中对应损失函数值最小的称之为**全局最小值**, 这也是我们能到达到的损失函数的**最小值**, 也是对应地貌的**最低点**. 下图形象的展示了这个问题: 
<img src='https://d.pr/i/NKLDNT+'>
(图片来源: Ian Goodfellow [Deep Learning](http://www.deeplearningbook.org/) Section 4)

**备注2**: 一个非常有意思的问题是, 如果算法并严格不沿着对应特征向量的方向移动的时候, 会发生什么呢? 换言之, 如果移动方向在不同符号特征值对应的特征值向量上都有不为零的分量, 那么会发生什么的? 算法会下降还是上升呢?  为了讨论这个问题, 我们考虑对应问题的**连续**版本: 即, 算法可以在空间![](http://latex.codecogs.com/gif.latex?\\Xi) 中连续的移动, 自变量变为时间![](http://latex.codecogs.com/gif.latex?t). 也就是说, 随着时间变化, 算法在空间中画出一条连续的轨迹. 我们知道这样的连续系统中会出现**指数收敛**的情况, 举一个一维情况下的例子, 考虑一个由如下常微分方程(ODE)表述的动力系统: ![](http://latex.codecogs.com/gif.latex?\\frac{dx}{dt}=\\alpha{x}), ![](http://latex.codecogs.com/gif.latex?x=0) 是该系统的一个平衡点(或者驻点, 由于对应梯度为零), 而这样的方程的解为 ![](http://latex.codecogs.com/gif.latex?x=x_0\\exp(\\alpha{t})), 其中![](http://latex.codecogs.com/gif.latex?x_0) 为其初值(初始位置). 因此, 当![](http://latex.codecogs.com/gif.latex?\\alpha>0) 时, ![](http://latex.codecogs.com/gif.latex?x) 将会随时间指数增大, 从而远离对应的驻点(**发散**). 相反的, 如果![](http://latex.codecogs.com/gif.latex?\\alpha<0), ![](http://latex.codecogs.com/gif.latex?x) 将会随时间指数减小至零, 从而收敛到对应的驻点(**收敛**).  事实上, 在上述的梯度系统中, 也有类似的情况发生. 当我们考虑一个对应驻点附近的线性空间的时候, 我们总可以通过线性变化将算法移动的方向投影在Hessian特征向量所确定的坐标系中. 那么, 对应正特征值方向上的投影的部分就是指数发散, 从而导致我们离开对应的驻点![](http://latex.codecogs.com/gif.latex?W^*). 因此, 我们可以总结, 一旦我们移动的方向在对应正特征值方向上投影不为零, 算法的运动轨迹就会飞速的远离对应驻点. 当且仅当移动方向垂直于**所有**正特征值对应特征向量确定的空间的之后, 我们才会靠近对应驻点. 如下图所示:
<img src='https://d.pr/i/9sG7VF+' style="width:200px;height:200px;">
基于上面的描述, 我们就有了一种描述算法在一个对应驻点附近运动情况的方法: 如果依旧在线性空间的基础上考虑, 我们可以将该驻点邻域空间划分成三个子空间: 稳定(stable), 对应Hessian特征值为负; 不稳定(unstable), 对应特征值为正数和对应特征值为零的(center). 根据对应线性子空间的维度来判断算法被该驻点**吸引**的可能性大小(通常称之为分析basin of attraction). 事实上, [Morse theory](https://en.wikipedia.org/wiki/Morse_theory)给出了非退化驻点(non-degenerate critical point, 对应Hessain矩阵不包含零特征值)的一些很有趣/很强的结果, 然而对应退化的驻点, 更近一步说, 退化的鞍点则需要进一步更加深入的研究.

### 深度学习的地貌: 为什么?

深度学习地貌的科研意义是毋庸置疑的, 那么其如此火热的今天, 为什么学界迟迟还没有拿出一套令人信服的研究方案, 换言之, 这个问题本身难在哪里?

#### 难点1: 高维(high-dimensional)

深度学习问题的**高维**特点体现在很多方面, 比如:

* 高维数据: 在机器视觉, 自然语言处理, 语音识别等等深度学习获得重大突破的领域, 数据无一例外的有**海量**和**高维**两个特点. 海量意味着用以训练的数据数目非常多, 而高维则代表每个数据"很大": 例如, 知名的计算机视觉数据集 [MNIST](http://yann.lecun.com/exdb/mnist/) 和 [ImageNet](http://www.image-net.org/) 中图片数据的维度就分别是28* 28 = 784和256*256= 65536(常用彩色).  事实上, 机器学习中高维问题已经是一个老生常谈的话题了: 在传统统计学习(statistical learning)的框架下, 我们常常称之为**维度灾难(curse of dimensionality)**. 常见的两种理解如下:
    * 需要指数增长的训练数据数目: 在机器学习经典教材 Pattern Recognition and Machine Learning(PRML by Bishop) 一书中就提到的是这个角度
    * 高维空间中反直觉(counterintuitive): 例如不同数据点之间的欧氏距离几乎约等于一个常数, 无论远近(在分类问题中: 同类还是不同类), 这就导致了, 很多基于低维空间(3D)直觉的算法在高维数据上行不通(例如最邻近搜索:nearest neighbor search. 数据之间的距离都差不多, 而不是距离近的同一类, 远的是不同类). 亦或是高维空间的体积分布和我们熟悉的三维非常不同.

* 高维(模型)参数: 导致的计算(搜索)量随维度的指数上升, 需要大量的计算资源.

**备注3**: 由于深度神经网络的参数远远大于数据的维度和数目, 依据统计学习的理论, 这样的模型应该是没办法进行有效的训练的. 即由于模型复杂度太高, 一定会发生过拟合(overfitting)的现象, 从而导致模型的泛化性能不好. 然而, 现实生活中获得的深度神经网络模型, 似乎并没有遇到过拟合的问题. 这也是深度学习理论的一个非常重要的研究方向.



#### 难点2:非凸(non-convex)


具体来说, 对于一个任意的非凸损失函数(loss function), 找到其全局最小值(global minimum)往往是NP-complete的(参考[Some NP-complete problems in quadratic and nonlinear programming](https://link.springer.com/article/10.1007/BF02592948)). 非常不幸的是, 在[Training a 3-node neural network is NP-complete](https://github.com/Zhenyu-LIAO/Deep-Learning-Landscape/blob/master/references/Training%20a%203-node%20neural%20network%20is%20NP-complete.pdf)已经得到证明, 即使是训练非常简单神经网络事实上也是NP-complete的. 因此, 长期以来, **成功训练**一个神经网络一直被认为是非常困难, 甚至不可能的.

然而, 随着我们对非凸问题的理解的不断深入, [When Are Nonconvex Problems Not Scary?](https://arxiv.org/abs/1510.06096)一文的作者指出: 很多常见的非凸优化问题, 例如 phase retrieval, independent component analysis 以及 orthogonal tensor decomposition等等, 都具有以下特点:

* 所有的局部最小值都是(等价的)全局最小值 (all local minima are also global)
* 在任何鞍点的"附近", 目标损失函数都具有一个具有**负数曲率**的(下降)方向(a negative directional curvature), 因此有沿着这个方向继续下降(使目标损失函数的值继续减小)的可能, 进一步的, 这提供了一种"有效地寻找到**全局最小值**"的可能性.

因而是有希望实现有效的优化的.

因此, 我们非常渴望知道, 深度学习是否也具有和上面相同或者相似的性质? 通过怎样的优化方法我们可以有效的到达(我们"渴望"的)全局最小值.

#### 难点3: 泛化性能

事实上, 问题远远比这更加复杂. 在上文中, 我们一直在讨论的是深度神经网络中的**训练**的问题, 即, 针对于一组给定的*训练数据*, 我们如何通过有效的优化手段(算法)训练我们的神经网络, 使网络的输出能够成功**拟合**这部分训练数据, 通常情况下, 找到对应的目标损失函数的**全局最小值**. 然而, 在机器学习或者深度学习中, 真正的核心问题的是**泛化性能**(generalization performance)，成功拟合了训练数据并不保证泛化性能. 事实上, 往往训练集找到的全局最优点在测试集的表现很差。 更令人困惑的是，大量实验表明，局部最优点甚至鞍点貌似具有相似的泛化性能。遗憾的是, 由于神经网络其复杂的结构, 基于传统统计学习方法的, 对于泛化性能的估计往往比较悲观, 事实上, 一批又一批的科研工作者投身理解深度模型泛化的理论工作中.

* 我们是否需要获得全局最小值? 是否**局部最小值**或者**鞍点**就可以保证很好的泛化性能. 例如: [Are Saddles Good Enough for Deep Learning?](https://arxiv.org/pdf/1706.02052.pdf)
* 如果是的, 具有怎样特征(比如, 对应Hessian或者Jacobian矩阵)的局部最小值或者鞍点才能够获得良好的泛化性能? 




