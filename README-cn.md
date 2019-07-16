# numpy-ml
Ever wish you had an inefficient but somewhat legible collection of machine
learning algorithms implemented exclusively in numpy? No?
祝你有一个用numpy实现的低效的但易懂机器学习算法。

## Models 模型
This repo includes code for the following models:
这个repo包含以下实现：

1. **Gaussian mixture model** （高斯混合模型）
    - EM training （EM训练）

2. **Hidden Markov model** （隐马尔可夫模型）
    - Viterbi decoding（维特比解码）
    - Likelihood computation（似然计算）
    - MLE parameter estimation via Baum-Welch/forward-backward algorithm（通过Baum-Welch/forward-backward算法进行MLE参数估计）

3. **Latent Dirichlet allocation** (topic model) 隐狄利克雷分配模型（主题模型）
    - Standard model with MLE parameter estimation via variational EM（通过变分EM进行EM参数估计的标准模型）
    - Smoothed model with MAP parameter estimation via MCMC （通过MCMC进行MAP参数估计的平滑模型）

4. **Neural networks** 神经网络
    * Layers / Layer-wise ops 层与层级运算
        - Add 添加
        - Flatten 扁平化
        - Multiply 乘积
        - Softmax
        - Fully-connected/Dense 全连接/Dense
        - Sparse evolutionary connections 稀疏化连接
        - LSTM 长短期记忆网络
        - Elman-style RNN Elman风格的RNN
        - Max + average pooling 最大+平均池化
        - Dot-product attention 点积注意力
        - Restricted Boltzmann machine (w. CD-n training) 受限玻尔兹曼机（w. CD-n 训练）
        - 2D deconvolution (w. padding and stride) 二维转置卷积（w. 填充与步幅）
        - 2D convolution (w. padding, dilation, and stride) 二维卷积（w. 填充，扩张和步幅）
        - 1D convolution (w. padding, dilation, stride, and causality) 一维卷积（w. 填充，扩张，步幅和因果关系）
    * Modules 模型
        - Bidirectional LSTM 双向LSTM
        - ResNet-style residual blocks (identity and convolution) ResNet风格的残差块（恒等变换和卷积）
        - WaveNet-style residual blocks with dilated causal convolutions WaveNet 风格的残差块（带有扩张因果卷积）
        - Transformer-style multi-headed scaled dot product attention Transformer 风格的多头缩放点积注意力
    * Regularizers 正则化项
        - Dropout
    * Normalization 归一化 正态化
        - Batch normalization (spatial and temporal) 批归一化（时间上和空间上）
        - Layer normalization (spatial and temporal) 层归一化（时间上和空间上）
    * Optimizers 优化器
        - SGD w/ momentum SGD w/ 动量
        - AdaGrad 
        - RMSProp 
        - Adam
    * Learning Rate Schedulers 学习率调度器
        - Constant 常数
        - Exponential 指数
        - Noam/Transformer 诺姆变换
        - Dlib scheduler Dlib调度器
    * Weight Initializers
        - Glorot/Xavier uniform and normal Glorot/Xavier统一化与正态化
        - He/Kaiming uniform and normal He/Kaiming统一化与正态化
        - Standard and truncated normal 标准和截断正态分布初始化
    * Losses 损失
        - Cross entropy 交叉熵
        - Squared error 平方差
        - Bernoulli VAE loss 伯努利VAE损失
        - Wasserstein loss with gradient penalty 带有梯度惩罚的沃瑟斯坦损失
    * Activations 激活函数
        - ReLU 整流函数
        - Tanh 双曲函数
        - Affine 仿射函数
        - Sigmoid S型函数
        - Leaky ReLU 带泄露整流函数
        - ELU 指数线性单元
        - SELU 扩展指数线性单元
        - Exponential 指数
        - Hard Sigmoid 分段线性近似Sigmoid
        - Softplus 
    * Models 模型
        - Bernoulli variational autoencoder 伯努利变分自编码器
        - Wasserstein GAN with gradient penalty 带梯度惩罚的沃瑟斯坦GAN
    * Utilities 神经网络工具
        - `col2im` (MATLAB port)
        - `im2col` (MATLAB port)
        - `conv1D`
        - `conv2D`
        - `deconv2D`
        - `minibatch`

5. **Tree-based models** 基于树的模型
    - Decision trees (CART) 决策树（CART）
    - [Bagging] Random forests 随机森林
    - [Boosting] Gradient-boosted decision trees 梯度提升决策树

6. **Linear models** 线性模型
    - Ridge regression 岭回归
    - Logistic regression 逻辑回归
    - Ordinary least squares 最小二乘法
    - Bayesian linear regression w/ conjugate priors 贝叶斯回归 w/共轭先验
        - Unknown mean, known variance (Gaussian prior) 未知平均值，已知方差（高斯先验）
        - Unknown mean, unknown variance (Normal-Gamma / Normal-Inverse-Wishart prior) 未知平均值，未知方差（正态伽玛 / 正态逆威沙特先验）

7. **n-Gram sequence models** N元序列模型
    - Maximum likelihood scores 最大似然值
    - Additive/Lidstone smoothing Additive/Lidstone平滑
    - Simple Good-Turing smoothing 简单Good-Turing平滑

8. **Reinforcement learning models** 强化学习模型
    - Cross-entropy method agent 交叉熵方法智能体
    - First visit on-policy Monte Carlo agent 首次访问on-policy蒙特卡洛智能体
    - Weighted incremental importance sampling Monte Carlo agent 加权增量重要采样蒙特卡洛智能体
    - Expected SARSA agent （Expected SARSA 智能体）
    - TD-0 Q-learning agent （TD-0 Q-learning 智能体）
    - Dyna-Q / Dyna-Q+ with prioritized sweeping （Dyna-Q / Dyna-Q+ 优先扫描）

9. **Nonparameteric models** 非参数模型
    - Nadaraya-Watson kernel regression Nadaraya-沃森核回归
    - k-Nearest neighbors classification and regression k近邻分类与回归
    - Gaussian process regression 高斯过程回归

10. **Preprocessing** 预处理
    - Discrete Fourier transform (1D signals) 离散傅里叶变换（一维信号）
    - Discrete cosine transform (type-II) (1D signals) 离散余弦变换（Type II标准）（一维信号）
    - Bilinear interpolation (2D signals) 双线性插值（二维信号）
    - Nearest neighbor interpolation (1D and 2D signals) 最近邻插值（二维信号）
    - Autocorrelation (1D signals) 自相关（一维信号）
    - Signal windowing 信号窗口
    - Text tokenization 文本分词
    - Feature hashing 特征哈希
    - Feature standardization 特征标准化
    - One-hot encoding / decoding One-hot编/解码
    - Huffman coding / decoding 赫夫曼编/解码
    - Term frequency-inverse document frequency encoding 词频逆文档频率编码
    - MFCC encoding

11. **Utilities** 工具
    - Similarity kernels 相似度核
    - Distance metrics 距离度量
    - Priority queues 优先级队列
    - Ball tree data structure 波尔树数据结构

## Contributing 代码贡献

Am I missing your favorite model? Is there something that could be cleaner /
less confusing? Did I mess something up? Submit a PR! The only requirement is
that your models are written with just the [Python standard
library](https://docs.python.org/3/library/) and [numpy](https://www.numpy.org/). The
[SciPy library](https://scipy.github.io/devdocs/) is also permitted under special
circumstances ;)

See full contributing guidelines [here](./CONTRIBUTING.md). 

我遗漏了你喜欢的模型了？有更清晰或者简洁的代码吗？我哪里搞砸了? 提交 PR! 唯一的要求就是，你的模型必须用 [Python standard
library](https://docs.python.org/3/library/) 和 [numpy](https://www.numpy.org/)实现。特殊情况下[SciPy library](https://scipy.github.io/devdocs/)也是可以用的 ;)

完整代码贡献规则请看 [这里](./CONTRIBUTING-CN.md). 
