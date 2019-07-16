# Utilities 工具集

The utilities module implements a number of useful functions and objects that
power other ML algorithms across the repo.
工具集实现了一组有用的方法和对象，以处理ML算法。

- `data_structures.py` implements a few useful data structures
    实现一些方便的数据结构
    - A max- and min-heap ordered priority queue
      一个最大/最小 堆排序优先队列
    - A [ball tree](https://en.wikipedia.org/wiki/Ball_tree) with the KNS1 algorithm ([Omohundro, 1989](http://ftp.icsi.berkeley.edu/ftp/pub/techreports/1989/tr-89-063.pdf); [Moore & Gray, 2006](http://people.ee.duke.edu/~lcarin/liu06a.pdf))
      用KNS1算法实现的波尔树
- `kernels.py` implements several general-purpose similarity kernels
  实现一组普通的相似核函数
    - Linear kernel 线性核函数
    - Polynomial kernel 多项核函数
    - Radial basis function kernel 径向基函数内核函数

- `distance_metrics.py` implements common distance metrics
  实现通用距离矩阵
    - Euclidean (L2) distance 欧式距离（L2）
    - Manhattan (L1) distance 曼哈顿距离（L1）
    - Chebyshev (L-infinity) distance 切比雪夫距离（L-inf）
    - Minkowski-p distance 明氏距离
    - Hamming distance 汉明距离

- `windows.py` implements several common windowing functions
  实现通用窗口函数
    - Hann 汉恩窗口
    - Hamming 汉明窗口
    - Blackman-Harris 布莱克曼-哈里斯窗口
    - Generalized cosine 广义余弦窗口

- `testing.py` implements helper functions that prove useful when writing unit
  tests, including data generators and various assert statements
  实现helper方法，供单元测试使用，包括数据生成与一些断言。
