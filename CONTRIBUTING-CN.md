## Contributing 代码贡献

Thank you for contributing to numpy-ml! 
感谢你对numpy-ml的支持！

### General guidelines 常规指引
1. Please include a clear list of what you've done
  请包含你所做事情的清晰列表
2. For pull requests, please make sure all commits are [*atomic*](https://en.wikipedia.org/wiki/Atomic_commit) (i.e., one feature per commit)
  为了pull请求，请确保所有提交是 [*atomic*](https://en.wikipedia.org/wiki/Atomic_commit) (即一次提交一个功能)
3. If you're submitting a new model / feature / module, **please include proper documentation and unit tests.**
  如果你提交新模型、功能、模块， ** 请包含相应的文档和单元测试。 **
    - See the `test.py` file in one of the existing modules for examples of unit tests.
      参考现有模块里面的 `test.py` 来做单元测试。
    - Documentation is loosely based on the [NumPy docstring style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html). When in doubt, refer to existing examples 
      文档标准参考[NumPy文档风格](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html). 如果有疑惑参考现有示例
4. Please format your code using the [black](https://github.com/python/black) defaults. You can use this [online formatter](https://black.now.sh/).
  请用[black](https://github.com/python/black) 格式化你的代码。你可以使用在线格式化工具(https://black.now.sh/)。

### Specific guidelines 特殊指引
#### I have a new model / model component to contribute 新模型/模型组件贡献
- Awesome - create a [pull request](https://github.com/ddbourgin/numpy-ml/pulls)! When preparing your PR, please include a brief description of the model, the canonical reference(s) in the literature, and, most importantly unit tests against an existing implementation!
  非常棒 - 首先创建一个[PR](https://github.com/ddbourgin/numpy-ml/pulls)！当你准备好了，请包含一个模型的简要概述，权威的参考文献，并且，最主要的是对现有实现的做单元测试！
  - Refer to the `test.py` file in one of the existing modules for examples.
    参考现有模块里面的 `test.py` 来做单元测试。
  

#### I have a major new enhancement / adjustment that will affect multiple models
#### 重要提升/校正，会影响多个模型
- Please post an [issue](https://github.com/ddbourgin/numpy-ml/issues) with your proposal before you begin working on it. When outlining your proposal, please include as much detail about your intended changes as possible.
  请发起[issue](https://github.com/ddbourgin/numpy-ml/issues)，填写你的建议和意见，并开始你的代码。当你发表了你的意见或建议，请尽量包含详细的内容和你清晰的意图。

#### I found a bug
#### 发现了一个bug
- If there isn't already an [open issue](https://github.com/ddbourgin/numpy-ml/issues), please start one! When creating your issue, include:
  如果有bug并且不在[open issue](https://github.com/ddbourgin/numpy-ml/issues)里面，请提交bug！当你创建issue，请包含：
  1. A title and clear description 一个简短的清晰描述
  2. As much relevant information as possible 尽量与bug信息相关
  3. A code sample demonstrating the expected behavior that is not occurring 一端与预期行为不一致的代码示例演示

#### I fixed a bug
#### 修复bug
- Thank you! Please open a new [pull request](https://github.com/ddbourgin/numpy-ml/pulls) with the patch. When doing so, ensure the PR description clearly describes the problem and solution. Include the relevant issue number if applicable.
  谢谢！请开一个[PR](https://github.com/ddbourgin/numpy-ml/pulls)，附上你的补丁。当你做完补丁，确保PR清晰描述问题与解决方案。issue号如果存在，请提供相关的issue号。
