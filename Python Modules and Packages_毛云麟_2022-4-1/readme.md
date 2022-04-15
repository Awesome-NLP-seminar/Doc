# Python Modules and Packages

> 参考：
> 1. [彻底明白Python package和模块](https://www.jianshu.com/p/178c26789011)
> 2. [Python Modules vs Packages](https://data-flair.training/blogs/python-modules-vs-packages/#:~:text=A%20module%20is%20a%20file,does%20not%20apply%20to%20modules.)
> 3. [`__init__.py`的神奇用法](https://zhuanlan.zhihu.com/p/115350758)

## 讲解

[learn.ipynb](learn.ipynb) 有演示代码

***具体讲解查看 [文件](learn.html)***

## 总结

1. 在 `from package import *` 语句中，如果 `__init__.py` 中定义了 `__all__` 魔法变量，那么在`__all__`内的所有元素都会被作为模块自动被导入（ImportError任然会出现，如果自动导入的模块不存在的话）。

2. 如果 `__init__.py` 中没有 `__all__` 变量，导出将按照以下规则执行：
    - 此 package 被导入，并且执行 `__init__.py` 中可被执行的代码
    - `__init__.py` 中定义的 variable 被导入
    - `__init__.py` 中被显式导入的 module 被导入

3. `__init__.py`的设计原则
   
   `__init__.py`的原始使命是声明一个模块，所以它可以是一个空文件。在`__init__.py`中声明的所有类型和变量，就是其代表的模块的类型和变量。我们在利用`__init__.py`时，应该遵循如下几个原则：

    - 不要污染现有的命名空间。模块一个目的，是为了避免命名冲突，如果你在种用`__init__.py`时违背这个原则，是反其道而为之，就没有必要使用模块了。

    - 利用`__init__.py`对外提供类型、变量和接口，对用户隐藏各个子模块的实现。一个模块的实现可能非常复杂，你需要用很多个文件，甚至很多子模块来实现，但用户可能只需要知道一个类型和接口。就像我们的arithmetic例子中，用户只需要知道四则运算有add、sub、mul、dev四个接口，却并不需要知道它们是怎么实现的，也不想去了解arithmetic中是如何组织各个子模块的。由于各个子模块的实现有可能非常复杂，而对外提供的类型和接口有可能非常的简单，我们就可以通过这个方式来对用户隐藏实现，同时提供非常方便的使用。

    - 只在`__init__.py`中导入有必要的内容，不要做没必要的运算。像我们的例子，import arithmetic语句会执行__ini__.py中的所有代码。如果我们在`__init__.py`中做太多事情，每次import都会有额外的运算，会造成没有必要的开销。一句话，`__init__.py`只是为了达到B中所表述的目的，其它事情就不要做啦。

