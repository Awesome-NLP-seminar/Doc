"""An empty effects package

This is the effects package, providing hardly anything!"""

# import reverse #报错找不到 module

# import effects.reverse #报错找不到 module

# import sound.effects.reverse # 绝对路径成功

# from . import surround # 相对路径成功

## https://blog.csdn.net/suiyueruge1314/article/details/102683546
# 注意当前路径
from ..filters import equalizer # ValueError: attempted relative import beyond top-level package


# 使用通配符 * 时，都会被导入
# __all__ = ["echo", "reverse"]

print("effects package is getting imported!")
