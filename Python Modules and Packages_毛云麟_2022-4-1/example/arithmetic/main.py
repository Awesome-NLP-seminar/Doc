# 四种导入方式

import ops.add
import ops.sub as sub

from ops.mul import mul
from ops import dev

def letscook(x, y, oper):
    r = 0
    if oper == "+":
        r = ops.add.add(x, y)
    elif oper == "-":
        r = sub.sub(x, y)
    elif oper == "*":
        r = mul(x, y)
    else:
        r = dev.dev(x, y)

    print("{} {} {} = {}".format(x, oper, y, r))

x, y = 3, 8

letscook(x, y, "+")
letscook(x, y, "-")
letscook(x, y, "*")
letscook(x, y, "/")