import ops as op

def letscook(x, y, oper):
    r = 0
    if oper == "+":
        r = op.add(x, y)
    elif oper == "-":
        r = op.sub(x, y)
    elif oper == "*":
        r = op.mul(x, y)
    else:
        r = op.dev(x, y)

    print("{} {} {} = {}".format(x, oper, y, r))

x, y = 3, 8

letscook(x, y, "+")
letscook(x, y, "-")
letscook(x, y, "*")
letscook(x, y, "/")