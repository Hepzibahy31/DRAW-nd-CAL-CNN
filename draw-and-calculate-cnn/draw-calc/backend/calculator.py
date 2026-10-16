"""Safe calculator. It never uses eval(): it reads the characters itself and applies
the BODMAS rule (x and / first, then + and -), using exact fractions."""
from fractions import Fraction

OPS = "+-×÷"


def parse(chars):
    chars = [c for c in chars if c]
    if chars and chars[-1] == "=":
        chars = chars[:-1]
    if "=" in chars:
        raise ValueError("The = sign can only be at the end.")
    if not chars:
        raise ValueError("Nothing to calculate yet.")
    if len(chars) > 30:
        raise ValueError("Too many characters (maximum 30).")
    nums, ops, cur, neg = [], [], "", False
    for i, c in enumerate(chars):
        if c.isdigit():
            cur += c
        elif c in OPS:
            if cur == "":
                if c == "-" and i == 0:
                    neg = True
                    continue
                raise ValueError("Two operators in a row, or an operator at the start.")
            nums.append(Fraction(int(cur)) * (-1 if neg else 1)); neg = False
            ops.append(c); cur = ""
        else:
            raise ValueError("Unknown character: " + c)
    if cur == "":
        raise ValueError("The expression cannot end with an operator.")
    if len(cur) > 12:
        raise ValueError("Numbers can have at most 12 digits.")
    nums.append(Fraction(int(cur)) * (-1 if neg else 1))
    for n in nums:
        if len(str(abs(n.numerator))) > 12:
            raise ValueError("Numbers can have at most 12 digits.")
    return nums, ops


def show(x):
    if x.denominator == 1:
        return str(x.numerator)
    return ("%.6f" % float(x)).rstrip("0").rstrip(".")


def calculate(chars):
    nums, ops = parse(chars)
    steps = []
    expr = lambda: " ".join(show(n) if i == 0 else o + " " + show(n) for i, (o, n) in enumerate(zip([""] + ops, nums)))
    for group in ("×÷", "+-"):
        while any(o in group for o in ops):
            i = next(k for k, o in enumerate(ops) if o in group)
            a, b, o = nums[i], nums[i + 1], ops[i]
            if o == "÷" and b == 0:
                raise ValueError("Division by zero is not possible.")
            r = a + b if o == "+" else a - b if o == "-" else a * b if o == "×" else a / b
            nums[i:i + 2] = [r]; ops.pop(i)
            steps.append({"text": "%s %s %s = %s" % (show(a), o, show(b), show(r)), "left": expr() if ops else show(r)})
    return {"result": show(nums[0]), "steps": steps}
