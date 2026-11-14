# -*- coding: utf-8 -*-
"""几何与自由空间。

P、Q 为平面折线，点坐标均为有理数（Fraction）。
对单元格 (i,j)（P 的第 i 段 × Q 的第 j 段，i in [0,m-1), j in [0,n-1)），
局部坐标 s,t in [0,1]：

    P(i+s) = Pi + s*dP_i,   Q(j+t) = Qj + t*dQ_j

记 A = Pi - Qj，D = dP_i - dQ_j。约束
    ||P(i+s) - Q(j+t)||^2 <= epsilon^2
展开为关于 s,t 的二次函数 f(s,t) = A2 + 2 A·D s - 2 A·dQ t
        + D2 s^2 - 2 dP·dQ s t + dQ2 t^2 - eps2。
f 为凸函数（平方距离），故每个单元的自由空间 {f<=0} 为凸集。

四条边上 f 退化为一元二次（或更低次），精确求其与 [0,1] 的交区间，
端点为有理数或单项根式 a+b*sqrt(c)。
"""

from fractions import Fraction as Q

from .exact import Algnum, Expr, sign_of


# ---------- 向量运算（有理） ----------

def vsub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def vadd(a, b):
    return (a[0] + b[0], a[1] + b[1])


def vscale(a, k):
    return (a[0] * k, a[1] * k)


def vdot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def vlen2(a):
    return vdot(a, a)


def point_at(poly, k, t):
    """poly 第 k 段上局部参数 t in [0,1] 处的点（有理）。"""
    a = poly[k]
    b = poly[k + 1]
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


# ---------- 一元二次 f(x)=a x^2 + b x + c 的精确区间求解 ----------

def _quadratic_roots(a: Q, b: Q, c: Q):
    """a x^2 + b x + c = 0 的实根（升序 Algnum 列表，0/1/2 个）。"""
    disc = b * b - 4 * a * c
    sd = sign_of(Expr.rational(disc))
    if sd < 0:
        return []
    if sd == 0:
        return [Algnum.from_q(-b / (2 * a))]
    sqrt_d = Algnum.sqrt_rational(disc)  # sqrt(判别式)
    two_a = 2 * a
    r1 = Algnum.from_q(-b / two_a) + sqrt_d.mul_rational(Q(1, two_a.numerator) * Q(two_a.denominator)
                                                         if False else Q(1) / two_a)
    r0 = Algnum.from_q(-b / two_a) - sqrt_d.mul_rational(Q(1) / two_a)
    return [r0, r1]


def solve_convex_interval(a: Q, b: Q, c: Q):
    """求 {x in [0,1] : a x^2 + b x + c <= 0}，a >= 0。

    返回 (lo, hi)（Algnum，闭区间）或 None。a=0 时按线性/常数处理。
    """
    a, b, c = Q(a), Q(b), c if isinstance(c, Algnum) else Q(c)
    # 常数项可能来自精确的代数运算？本实现中边的二次式系数全为有理数。
    assert isinstance(c, Q)
    if a == 0:
        if b == 0:
            return (Algnum(Q(0)), Algnum(Q(1))) if c <= 0 else None
        # b x + c <= 0
        root = Algnum.from_q(-c / b)
        zero, one = Algnum(Q(0)), Algnum(Q(1))
        if b > 0:  # x <= root
            if root <= zero:
                return None
            return (zero, min_algnum(root, one))
        else:      # x >= root
            if root >= one:
                return None
            return (max_algnum(zero, root), one)
    roots = _quadratic_roots(a, b, c)
    if len(roots) < 2:
        # 判别式 <=0：a>0 时 f 恒非负；等于 0 仅切点，若切点在 [0,1] 内，
        # 自由集是一个点（等号通过，必须保留）。
        if len(roots) == 1 and Algnum(Q(0)) <= roots[0] <= Algnum(Q(1)):
            return (roots[0], roots[0])
        return None
    lo, hi = roots
    zero, one = Algnum(Q(0)), Algnum(Q(1))
    lo = max_algnum(lo, zero)
    hi = min_algnum(hi, one)
    if lo <= hi:
        return (lo, hi)
    return None


def max_algnum(x: Algnum, y: Algnum) -> Algnum:
    return x if x >= y else y


def min_algnum(x: Algnum, y: Algnum) -> Algnum:
    return x if x <= y else y


# ---------- 单元格 ----------

class Cell:
    """缓存单元格 (i,j) 的几何量与四条边的自由区间。"""

    __slots__ = ("i", "j", "a2", "ad", "a_dq", "d2", "dp_dq", "dq2",
                 "bottom", "top", "left", "right")

    def __init__(self, P, Q, i, j, eps2: Q):
        self.i, self.j = i, j
        Pi, Pj = P[i], P[i + 1]
        Qj, Qj1 = Q[j], Q[j + 1]
        dP = vsub(Pj, Pi)
        dQ = vsub(Qj1, Qj)
        A = vsub(Pi, Qj)
        D = vsub(dP, dQ)
        self.a2 = vlen2(A) - eps2                       # f(0,0)
        self.ad = vdot(A, D)                            # f 中 s 一次项系数之半
        self.a_dq = vdot(A, dQ)                         # f 中 t 一次项系数之半（带负号）
        self.d2 = vlen2(D)                              # s^2 系数
        self.dp_dq = vdot(dP, dQ)                       # st 系数之半（带负号）
        self.dq2 = vlen2(dQ)                            # t^2 系数
        # 底边 t=0：f(s,0) = d2 s^2 + 2 ad s + a2
        self.bottom = solve_convex_interval(self.d2, 2 * self.ad, self.a2)
        # 顶边 t=1：f(s,1) = d2 s^2 + 2(ad - dp_dq) s + (a2 - 2 a_dq + dq2)
        self.top = solve_convex_interval(
            self.d2, 2 * (self.ad - self.dp_dq),
            self.a2 - 2 * self.a_dq + self.dq2)
        # 左边 s=0：f(0,t) = dq2 t^2 - 2 a_dq t + a2
        self.left = solve_convex_interval(self.dq2, -2 * self.a_dq, self.a2)
        # 右边 s=1：f(1,t) = dq2 t^2 - 2(a_dq + dp_dq) t + (a2 + 2 ad + d2)
        self.right = solve_convex_interval(
            self.dq2, -2 * (self.a_dq + self.dp_dq),
            self.a2 + 2 * self.ad + self.d2)


def corner_free(P, Q, i, j, eps2: Q) -> bool:
    """网格角点 (i,j) 是否自由：||P(i)-Q(j)||^2 <= eps2（纯有理判定）。"""
    d = vsub(P[i], Q[j])
    return vlen2(d) <= eps2
