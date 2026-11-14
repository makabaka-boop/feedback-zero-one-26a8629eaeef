# -*- coding: utf-8 -*-
"""精确算术层（仅依赖标准库 fractions）。

提供：
  * Q        : Fraction 的别名（有理数 p/q，分母恒为正）。
  * Algnum   : 单项根式 a + b*sqrt(c)，a,b 为有理数，c 为非负平方自由整数。
  * Expr     : 一般根式线性组合（dict: 平方自由整数 -> 有理系数），
               用于跨根式比较时的精确符号判定。
  * sign_of  : 递归“取共轭乘式”精确符号判定，纯整数运算，无浮点。

所有比较都基于 sign_of，绝不使用浮点容差。
"""

from fractions import Fraction as Q
from math import isqrt


def squarefree_part(n: int) -> tuple[int, int]:
    """返回 (s, k)，使 n == s * k^2，s 为平方自由正整数，n >= 1。"""
    s = 1
    k = 1
    p = 2
    x = n
    while p * p <= x:
        e = 0
        while x % p == 0:
            x //= p
            e += 1
        if e & 1:
            s *= p
        k *= p ** (e // 2)
        p += 1 if p == 2 else 2  # 2 -> 3, 之后只看奇数
    if x > 1:
        s *= x  # 残余素因子，指数为 1
    return s, k


class Expr:
    """多个根式项的线性组合 sum(coef[sf] * sqrt(sf))，sf=0/1 表示有理项。

    内部字典只保留非零系数；有理项统一存在键 1 下（sqrt(1)=1）。
    """

    __slots__ = ("terms",)

    def __init__(self, terms=None):
        d = {}
        if terms:
            for sf, coef in terms.items():
                if coef:
                    d[sf] = d.get(sf, Q(0)) + coef
        self.terms = {sf: c for sf, c in d.items() if c}

    # --- 基本构造 ---
    @staticmethod
    def rational(x) -> "Expr":
        if isinstance(x, Expr):
            return Expr(x.terms)
        return Expr({1: Q(x)})

    @staticmethod
    def sqrt_fraction(r: Q) -> "Expr":
        """sqrt(r)，r >= 0 的有理数，返回精确表达式。"""
        r = Q(r)
        if r < 0:
            raise ValueError("sqrt of negative rational")
        num_s, num_k = squarefree_part(r.numerator)
        den_s, den_k = squarefree_part(r.denominator)
        # sqrt(r) = num_k/den_k * sqrt(num_s/den_s)
        sf = num_s * den_s
        coef = Q(num_k, den_k)
        # num_s, den_s 各自平方自由但乘积可能不平方自由（公共素因子指数和为 2）
        s2, k2 = squarefree_part(sf)
        coef *= k2
        return Expr({s2: coef})

    # --- 算术 ---
    def __add__(self, other) -> "Expr":
        other = other if isinstance(other, Expr) else Expr.rational(other)
        d = dict(self.terms)
        for sf, c in other.terms.items():
            d[sf] = d.get(sf, Q(0)) + c
        return Expr(d)

    __radd__ = __add__

    def __sub__(self, other) -> "Expr":
        return self + (-other if isinstance(other, Expr) else Expr.rational(-Q(other)))

    def __neg__(self) -> "Expr":
        return Expr({sf: -c for sf, c in self.terms.items()})

    def __mul__(self, other) -> "Expr":
        other = other if isinstance(other, Expr) else Expr.rational(other)
        d = {}
        for s1, c1 in self.terms.items():
            for s2, c2 in other.terms.items():
                sf, k = squarefree_part(s1 * s2)
                d[sf] = d.get(sf, Q(0)) + c1 * c2 * k
        return Expr(d)

    __rmul__ = __mul__

    def scale(self, q) -> "Expr":
        q = Q(q)
        return Expr({sf: c * q for sf, c in self.terms.items()})

    def is_zero(self) -> bool:
        return not self.terms

    # --- 精确符号 ---
    def sign(self) -> int:
        return sign_of(self)

    def __float__(self) -> float:  # 仅供人工调试，不参与任何判定
        return sum(float(c) * (1.0 if sf == 1 else isqrt(sf) ** 0.5)
                   for sf, c in self.terms.items())

    def __repr__(self) -> str:
        return "Expr(%s)" % self.terms


def sign_of(e: Expr) -> int:
    """精确判定 e 的符号，返回 -1/0/1。

    方法：挑出最大平方自由键 s，把 e 写成 A + B*sqrt(s)（A 不含 s 项）。
    e 与“共轭” A - B*sqrt(s) 的乘积为 A^2 - s*B^2，消去了根式 s，
    递归判定该乘积与 A 的符号即可恢复 e 的符号；每次消去一个根式，必终止。
    全程整数/有理数运算。
    """
    if not isinstance(e, Expr):
        e = Expr.rational(e)
    if not e.terms:
        return 0
    s = max(e.terms)
    if s <= 1:
        # 仅剩有理项
        v = e.terms.get(1, Q(0))
        return 0 if v == 0 else (1 if v > 0 else -1)
    b = e.terms.get(s, Q(0))
    a = Expr({sf: c for sf, c in e.terms.items() if sf != s})
    if a.is_zero():
        return 1 if b > 0 else (-1 if b < 0 else 0)
    prod = a * a - Expr({s: b * b}).scale(s)  # A^2 - s*B^2
    p_sign = sign_of(prod)
    if p_sign == 0:
        # A^2 = s B^2，sqrt(s) 无理（s>1 平方自由），仅当 A=B=0 成立
        return 0
    if p_sign > 0:
        # A^2 - sB^2 > 0：|A| > |B|sqrt(s)，e 与 A 同号
        return sign_of(a)
    # p_sign < 0：|A| < |B|sqrt(s)，e 与 B 同号
    return 1 if b > 0 else -1


class Algnum:
    """单项根式 a + b*sqrt(c)（a,b 为有理数，c 为非负平方自由整数）。

    见证输出的每个坐标均能表示成该形式（见任务规格 [a,b,c]）。
    比较时提升为 Expr 走精确符号判定。
    """

    __slots__ = ("a", "b", "c")

    def __init__(self, a: Q, b: Q = Q(0), c: int = 0):
        self.a = Q(a)
        self.b = Q(b)
        if c < 0:
            raise ValueError("radicand must be >= 0")
        # 规范化：c 平方自由；完全平方已在 sqrt_fraction 里折进系数，
        # 这里再保险处理 c=1 与零系数。
        if c > 1:
            sf, k = squarefree_part(c)
            if sf != c:
                self.b = self.b * k
                c = sf
        if self.b == 0:
            c = 0
        self.c = c

    # --- 构造 ---
    @staticmethod
    def from_q(x: Q) -> "Algnum":
        return Algnum(Q(x))

    @staticmethod
    def sqrt_rational(r: Q) -> "Algnum":
        """sqrt(r)（r >= 0 有理数），返回单项 Algnum。"""
        e = Expr.sqrt_fraction(Q(r))
        # sqrt_fraction 保证结果是单项
        if not e.terms:
            return Algnum(Q(0))
        sf, coef = next(iter(e.terms.items()))
        if sf == 1:
            return Algnum(coef)
        return Algnum(Q(0), coef, sf)

    def to_expr(self) -> Expr:
        if self.b == 0 or self.c == 0:
            return Expr.rational(self.a)
        return Expr({1: self.a, self.c: self.b})

    # --- 算术（结果仍为单项根式：同 c 的加减、有理倍乘） ---
    def __add__(self, other: "Algnum") -> "Algnum":
        other = other if isinstance(other, Algnum) else Algnum.from_q(Q(other))
        if self.c == other.c:
            return Algnum(self.a + other.a, self.b + other.b, self.c)
        if other.c == 0:
            return Algnum(self.a + other.a, self.b, self.c)
        if self.c == 0:
            return Algnum(self.a + other.a, other.b, other.c)
        raise ValueError("adding distinct radicals is not representable as Algnum")

    def __sub__(self, other: "Algnum") -> "Algnum":
        return self + Algnum(-other.a, -other.b, other.c)

    def __neg__(self) -> "Algnum":
        return Algnum(-self.a, -self.b, self.c)

    def mul_rational(self, q: Q) -> "Algnum":
        return Algnum(self.a * Q(q), self.b * Q(q), self.c)

    # --- 比较（精确） ---
    def _cmp(self, other) -> int:
        other = other if isinstance(other, Algnum) else Algnum.from_q(Q(other))
        return sign_of(self.to_expr() - other.to_expr())

    def __eq__(self, other) -> bool:
        try:
            return self._cmp(other) == 0
        except (ValueError, TypeError):
            return NotImplemented

    def __lt__(self, other):
        return self._cmp(other) < 0

    def __le__(self, other):
        return self._cmp(other) <= 0

    def __gt__(self, other):
        return self._cmp(other) > 0

    def __ge__(self, other):
        return self._cmp(other) >= 0

    def __hash__(self):
        return hash((self.a, self.b, self.c))

    def __repr__(self):
        if self.c:
            return "(%s + %s*sqrt(%d))" % (self.a, self.b, self.c)
        return repr(self.a)


def qstr(r: Q) -> str:
    """有理数 -> 约分正分母 p/q 串；整数写成 p（等价于 p/1，取最简形式）。"""
    r = Q(r)
    if r.denominator == 1:
        return str(r.numerator)
    return "%d/%d" % (r.numerator, r.denominator)


def algnum_to_json(x: Algnum):
    """Algnum -> 见证 JSON 值：有理给 p/q 串，根式给 [a,b,c]（三项均为有理串）。"""
    if not isinstance(x, Algnum):
        x = Algnum.from_q(Q(x))
    if x.c == 0 or x.b == 0:
        return qstr(x.a)
    return [qstr(x.a), qstr(x.b), str(x.c)]
