# -*- coding: utf-8 -*-
import sys
import io

# 强制标准输出使用 UTF-8（解决 Windows 控制台编码问题）
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import math
import cmath
import random

# ============================
# CKKS Parameters
# ============================

N = 8                     # ring dimension
slots = N // 2

q = 1 << 10
Q = 1 << 60
Delta = 16

random.seed(42)
np.random.seed(42)

# primitive root
zeta = cmath.exp(1j * math.pi / N)

# ============================
# Evaluation Points
# ============================

roots = [
    zeta ** (2 * i + 1)
    for i in range(slots)
]

# ============================
# Vandermonde Matrix
# ============================

U = np.zeros((slots, N), dtype=complex)

for i in range(slots):
    for j in range(N):
        U[i, j] = roots[i] ** j

# ==========================================
# U_a, U_b
# ==========================================

U_a = np.zeros((slots, slots), dtype=complex)
U_b = np.zeros((slots, slots), dtype=complex)
for i in range(slots):
    for j in range(slots):
        U_a[i, j] = roots[i] ** j
        U_b[i, j] = roots[i] ** (j + slots)

# CRT = (U;\overline{U})
CRT = np.vstack([
    U,
    np.conjugate(U)
])
CRT_inv = CRT.conjugate().T / N


def encode(coeffs):
    coeffs = np.asarray(coeffs)
    return U @ coeffs

# ============
# 矩阵乘法本质是线性组合，密文下可以通过加乘旋转实现；下面的STC同理
# ============
def CTS(z):
    z = np.asarray(z, dtype=complex)
    z_bar = np.conjugate(z)
    ta = (
        U_a.conjugate().T @ z
        +
        U_a.T @ z_bar
    ) / N
    tb = (
        U_b.conjugate().T @ z
        +
        U_b.T @ z_bar
    ) / N
    return ta.real, tb.real

# ============
# 采用直接的正弦近似，实际需要使用多项式逼近；这里只是做同态取模的说明。
# ============
def EvalMod(x):
    return (
        q/(2*math.pi)
        *
        np.sin(
            2*math.pi*x/q
        )
    )


def STC(ta, tb):
    ta = np.asarray(ta, dtype=complex)
    tb = np.asarray(tb, dtype=complex)
    z = (U_a @ ta + U_b @ tb)
    return z


def Decode(z):
    # 将长度为 slots 的槽值 z 扩展为长度为 N 的完整向量 z_ext，满足共轭对称性
    z_ext=np.concatenate([
        z,
        np.conjugate(z)
    ])
    t = CRT_inv @ z_ext
    return t.real

# ============================
# simulate ModRaise output
# ============================

I = np.random.randint(-3, 4, N)
m = np.random.uniform(-0.5, 0.5, N)
e = np.random.randint(-1, 2, N)
# 这里直接计算构造出t(X)，不采用实际的密文，只是用于说明，因此这里不会用到Q
# 给出t(X)，希望得到取模q后的结果
# 在实际操作时，需要保证Q远大于qI + m的可能值，不会产生模Q的回绕，才能完整保留q*I这一项
t = (q * I + Delta * m + e)
# Encode
z = encode(t)

# CTS
ta,tb = CTS(z)

print("CTS ta")
print(ta)
print("CTS tb")
print(tb)

# EvalMod
ta_new = EvalMod(ta)
tb_new = EvalMod(tb)
print("\nEvalMod ta")
print(ta_new)
print("\nEvalMod tb")
print(tb_new)

z_final = STC(
    ta_new,
    tb_new
)
print("\nSTC output")
print(z_final)

decoded = Decode(z_final)

print("\nDecoded")
print(decoded)

target = Delta * m + e
print("\nTarget")
print(target)