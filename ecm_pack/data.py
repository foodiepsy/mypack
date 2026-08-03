# data.py
# 参数查表 / 插值工具：把「常量、可调用对象、数据表」统一成可调用函数
# 设计目标：工业 ECM 的 OCV/R0/Rk/Ck/dUdT 都能以 1D/2D/3D 查表形式灌入。
import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d


def lookup_1d(x, y):
    """1D 查表：f(x) -> y（SoC -> OCV 等）。边界外推。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    f = interp1d(x, y, bounds_error=False, fill_value=(y[0], y[-1]))
    return lambda xq: float(f(xq))


def lookup_nd(axes, values):
    """
    N 维规则网格查表（2D/3D）。
    axes  : tuple of 1D arrays, 例如 (T, I, SoC)
    values: ndarray, 与网格同形状
    支持规则网格外推（fill_value=None）。
    """
    axes = tuple(np.asarray(a, dtype=float) for a in axes)
    values = np.asarray(values, dtype=float)
    interp = RegularGridInterpolator(
        axes, values, bounds_error=False, fill_value=None
    )

    def _f(*args):
        pt = np.atleast_2d(np.asarray(args, dtype=float))
        return float(interp(pt)[0])

    return _f


def as_callable(value, n_inputs=None):
    """
    把参数归一化为可调用函数：
      - 标量 / 数组常量 -> 忽略输入返回常量
      - 可调用对象     -> 原样返回
      - (axes, values) -> 查表
    这样 ECM 的每一个参数（OCV、R0、Rk、Ck、dUdT…）都表现出一致的接口。
    """
    if callable(value):
        return value
    # 查表数据：(axes_tuple, values) 或 (x, y)
    if isinstance(value, tuple) and len(value) == 2:
        axes, vals = value
        if isinstance(axes, tuple):
            return lookup_nd(axes, vals)
        else:
            return lookup_1d(axes, vals)
    # 标量常量
    const = float(value)

    def _const(*args):
        return const

    return _const


def temp_aware(value, empirical=None):
    """
    电气参数「温度行为」双路径统一分发器（核心约定）：

    - 可调用对象           -> 原样返回（最高优先级，用户已给出完整函数）
    - (axes, values) 数据表 -> 查表（轴顺序约定：
        1D 表 = (T_degC)；
        2D 表 = (SoC, T_degC)；
        3D 表 = (SoC, T_degC, I)）
    - 标量                 -> 套用 empirical(base) 经验公式（如 Arrhenius
        温度修正、容量-温度折损）；empirical=None 时退化为常数（向后兼容）。

    返回的函数统一签名 (Tdeg, I=0.0, soc=1.0) -> 参数值。
    这样「固定值 → 经验值变化 / 传入表 → 读表」两套逻辑由调用方一行切换。
    """
    if callable(value):
        return value
    if isinstance(value, tuple) and len(value) == 2:
        axes, vals = value
        if isinstance(axes, tuple):
            nd = len(axes)
            f = lookup_nd(axes, vals)

            def _g(Tdeg, I=0.0, soc=1.0, _f=f, _nd=nd):
                if _nd >= 3:      # (SoC, T_degC, I)
                    return _f(soc, Tdeg, I)
                elif _nd == 2:    # (SoC, T_degC)
                    return _f(soc, Tdeg)
                else:             # (T_degC)
                    return _f(Tdeg)

            return _g
        f = lookup_1d(axes, vals)
        return lambda Tdeg, I=0.0, soc=1.0, _f=f: _f(Tdeg)
    base = float(value)
    if empirical is None:
        def _const(*_a, _b=base):
            return _b

        return _const
    return empirical(base)


def build_3d_table(x0, x1, x2, func):
    """
    从函数 func(x0, x1, x2) 生成规则 3D 网格查表数据。
    用于「用解析式生成标定数据」或「把曲线函数转成表」。
    """
    X0, X1, X2 = np.meshgrid(x0, x1, x2, indexing="ij")
    V = np.vectorize(func)(X0, X1, X2)
    return (tuple(np.asarray(a) for a in (x0, x1, x2)), V)
