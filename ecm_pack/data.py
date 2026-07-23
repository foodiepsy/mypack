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


def build_3d_table(x0, x1, x2, func):
    """
    从函数 func(x0, x1, x2) 生成规则 3D 网格查表数据。
    用于「用解析式生成标定数据」或「把曲线函数转成表」。
    """
    X0, X1, X2 = np.meshgrid(x0, x1, x2, indexing="ij")
    V = np.vectorize(func)(X0, X1, X2)
    return (tuple(np.asarray(a) for a in (x0, x1, x2)), V)
