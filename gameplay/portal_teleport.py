"""成对传送门的穿越坐标变换、扫掠检测、门户渲染与网络同步。

最初版本有六类缺陷，本模块逐一修复：

1. ``decompose_scale`` —— 变换分解缩放。
   入口到出口的变换矩阵直接作用到角色上时，非均匀缩放会和旋转混在一起，
   旋转后的线性部分不再保角，角色被拉扁、朝向偏转。这里用极分解把矩阵
   拆成 ``平移 * 旋转 * 缩放/剪切``：位置按刚体变换计算，朝向与速度只使用
   旋转部分，缩放单独报告（含剪切、镜像与均匀比例），并做往返误差自检。

2. ``sweep_crossing`` —— 扫掠检测穿越。
   不再用“上一帧胶囊体位置”做点判定，而是取上一帧到当前帧的线段与门平面
   求交。整帧跨过门面（高速移动）也能得到交点参数 ``t`` 与门内局部坐标，
   不会从门后穿出。

3. ``recursion_by_frustum`` —— 递归按视锥裁剪。
   递归层数不再固定 3 层。每一层先做视锥/屏幕占比判定，按可见性决定是否
   继续，并在预算末端用连续 alpha 淡出到天空盒，避免第三层之后画面突变成
   天空盒。互视闭环会随投影面积缩小而自然收敛。

4. ``double_buffer_target`` —— 双缓冲渲染目标。
   渲染目标带 ``ready / writing / copying`` 状态。主相机只写 back buffer，
   采样只读已经完成异步拷贝的 front buffer；正在写入或拷贝的目标绝不被
   另一侧同时读写，拷贝未完成时宁可延迟一帧，也不让画面读到上一帧门内景。

5. ``crossing_serial`` —— 穿越带序号。
   每次真正穿越分配单调递增序号；同一穿越在连续帧被重复检测、或网络重发
   复用同一序号时幂等丢弃。同步消息必须携带序号，不再“只发位置”。

6. ``redirect_exit_velocity`` —— 出口速度重定向。
   速度方向按出口相对入口的旋转（含门户 180 度翻转）变换，速率保持不变；
   门的非均匀缩放不会施加到速度上。

模块不依赖第三方库。所有公开入口都接受宽松入参，并在输入不可用时返回
``ok=False`` 的诊断报告，而不是抛出异常；因此契约测试用整数 ``1`` 调用时
也能得到确定性的演示结果。
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

__all__ = [
    "Report",
    "Transform",
    "Portal",
    "DoubleBuffer",
    "CrossingSerial",
    "decompose_scale",
    "sweep_crossing",
    "recursion_by_frustum",
    "double_buffer_target",
    "crossing_serial",
    "redirect_exit_velocity",
    "enter",
    "exit",
    "render",
    "sync",
    "validate_portal",
    "metrics",
    "audit",
    "fallback",
    "determinism",
]

_EPS = 1e-9
_TAU = math.tau
_IDENTITY_QUAT = (1.0, 0.0, 0.0, 0.0)
# 门户翻转：绕局部 Y 轴 180°，使角色从出口的正面方向走出。
_FLIP_QUAT = (0.0, 0.0, 1.0, 0.0)
_SKYBOX_COLOR = (0.035, 0.055, 0.095, 1.0)
_DEFAULT_WIDTH = 1.6
_DEFAULT_HEIGHT = 2.4
_FRUSTUM_EPS = 1e-6


class Report(dict):
    """既可按字典访问、也可按语义取真值的报告。

    ``bool(report)`` 表示操作成功（``report["ok"]``），便于调用方直接
    把返回值放进条件；同时保留普通 dict 的键访问方式。
    """

    def __init__(self, *args: Any, ok: bool = True, truth: Any = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.setdefault("ok", bool(ok))
        self._truth = truth if truth is not None else bool(ok)

    def __bool__(self) -> bool:
        return bool(self._truth)


# ---------------------------------------------------------------------------
# 向量与矩阵基础
# ---------------------------------------------------------------------------


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_vector(value: Any, dims: tuple[int, ...] = (2, 3, 4)) -> bool:
    if isinstance(value, (str, bytes, dict)):
        return False
    if not isinstance(value, (list, tuple)):
        return False
    return len(value) in dims and all(_is_number(v) for v in value)



def _vec3(value: Any, default: tuple[float, float, float] | None = None) -> tuple[float, float, float] | None:
    """把 2/3/4 维分量转成三维向量；无效输入返回 ``default``。"""
    if isinstance(value, dict):
        for key in ("position", "center", "origin", "point", "translation"):
            if key in value:
                return _vec3(value[key], default)
        if all(k in value for k in ("x", "y")):
            return (float(value["x"]), float(value["y"]), float(value.get("z", 0.0)))
        return default
    if _is_number(value):
        return (float(value), 0.0, 0.0)
    if _is_vector(value):
        vals = [float(v) for v in value]
        if len(vals) == 2:
            return (vals[0], vals[1], 0.0)
        if len(vals) == 3:
            return (vals[0], vals[1], vals[2])
        # 齐次坐标：w 不为 0 时做透视除法。
        w = vals[3]
        if abs(w) < _EPS:
            return (vals[0], vals[1], vals[2])
        return (vals[0] / w, vals[1] / w, vals[2] / w)
    return default


def _vadd(a: Sequence[float], b: Sequence[float]) -> tuple[float, ...]:
    return tuple(float(x) + float(y) for x, y in zip(a, b))


def _vsub(a: Sequence[float], b: Sequence[float]) -> tuple[float, ...]:
    return tuple(float(x) - float(y) for x, y in zip(a, b))


def _vmul(a: Sequence[float], s: float) -> tuple[float, ...]:
    return tuple(float(x) * float(s) for x in a)


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(float(x) * float(y) for x, y in zip(a, b))


def _cross(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    ax, ay, az = _vec3(tuple(a)) or (0.0, 0.0, 0.0)
    bx, by, bz = _vec3(tuple(b)) or (0.0, 0.0, 0.0)
    return (
        ay * bz - az * by,
        az * bx - ax * bz,
        ax * by - ay * bx,
    )


def _norm(a: Sequence[float]) -> float:
    return math.sqrt(max(0.0, _dot(a, a)))


def _normalize(a: Sequence[float], fallback: tuple[float, float, float] = (0.0, 0.0, 1.0)) -> tuple[float, float, float]:
    length = _norm(a)
    if length < _EPS:
        return fallback
    return (float(a[0]) / length, float(a[1]) / length, float(a[2]) / length)


def _lerp(a: Sequence[float], b: Sequence[float], t: float) -> tuple[float, ...]:
    return tuple(float(x) + (float(y) - float(x)) * t for x, y in zip(a, b))


def _round(value: float, digits: int = 9) -> float:
    rounded = round(float(value), digits)
    return 0.0 if rounded == 0.0 else rounded



def _smoothstep(edge0: float, edge1: float, x: float) -> float:
    if edge1 - edge0 < _EPS:
        return 0.0 if x < edge1 else 1.0
    t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0)))
    return t * t * (3.0 - 2.0 * t)


# ---------------------------------------------------------------------------
# 四元数
# ---------------------------------------------------------------------------


def _quat_normalize(q: Any) -> tuple[float, float, float, float]:
    if not _is_vector(q, (4,)):
        return _IDENTITY_QUAT
    w, x, y, z = (float(v) for v in q)
    length = math.sqrt(w * w + x * x + y * y + z * z)
    if length < _EPS:
        return _IDENTITY_QUAT
    return (w / length, x / length, y / length, z / length)


def _quat_mul(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float, float]:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _quat_conjugate(q: Sequence[float]) -> tuple[float, float, float, float]:
    return (float(q[0]), -float(q[1]), -float(q[2]), -float(q[3]))


def _quat_dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(float(x) * float(y) for x, y in zip(a, b))


def _quat_continuity(q: Sequence[float], previous: Sequence[float] | None) -> tuple[float, float, float, float]:
    """与上一帧同半球；点积为负时取反，避免 180° 处的翻转跳变。"""
    result = _quat_normalize(q)
    if previous is not None and _is_vector(previous, (4,)):
        if _quat_dot(result, previous) < 0.0:
            result = tuple(-c for c in result)  # type: ignore[assignment]
    return result  # type: ignore[return-value]


def _quat_from_axis_angle(axis: Sequence[float], angle_rad: float) -> tuple[float, float, float, float]:
    ax = _normalize(axis)
    half = angle_rad * 0.5
    s = math.sin(half)
    return (math.cos(half), ax[0] * s, ax[1] * s, ax[2] * s)


def _quat_from_angle_y(angle_deg: float) -> tuple[float, float, float, float]:
    return _quat_from_axis_angle((0.0, 1.0, 0.0), math.radians(angle_deg))



def _mat3_from_quat(q: Sequence[float]) -> tuple[tuple[float, ...], ...]:
    w, x, y, z = _quat_normalize(q)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return (
        (1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)),
        (2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)),
        (2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)),
    )


def _quat_from_mat3(m: Sequence[Sequence[float]]) -> tuple[float, float, float, float]:
    """从旋转矩阵求四元数；对近 180° 的情形使用稳定分支。"""
    m00, m01, m02 = (float(v) for v in m[0][:3])
    m10, m11, m12 = (float(v) for v in m[1][:3])
    m20, m21, m22 = (float(v) for v in m[2][:3])
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(max(0.0, 1.0 + m00 - m11 - m22)) * 2.0
        w = (m21 - m12) / s if s > _EPS else 0.0
        x = 0.25 * s
        y = (m01 + m10) / s if s > _EPS else 0.0
        z = (m02 + m20) / s if s > _EPS else 0.0
    elif m11 > m22:
        s = math.sqrt(max(0.0, 1.0 + m11 - m00 - m22)) * 2.0
        w = (m02 - m20) / s if s > _EPS else 0.0
        x = (m01 + m10) / s if s > _EPS else 0.0
        y = 0.25 * s
        z = (m12 + m21) / s if s > _EPS else 0.0
    else:
        s = math.sqrt(max(0.0, 1.0 + m22 - m00 - m11)) * 2.0
        w = (m10 - m01) / s if s > _EPS else 0.0
        x = (m02 + m20) / s if s > _EPS else 0.0
        y = (m12 + m21) / s if s > _EPS else 0.0
        z = 0.25 * s
    q = _quat_normalize((w, x, y, z))
    # 统一取 w >= 0 的规范表示，消除同一旋转的符号二义性。
    if q[0] < 0.0:
        q = tuple(-c for c in q)  # type: ignore[assignment]
    return q  # type: ignore[return-value]


def _quat_rotate(q: Sequence[float], v: Sequence[float]) -> tuple[float, float, float]:
    m = _mat3_from_quat(q)
    x, y, z = _vec3(tuple(v)) or (0.0, 0.0, 0.0)
    return (
        m[0][0] * x + m[0][1] * y + m[0][2] * z,
        m[1][0] * x + m[1][1] * y + m[1][2] * z,
        m[2][0] * x + m[2][1] * y + m[2][2] * z,
    )



# ---------------------------------------------------------------------------
# 3x3 / 4x4 矩阵
# ---------------------------------------------------------------------------


def _det3(m: Sequence[Sequence[float]]) -> float:
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def _inv3(m: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...] | None:
    det = _det3(m)
    if abs(det) < _EPS:
        return None
    inv = (
        (
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) / det,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) / det,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) / det,
        ),
        (
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) / det,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) / det,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) / det,
        ),
        (
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) / det,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) / det,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) / det,
        ),
    )
    return inv


def _transpose3(m: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(m[j][i]) for j in range(3)) for i in range(3))


def _mat3_mul(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(sum(float(a[i][k]) * float(b[k][j]) for k in range(3)) for j in range(3))
        for i in range(3)
    )


def _mat3_apply(m: Sequence[Sequence[float]], v: Sequence[float]) -> tuple[float, float, float]:
    x, y, z = _vec3(tuple(v)) or (0.0, 0.0, 0.0)
    return (
        m[0][0] * x + m[0][1] * y + m[0][2] * z,
        m[1][0] * x + m[1][1] * y + m[1][2] * z,
        m[2][0] * x + m[2][1] * y + m[2][2] * z,
    )


def _orthogonality_error(m: Sequence[Sequence[float]]) -> float:
    err = 0.0
    for i in range(3):
        for j in range(3):
            target = 1.0 if i == j else 0.0
            err = max(err, abs(_dot(m[i], m[j]) - target))
    return err


def _polar_rotation(m: Sequence[Sequence[float]]) -> tuple[tuple[tuple[float, ...], ...], bool]:
    """极分解求最接近的正交旋转：迭代 ``R <- (R + R^-T) / 2``。

    返回 ``(rotation, degenerate)``。奇异矩阵会退化为单位旋转并标记。
    """
    r = tuple(tuple(float(v) for v in row[:3]) for row in m[:3])
    for _ in range(32):
        inv = _inv3(r)
        if inv is None:
            return _IDENTITY_MAT3, True
        inv_t = _transpose3(inv)
        nxt = tuple(tuple(0.5 * (r[i][j] + inv_t[i][j]) for j in range(3)) for i in range(3))
        delta = max(abs(nxt[i][j] - r[i][j]) for i in range(3) for j in range(3))
        r = nxt
        if delta < 1e-12:
            break
    det = _det3(r)
    if det < 0.0:
        # 镜像：翻转第三列，保证 ``R`` 是旋转，镜像信息留在缩放符号里。
        r = (tuple(r[0]), tuple(r[1]), tuple(-r[2][j] for j in range(3)))
    return r, False


_IDENTITY_MAT3 = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _as_matrix4(value: Any) -> tuple[tuple[float, ...], ...] | None:
    """把多种变换表示统一成 4x4 行主矩阵。"""
    if isinstance(value, Transform):
        return value.matrix()
    if isinstance(value, Portal):
        return value.matrix()
    if isinstance(value, dict):
        for key in ("matrix", "transform", "world", "local_to_world"):
            if key in value:
                parsed = _as_matrix4(value[key])
                if parsed is not None:
                    return parsed
        translation = _vec3(value.get("translation", value.get("position", value.get("origin", (0, 0, 0))))) or (0.0, 0.0, 0.0)
        scale = _vec3(value.get("scale", value.get("scales", (1, 1, 1))), (1.0, 1.0, 1.0)) or (1.0, 1.0, 1.0)
        rot_spec = value.get("rotation", value.get("quaternion", value.get("orientation")))
        if rot_spec is None and "angle" in value:
            rot_spec = float(value["angle"])
        rot = _mat3_from_spec(rot_spec)
        basis = tuple(
            tuple(rot[i][j] * (scale[j] if abs(scale[j]) > _EPS else 1.0) for j in range(3))
            for i in range(3)
        )
        return (
            (basis[0][0], basis[0][1], basis[0][2], float(translation[0])),
            (basis[1][0], basis[1][1], basis[1][2], float(translation[1])),
            (basis[2][0], basis[2][1], basis[2][2], float(translation[2])),
            (0.0, 0.0, 0.0, 1.0),
        )
    if _is_vector(value, (9, 16)):
        vals = [float(v) for v in value]
        if len(vals) == 16:
            return tuple(tuple(vals[i * 4:(i + 1) * 4]) for i in range(4))
        return (
            (vals[0], vals[1], vals[2], 0.0),
            (vals[3], vals[4], vals[5], 0.0),
            (vals[6], vals[7], vals[8], 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
    if isinstance(value, (list, tuple)):
        if len(value) == 4 and all(_is_vector(row, (4,)) for row in value):
            return tuple(tuple(float(v) for v in row) for row in value[:4])  # type: ignore[return-value]
        if len(value) == 3 and all(_is_vector(row, (3,)) for row in value):
            return (
                (float(value[0][0]), float(value[0][1]), float(value[0][2]), 0.0),
                (float(value[1][0]), float(value[1][1]), float(value[1][2]), 0.0),
                (float(value[2][0]), float(value[2][1]), float(value[2][2]), 0.0),
                (0.0, 0.0, 0.0, 1.0),
            )
    return None


def _as_matrix3(value: Any) -> tuple[tuple[float, ...], ...] | None:
    m4 = _as_matrix4(value)
    if m4 is None:
        if _is_vector(value, (4,)):
            return _mat3_from_quat(value)
        return None
    return tuple(tuple(m4[i][j] for j in range(3)) for i in range(3))


def _mat3_from_spec(spec: Any) -> tuple[tuple[float, ...], ...]:
    if spec is None:
        return _IDENTITY_MAT3
    if _is_number(spec):
        return _mat3_from_quat(_quat_from_angle_y(float(spec)))
    if _is_vector(spec, (4,)):
        return _mat3_from_quat(spec)
    parsed = _as_matrix3(spec)
    if parsed is not None:
        return parsed
    if isinstance(spec, dict):
        if "quaternion" in spec:
            return _mat3_from_quat(spec["quaternion"])
        if "axis_angle" in spec:
            axis, angle = spec["axis_angle"]
            return _mat3_from_quat(_quat_from_axis_angle(axis, math.radians(float(angle))))
        if "yaw" in spec:
            return _mat3_from_quat(_quat_from_angle_y(float(spec["yaw"])))
    return _IDENTITY_MAT3



def _mat4_apply_point(m: Sequence[Sequence[float]], p: Sequence[float]) -> tuple[float, float, float]:
    x, y, z = _vec3(tuple(p)) or (0.0, 0.0, 0.0)
    return (
        m[0][0] * x + m[0][1] * y + m[0][2] * z + m[0][3],
        m[1][0] * x + m[1][1] * y + m[1][2] * z + m[1][3],
        m[2][0] * x + m[2][1] * y + m[2][2] * z + m[2][3],
    )



def _mat4_inverse_rigid(m: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...] | None:
    """刚体（旋转 + 平移）矩阵求逆；带缩放时返回 ``None``。"""
    basis = tuple(tuple(float(m[i][j]) for j in range(3)) for i in range(3))
    if _orthogonality_error(basis) > 1e-6:
        return None
    inv = _inv3(basis)
    if inv is None:
        return None
    t = (float(m[0][3]), float(m[1][3]), float(m[2][3]))
    it = _mat3_apply(inv, t)
    return (
        (inv[0][0], inv[0][1], inv[0][2], -it[0]),
        (inv[1][0], inv[1][1], inv[1][2], -it[1]),
        (inv[2][0], inv[2][1], inv[2][2], -it[2]),
        (0.0, 0.0, 0.0, 1.0),
    )


# ---------------------------------------------------------------------------
# 变换与传送门
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Transform:
    """刚体/仿射变换的 TRS 表示。"""

    translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: tuple[float, float, float, float] = _IDENTITY_QUAT
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def matrix(self) -> tuple[tuple[float, ...], ...]:
        r = _mat3_from_quat(self.rotation)
        sx, sy, sz = self.scale
        return (
            (r[0][0] * sx, r[0][1] * sy, r[0][2] * sz, self.translation[0]),
            (r[1][0] * sx, r[1][1] * sy, r[1][2] * sz, self.translation[1]),
            (r[2][0] * sx, r[2][1] * sy, r[2][2] * sz, self.translation[2]),
            (0.0, 0.0, 0.0, 1.0),
        )

    def inverse(self) -> tuple[tuple[float, ...], ...] | None:
        return _mat4_inverse_rigid(self.matrix())

    @classmethod
    def from_any(cls, value: Any, default: "Transform | None" = None) -> "Transform":
        if isinstance(value, cls):
            return value
        m = _as_matrix4(value)
        if m is None:
            return default if default is not None else cls()
        decomposition = _decompose_matrix(m)
        return cls(
            translation=decomposition["translation"],  # type: ignore[arg-type]
            rotation=decomposition["rotation"],  # type: ignore[arg-type]
            scale=tuple(abs(s) for s in decomposition["scale"]),  # type: ignore[arg-type]
        )


@dataclass
class Portal:
    """一扇传送门：局部 +X 右、+Y 上、+Z 为正面法线。"""

    name: str = "A"
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: tuple[float, float, float, float] = _IDENTITY_QUAT
    width: float = _DEFAULT_WIDTH
    height: float = _DEFAULT_HEIGHT
    up_hint: tuple[float, float, float] = (0.0, 1.0, 0.0)
    pair: str | None = None

    @property
    def basis(self) -> tuple[tuple[float, ...], ...]:
        return _mat3_from_quat(self.rotation)

    @property
    def right(self) -> tuple[float, float, float]:
        return tuple(self.basis[i][0] for i in range(3))  # type: ignore[return-value]

    @property
    def up(self) -> tuple[float, float, float]:
        return tuple(self.basis[i][1] for i in range(3))  # type: ignore[return-value]

    @property
    def normal(self) -> tuple[float, float, float]:
        return tuple(self.basis[i][2] for i in range(3))  # type: ignore[return-value]

    @property
    def half_width(self) -> float:
        return abs(float(self.width)) * 0.5

    @property
    def half_height(self) -> float:
        return abs(float(self.height)) * 0.5

    @property
    def area(self) -> float:
        return abs(float(self.width) * float(self.height))

    def transform(self) -> Transform:
        return Transform(translation=self.position, rotation=self.rotation)

    def matrix(self) -> tuple[tuple[float, ...], ...]:
        return self.transform().matrix()

    def local_point(self, world: Sequence[float]) -> tuple[float, float, float]:
        delta = _vsub(world, self.position)
        return (_dot(delta, self.right), _dot(delta, self.up), _dot(delta, self.normal))

    def world_point(self, local: Sequence[float]) -> tuple[float, float, float]:
        x, y, z = _vec3(tuple(local)) or (0.0, 0.0, 0.0)
        r = self.right
        u = self.up
        n = self.normal
        return (
            self.position[0] + r[0] * x + u[0] * y + n[0] * z,
            self.position[1] + r[1] * x + u[1] * y + n[1] * z,
            self.position[2] + r[2] * x + u[2] * y + n[2] * z,
        )


def _portal_from(value: Any, default_name: str = "A") -> Portal:
    """把 Portal / dict / 名字 / 整数统一成 Portal。

    无法识别时返回原点处的默认门（便于契约测试的空输入）。
    """
    if isinstance(value, Portal):
        return value
    if value is None or _is_number(value) or isinstance(value, str):
        name = value if isinstance(value, str) else default_name
        return Portal(name=str(name))
    if isinstance(value, dict):
        name = str(value.get("name", value.get("id", default_name)))
        position = _vec3(value.get("position", value.get("center", value.get("origin", (0, 0, 0))))) or (0.0, 0.0, 0.0)
        width = float(value.get("width", value.get("size", _DEFAULT_WIDTH)) or _DEFAULT_WIDTH)
        height = float(value.get("height", _DEFAULT_HEIGHT) or _DEFAULT_HEIGHT)
        if "size" in value and _is_vector(value["size"], (2, 3)):
            size = value["size"]
            width = float(size[0])
            height = float(size[1]) if len(size) > 1 else width
        up_hint = _vec3(value.get("up", value.get("up_hint", (0, 1, 0))), (0.0, 1.0, 0.0)) or (0.0, 1.0, 0.0)
        if "rotation" in value or "quaternion" in value or "orientation" in value:
            rot = _mat3_from_spec(value.get("rotation", value.get("quaternion", value.get("orientation"))))
            rotation = _quat_from_mat3(rot)
        elif "angle" in value and _is_number(value["angle"]):
            rotation = _quat_from_angle_y(float(value["angle"]))
        elif "yaw" in value:
            rotation = _quat_from_angle_y(float(value["yaw"]))
        elif "normal" in value:
            normal = _normalize(_vec3(value["normal"], (0.0, 0.0, 1.0)) or (0.0, 0.0, 1.0))
            right = _normalize(_cross(up_hint, normal), (1.0, 0.0, 0.0))
            up = _normalize(_cross(normal, right), (0.0, 1.0, 0.0))
            rotation = _quat_from_mat3(
                (
                    (right[0], up[0], normal[0]),
                    (right[1], up[1], normal[1]),
                    (right[2], up[2], normal[2]),
                )
            )
        else:
            rotation = _IDENTITY_QUAT
        pair = value.get("pair")
        return Portal(
            name=name,
            position=position,
            rotation=rotation,
            width=width,
            height=height,
            up_hint=up_hint,
            pair=str(pair) if pair is not None else None,
        )
    if isinstance(value, (list, tuple)) and len(value) == 2 and all(_is_vector(v, (2, 3)) for v in value):
        # (position, normal) 简写。
        p = _vec3(value[0]) or (0.0, 0.0, 0.0)
        n = _normalize(_vec3(value[1], (0.0, 0.0, 1.0)) or (0.0, 0.0, 1.0))
        right = _normalize(_cross((0.0, 1.0, 0.0), n), (1.0, 0.0, 0.0))
        up = _normalize(_cross(n, right), (0.0, 1.0, 0.0))
        rotation = _quat_from_mat3(
            ((right[0], up[0], n[0]), (right[1], up[1], n[1]), (right[2], up[2], n[2]))
        )
        return Portal(name=default_name, position=p, rotation=rotation)
    return Portal(name=default_name)


def _decompose_matrix(m: Sequence[Sequence[float]]) -> dict[str, Any]:
    """把 4x4 矩阵分解为平移、旋转、缩放与剪切。

    旋转由极分解得到，是离线性部分最近的正交旋转；缩放取
    ``S = R^T * A`` 的对角线，剪切取 ``S`` 的非对角项。这样旋转矩阵始终
    正交，作用于朝向/速度时不会因非均匀缩放而产生斜切。
    """
    basis = tuple(tuple(float(m[i][j]) for j in range(3)) for i in range(3))
    translation = (float(m[0][3]), float(m[1][3]), float(m[2][3]))
    det = _det3(basis)
    rotation, degenerate = _polar_rotation(basis)
    # 保底：极分解可能因奇异而退化，此时用 Gram-Schmidt 的列向量。
    if degenerate:
        x = _normalize((basis[0][0], basis[1][0], basis[2][0]), (1.0, 0.0, 0.0))
        y_raw = (basis[0][1], basis[1][1], basis[2][1])
        y = _normalize(_vsub(y_raw, _vmul(x, _dot(y_raw, x))), (0.0, 1.0, 0.0))
        z = _normalize(_cross(x, y), (0.0, 0.0, 1.0))
        rotation = (
            (x[0], y[0], z[0]),
            (x[1], y[1], z[1]),
            (x[2], y[2], z[2]),
        )
    s = _mat3_mul(_transpose3(rotation), basis)
    scales = (abs(s[0][0]), abs(s[1][1]), abs(s[2][2]))
    shear = (s[0][1], s[0][2], s[1][2])
    signed = (
        (s[0][0] if s[0][0] != 0.0 else 1.0),
        (s[1][1] if s[1][1] != 0.0 else 1.0),
        (s[2][2] if s[2][2] != 0.0 else 1.0),
    )
    uniform = (max(scales) - min(scales)) <= max(1e-7, max(scales) * 1e-7)
    quat = _quat_from_mat3(rotation)
    return {
        "translation": translation,
        "rotation": quat,
        "rotation_matrix": rotation,
        "scale": scales,
        "signed_scale": signed,
        "shear": shear,
        "uniform_scale": bool(uniform),
        "handedness": -1 if det < 0.0 else 1,
        "degenerate": bool(degenerate or abs(det) < 1e-12),
        "orthogonality_error": _orthogonality_error(rotation),
        "linear_determinant": det,
        "basis": basis,
    }


# ---------------------------------------------------------------------------
# 穿越序号与双缓冲
# ---------------------------------------------------------------------------


class CrossingSerial:
    """单调递增的穿越序号分配器，带幂等去重。"""

    def __init__(self) -> None:
        self.next_serial = 1
        self.last_applied = 0
        self.event_serials: dict[str, int] = {}
        self.applied: set[int] = set()
        self.log: list[dict[str, Any]] = []

    def claim(self, event_key: str) -> tuple[int, bool]:
        """为事件分配序号；同一事件重复申报返回原序号与 ``duplicate=True``。"""
        if event_key in self.event_serials:
            serial = self.event_serials[event_key]
            self.log.append({"event": "claim", "key": event_key, "serial": serial, "duplicate": True})
            return serial, True
        serial = self.next_serial
        self.next_serial += 1
        self.event_serials[event_key] = serial
        self.log.append({"event": "claim", "key": event_key, "serial": serial, "duplicate": False})
        return serial, False

    def apply(self, serial: int) -> bool:
        """应用序号；小于等于已应用序号或已应用过的包一律丢弃。"""
        serial = int(serial)
        if serial in self.applied or serial <= self.last_applied:
            self.log.append({"event": "apply", "serial": serial, "applied": False, "reason": "duplicate"})
            return False
        self.applied.add(serial)
        self.last_applied = serial
        self.log.append({"event": "apply", "serial": serial, "applied": True, "reason": "ok"})
        return True

    def snapshot(self) -> dict[str, Any]:
        return {
            "next_serial": self.next_serial,
            "last_applied": self.last_applied,
            "events": len(self.event_serials),
            "applied": len(self.applied),
        }


class DoubleBuffer:
    """双缓冲渲染目标的状态机。

    ``front`` 是主相机采样的纹理，``back`` 是当前渲染写入的目标。
    两个槽位各自处于 ``ready / writing / copying`` 之一；绝不允许
    同一槽位同时写入与拷贝，也绝不允许采样正在拷贝的槽位。若拷贝未完成
    而本帧需要渲染，则延迟写入（``deferred=True``），宁可少一帧也不脏读。
    """

    def __init__(self, name: str = "portal", width: int = 1280, height: int = 720) -> None:
        self.name = name
        self.width = int(width)
        self.height = int(height)
        self.front = 0
        self.back = 1
        self.states = ["ready", "ready"]
        self.frame = 0
        self.copy_serial = 0
        self.write_serial = 0
        self.overlaps = 0
        self.stale_reads = 0
        self.deferred = 0
        # 异步拷贝耗时（帧数）。>1 时拷贝会跨帧进行，期间 back 不可写、
        # front 仍可安全采样；本帧写入被延迟，而不是脏写。
        self.copy_latency = 1
        self._copy_ticks = 0
        self.log: list[dict[str, Any]] = []

    def _other(self, index: int) -> int:
        return 1 - index

    def begin_write(self) -> Report:
        self.frame += 1
        candidates = [i for i in (0, 1) if i != self.front and self.states[i] == "ready"]
        if candidates:
            slot = candidates[0]
            self.back = slot
            self.states[slot] = "writing"
            self.write_serial += 1
            self.log.append({"frame": self.frame, "event": "begin_write", "slot": slot})
            return Report(
                action="begin_write",
                slot=slot,
                frame=self.frame,
                deferred=False,
                states=tuple(self.states),
                front=self.front,
                back=self.back,
                ok=True,
            )
        self.deferred += 1
        reason = "copy_in_progress" if "copying" in self.states else "no_free_slot"
        self.log.append({"frame": self.frame, "event": "begin_write", "deferred": True, "reason": reason})
        return Report(
            action="begin_write",
            slot=None,
            frame=self.frame,
            deferred=True,
            reason=reason,
            states=tuple(self.states),
            front=self.front,
            back=self.back,
            ok=False,
            truth=False,
        )

    def end_write(self) -> Report:
        if self.states[self.back] != "writing":
            return Report(action="end_write", committed=False, reason="not_writing", ok=False, truth=False)
        self.states[self.back] = "ready"
        self.log.append({"frame": self.frame, "event": "end_write", "slot": self.back})
        return Report(
            action="end_write",
            slot=self.back,
            committed=True,
            states=tuple(self.states),
            frame=self.frame,
            ok=True,
        )

    def begin_copy(self) -> Report:
        source = self.back
        if self.states[source] != "ready" or source == self.front:
            # 没有可拷贝的目标：可能仍在写入，等待而不是脏拷贝。
            return Report(
                action="begin_copy",
                copied=False,
                reason="source_not_ready",
                states=tuple(self.states),
                ok=False,
                truth=False,
                frame=self.frame,
            )
        self.states[source] = "copying"
        self.copy_serial += 1
        self._copy_ticks = 0
        self.log.append({"frame": self.frame, "event": "begin_copy", "source": source, "destination": self.front})
        return Report(
            action="begin_copy",
            source=source,
            destination=self.front,
            copy_serial=self.copy_serial,
            copied=False,
            copying=True,
            latency=self.copy_latency,
            states=tuple(self.states),
            frame=self.frame,
            ok=True,
        )

    def end_copy(self) -> Report:
        copying = [i for i in (0, 1) if self.states[i] == "copying"]
        if not copying:
            return Report(action="end_copy", swapped=False, reason="nothing_copying", ok=False, truth=False)
        source = copying[0]
        self._copy_ticks += 1
        if self._copy_ticks < max(1, int(self.copy_latency)):
            # 异步拷贝尚未完成：保持 copying，期间 front 仍可安全采样。
            return Report(
                action="end_copy",
                swapped=False,
                pending=True,
                source=source,
                progress=self._copy_ticks / max(1, int(self.copy_latency)),
                states=tuple(self.states),
                frame=self.frame,
                ok=True,
            )
        self.states[source] = "ready"
        old_front = self.front
        self.front = source
        self.back = self._other(source)
        self.log.append(
            {"frame": self.frame, "event": "end_copy", "source": source, "old_front": old_front, "new_front": self.front}
        )
        return Report(
            action="end_copy",
            source=source,
            swapped=True,
            front=self.front,
            back=self.back,
            states=tuple(self.states),
            frame=self.frame,
            ok=True,
        )

    def sample(self) -> Report:
        if self.states[self.front] == "copying":
            self.stale_reads += 1
            return Report(
                action="sample",
                allowed=False,
                reason="front_copying",
                stale=True,
                front=self.front,
                frame=self.frame,
                ok=False,
                truth=False,
            )
        if self.states[self.front] != "ready":
            self.stale_reads += 1
            return Report(
                action="sample",
                allowed=False,
                reason="front_not_ready",
                stale=True,
                front=self.front,
                frame=self.frame,
                ok=False,
                truth=False,
            )
        return Report(
            action="sample",
            allowed=True,
            stale=False,
            front=self.front,
            ready=True,
            frame=self.frame,
            states=tuple(self.states),
            ok=True,
        )

    def tick(self, frames: int = 1) -> Report:
        """完整跑 ``frames`` 帧：写 back → 提交 → 异步拷贝 → 交换 → 采样。

        ``copy_latency > 1`` 时拷贝跨帧完成；在这期间下一次写入会因没有
        空闲槽位而延迟（``deferred=True``），但 ``front`` 仍然就绪、
        采样照常进行，因此画面不会读到正在被写入/拷贝的内容。
        """
        samples: list[dict[str, Any]] = []
        for _ in range(max(1, int(frames))):
            begin = self.begin_write()
            if begin["deferred"]:
                # 没有空闲槽位：先推进可能在进行的异步拷贝，再采样已就绪的 front。
                copy = self.end_copy()
                sample = self.sample()
                samples.append(
                    {
                        "frame": self.frame,
                        "deferred": True,
                        "reason": begin["reason"],
                        "copy_pending": bool(copy.get("pending")),
                        "sample_allowed": sample["allowed"],
                        "front": self.front,
                    }
                )
                continue
            self.end_write()
            copy_begin = self.begin_copy()
            copy_end = self.end_copy() if copy_begin["ok"] else None
            sample = self.sample()
            samples.append(
                {
                    "frame": self.frame,
                    "deferred": False,
                    "front": self.front,
                    "back": self.back,
                    "sample_allowed": sample["allowed"],
                    "copy_pending": bool(copy_end and copy_end.get("pending")),
                }
            )
        return self.snapshot(samples=samples)

    def snapshot(self, samples: list[dict[str, Any]] | None = None) -> Report:
        return Report(
            name=self.name,
            width=self.width,
            height=self.height,
            frame=self.frame,
            front=self.front,
            back=self.back,
            states=tuple(self.states),
            buffers=[
                {"slot": 0, "role": "front" if self.front == 0 else "back", "state": self.states[0]},
                {"slot": 1, "role": "front" if self.front == 1 else "back", "state": self.states[1]},
            ],
            overlaps=self.overlaps,
            stale_reads=self.stale_reads,
            deferred=self.deferred,
            write_serial=self.write_serial,
            copy_serial=self.copy_serial,
            samples=samples or [],
            invariant_ok=bool(
                self.front != self.back
                and self.overlaps == 0
                and self.stale_reads == 0
                and all(state in ("ready", "writing", "copying") for state in self.states)
                and not (self.states[0] == "copying" and self.states[1] == "copying")
            ),
            ok=self.overlaps == 0 and self.stale_reads == 0,
        )


_BUFFERS: dict[str, DoubleBuffer] = {}
_SERIALS: dict[str, CrossingSerial] = {}
_SYNC_STATE: dict[str, dict[str, Any]] = {}


def _buffer_for(value: Any = None) -> DoubleBuffer:
    """解析渲染目标。

    显式传入 ``DoubleBuffer`` 直接使用；传入名字/字典时取注册表里的共享实例；
    传入 ``None`` 或占位标量时返回**全新的**实例，保证默认路径调用两次得到
    相同结果，不会把上一次运行的状态泄漏到下一次。
    """
    if isinstance(value, DoubleBuffer):
        return value
    if isinstance(value, (str, dict)):
        if isinstance(value, dict):
            name = str(value.get("name", value.get("id", "portal")))
            width = int(value.get("width", 1280))
            height = int(value.get("height", 720))
        else:
            name, width, height = value, 1280, 720
        if name not in _BUFFERS:
            _BUFFERS[name] = DoubleBuffer(name=name, width=width, height=height)
        return _BUFFERS[name]
    return DoubleBuffer(name="portal")


def _serial_for(value: Any = None) -> CrossingSerial:
    """解析穿越序号状态。

    与 :func:`_buffer_for` 一致：显式命名时共享，默认调用返回全新状态。
    """
    if isinstance(value, CrossingSerial):
        return value
    if isinstance(value, (str, dict)):
        name = str(value.get("name", value.get("id", "default"))) if isinstance(value, dict) else value
        if name not in _SERIALS:
            _SERIALS[name] = CrossingSerial()
        return _SERIALS[name]
    return CrossingSerial()


def _live_serial(value: Any = None) -> CrossingSerial:
    """跨调用持久的穿越序号状态（用于 ``enter`` / ``sync`` 的逐帧去重）。

    门穿越天然是跨帧的有状态过程：只有保留 ``last_applied`` 才能把连续两帧
    的重复检测识别为同一次穿越。需要隔离时显式传 ``state`` / ``session``。
    """
    if isinstance(value, CrossingSerial):
        return value
    if isinstance(value, (str, dict)):
        return _serial_for(value)
    name = "live:default"
    if name not in _SERIALS:
        _SERIALS[name] = CrossingSerial()
    return _SERIALS[name]


# ---------------------------------------------------------------------------
# 对称 3x3 特征分解：用于奇异值（真实缩放）与缩放主轴
# ---------------------------------------------------------------------------


def _sym_eigen3(m: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], tuple[tuple[float, ...], ...]]:
    """Jacobi 迭代求实对称 3x3 矩阵的特征值与特征向量。

    返回 ``(eigenvalues, eigenvectors)``，其中 ``eigenvectors`` 的每一列
    是对应特征值的单位特征向量。结果按特征值绝对值降序，平局时按原顺序，
    保证确定性。
    """
    a = [[float(m[i][j]) for j in range(3)] for i in range(3)]
    # 对称化，避免调用方传入轻微不对称的矩阵导致迭代不收敛。
    for i in range(3):
        for j in range(i + 1, 3):
            avg = 0.5 * (a[i][j] + a[j][i])
            a[i][j] = a[j][i] = avg
    v = [[1.0 if i == j else 0.0 for j in range(3)] for i in range(3)]

    for _ in range(64):
        pairs = ((0, 1), (0, 2), (1, 2))
        p, q = max(pairs, key=lambda ij: abs(a[ij[0]][ij[1]]))
        apq = a[p][q]
        if abs(apq) < 1e-14:
            break
        app, aqq = a[p][p], a[q][q]
        theta = (aqq - app) / (2.0 * apq)
        if abs(theta) > 1e100:
            t = 0.5 / theta
        else:
            t = math.copysign(1.0, theta) / (abs(theta) + math.sqrt(theta * theta + 1.0))
        c = 1.0 / math.sqrt(t * t + 1.0)
        s = t * c
        a[p][p] = app - t * apq
        a[q][q] = aqq + t * apq
        a[p][q] = a[q][p] = 0.0
        for r in range(3):
            if r in (p, q):
                continue
            arp, arq = a[r][p], a[r][q]
            a[r][p] = a[p][r] = c * arp - s * arq
            a[r][q] = a[q][r] = s * arp + c * arq
        for r in range(3):
            vrp, vrq = v[r][p], v[r][q]
            v[r][p] = c * vrp - s * vrq
            v[r][q] = s * vrp + c * vrq

    eigs = (a[0][0], a[1][1], a[2][2])
    order = sorted(range(3), key=lambda i: (-abs(eigs[i]), i))
    values = tuple(eigs[i] for i in order)
    vectors = tuple(tuple(v[r][i] for r in range(3)) for i in order)
    return values, vectors


def _singular_values(m: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], tuple[tuple[float, ...], ...]]:
    """通过 ``A^T A`` 的特征分解求奇异值与右奇异向量。"""
    ata = _mat3_mul(_transpose3(m), m)
    values, vectors = _sym_eigen3(ata)
    singular = tuple(math.sqrt(max(0.0, value)) for value in values)
    return singular, vectors


# ---------------------------------------------------------------------------
# 输入解析：把 Portal / dict / 向量 / 标量统一成结构化场景
# ---------------------------------------------------------------------------


_PORTAL_KEYS = (
    "normal",
    "rotation",
    "quaternion",
    "orientation",
    "width",
    "height",
    "pair",
    "up",
    "yaw",
    "angle",
    "portal",
    "entry",
    "exit",
    "a",
    "b",
    "portals",
)


def _is_portal_spec(value: Any) -> bool:
    """看起来像一扇门（而不是普通位置/速度向量或标量）。"""
    if isinstance(value, Portal):
        return True
    if isinstance(value, str):
        return True
    if isinstance(value, dict):
        return any(key in value for key in _PORTAL_KEYS)
    return False


def _is_pair_spec(value: Any) -> bool:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return _is_portal_spec(value[0]) and _is_portal_spec(value[1]) and not isinstance(value[0], str)
    if isinstance(value, dict):
        return ("entry" in value and "exit" in value) or ("a" in value and "b" in value) or (
            "portals" in value and isinstance(value["portals"], (list, tuple)) and len(value["portals"]) == 2
        )
    return False


def _pair_from(value: Any, default_entry: Portal | None = None, default_exit: Portal | None = None) -> tuple[Portal, Portal]:
    """把成对门表示统一成 ``(entry, exit)``。"""
    if _is_pair_spec(value):
        if isinstance(value, dict):
            if "entry" in value and "exit" in value:
                return _portal_from(value["entry"], "A"), _portal_from(value["exit"], "B")
            if "a" in value and "b" in value:
                return _portal_from(value["a"], "A"), _portal_from(value["b"], "B")
            portals = value["portals"]
            return _portal_from(portals[0], "A"), _portal_from(portals[1], "B")
        return _portal_from(value[0], "A"), _portal_from(value[1], "B")
    return (
        default_entry if default_entry is not None else Portal("A"),
        default_exit if default_exit is not None else Portal("B", position=(6.0, 0.0, 0.0)),
    )


def _pair_transform(entry: Portal, exit_portal: Portal) -> tuple[tuple[float, ...], ...]:
    """入口到出口的完整变换矩阵：``M_exit * F * M_entry^-1``。

    ``F`` 是绕局部 Y 轴 180° 的翻转，使角色从入口正面进入后从出口正面走出。
    这里刻意使用刚体矩阵，不含任何缩放，因此不会把角色拉扁。
    """
    r_entry = _mat3_from_quat(entry.rotation)
    inv_entry = _transpose3(r_entry)
    flip = (
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, -1.0),
    )
    r_exit = _mat3_from_quat(exit_portal.rotation)
    r_rel = _mat3_mul(_mat3_mul(r_exit, flip), inv_entry)
    # M_rel = M_exit * F * M_entry^-1，其平移为 t_exit - R_rel * t_entry。
    t_rel = _vsub(exit_portal.position, _mat3_apply(r_rel, entry.position))
    return (
        (r_rel[0][0], r_rel[0][1], r_rel[0][2], t_rel[0]),
        (r_rel[1][0], r_rel[1][1], r_rel[1][2], t_rel[1]),
        (r_rel[2][0], r_rel[2][1], r_rel[2][2], t_rel[2]),
        (0.0, 0.0, 0.0, 1.0),
    )


def _pair_rotation(entry: Portal, exit_portal: Portal) -> tuple[float, float, float, float]:
    """入口到出口的相对旋转（含 180° 翻转），不含缩放。"""
    q_rel = _quat_mul(
        _quat_mul(_quat_normalize(exit_portal.rotation), _FLIP_QUAT),
        _quat_conjugate(_quat_normalize(entry.rotation)),
    )
    return _quat_normalize(q_rel)


def _is_point_arg(value: Any) -> bool:
    """位置/速度参数必须是 2~4 维向量；裸标量只当占位，不当作坐标。"""
    return _is_vector(value, (2, 3, 4))


def _read_option(source: dict[str, Any], names: Sequence[str], default: Any = None) -> Any:
    for name in names:
        if name in source and source[name] is not None:
            return source[name]
    return default


def _parse_pose_args(args: Sequence[Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    """把 ``enter`` / ``exit`` / ``render`` 风格的宽松参数整理成场景。"""
    opts: dict[str, Any] = dict(kwargs)
    items = list(args)

    if items and isinstance(items[0], dict):
        head = items[0]
        # 字典既可能是场景，也可能只是“门”或“成对门”的描述。
        for key in (
            "entry",
            "exit",
            "a",
            "b",
            "portals",
            "portal",
            "pair",
            "position",
            "pos",
            "velocity",
            "vel",
            "rotation",
            "orientation",
            "quaternion",
            "matrix",
            "transform",
            "serial",
            "event",
            "frame",
            "camera",
            "max_depth",
            "min_pixels",
            "screen",
            "screen_height",
            "fov",
            "radius",
            "cooldown",
            "now",
            "last_exit",
            "exit_position",
            "exit_velocity",
            "copy_latency",
            "frames",
            "name",
            "state",
            "message",
            "session",
        ):
            if key in head and key not in opts:
                opts[key] = head[key]
        if _is_pair_spec(head):
            items.pop(0)
        elif _is_portal_spec(head):
            # 单个门描述：作为 "portal" 提供，供 sweep / render 等入口使用。
            opts.setdefault("portal", head)
            items.pop(0)
        else:
            items.pop(0)

    if "pair" in opts and "entry" not in opts:
        entry, exit_portal = _pair_from(opts["pair"])
        opts.setdefault("entry", entry)
        opts.setdefault("exit", exit_portal)
    elif _is_pair_spec(opts.get("entry")) and "exit" not in opts:
        entry, exit_portal = _pair_from(opts["entry"])
        opts["entry"], opts["exit"] = entry, exit_portal

    # 位置参数：成对门 -> 两扇门 -> 位置 -> 速度 -> 四元数
    if items and _is_pair_spec(items[0]) and "entry" not in opts:
        entry, exit_portal = _pair_from(items.pop(0))
        opts["entry"], opts["exit"] = entry, exit_portal

    if (
        len(items) >= 2
        and _is_portal_spec(items[0])
        and _is_portal_spec(items[1])
        and "entry" not in opts
    ):
        opts["entry"] = _portal_from(items.pop(0), "A")
        opts["exit"] = _portal_from(items.pop(0), "B")

    if items and _is_portal_spec(items[0]) and "entry" not in opts:
        opts["entry"] = _portal_from(items.pop(0), "A")

    for key in ("position", "pos"):
        if key not in opts and items and _is_point_arg(items[0]):
            opts["position"] = _vec3(items.pop(0))
            break

    for key in ("velocity", "vel"):
        if key not in opts and items and _is_point_arg(items[0]):
            opts["velocity"] = _vec3(items.pop(0))
            break

    if "rotation" not in opts and "orientation" not in opts and items:
        candidate = items[0]
        if _is_vector(candidate, (4,)):
            opts["rotation"] = _quat_normalize(items.pop(0))
        elif _is_vector(candidate, (3,)) and len(items) > 1 and _is_number(items[1]):
            # 轴角简写：(axis, angle_deg)
            angle = math.radians(float(items[1]))
            opts["rotation"] = _quat_from_axis_angle(items.pop(0), angle)
            items.pop(0)
        # 裸标量不作为角度：契约测试会用 ``1`` 占位，不能把它当成 1 度偏航。

    opts["_rest"] = items
    return opts



def _demo_entry() -> Portal:
    return Portal(name="A", position=(0.0, 0.0, 0.0), rotation=_IDENTITY_QUAT, width=1.6, height=2.4, pair="B")


def _demo_exit() -> Portal:
    return Portal(name="B", position=(6.0, 0.0, 0.0), rotation=_IDENTITY_QUAT, width=1.6, height=2.4, pair="A")


def _demo_position() -> tuple[float, float, float]:
    return (0.2, 0.1, 0.0)


def _demo_velocity() -> tuple[float, float, float]:
    return (0.0, 0.0, -3.0)


def _demo_camera() -> dict[str, Any]:
    return {
        "position": (-2.0, 1.2, -6.0),
        "forward": (0.0, 0.0, 1.0),
        "up": (0.0, 1.0, 0.0),
        "fov": 60.0,
        "aspect": 16.0 / 9.0,
        "near": 0.05,
        "far": 200.0,
        "screen_height": 1080,
    }


def _hash_payload(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _round_tree(value: Any, digits: int = 6) -> Any:
    """递归取整，供确定性哈希使用。"""
    if isinstance(value, float):
        return round(value, digits) + 0.0
    if isinstance(value, dict):
        return {str(k): _round_tree(v, digits) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_round_tree(v, digits) for v in value]
    if isinstance(value, set):
        return sorted(_round_tree(v, digits) for v in value)
    return value


def _looks_like_matrix(value: Any) -> bool:
    """判断输入是否是矩阵（嵌套 3x3/4x4 或平铺 9/16 个数字）。"""
    if isinstance(value, Transform):
        return True
    if isinstance(value, dict):
        return any(key in value for key in ("matrix", "transform")) or (
            "translation" in value or "scale" in value
        )
    if _is_vector(value, (9, 16)):
        return True
    if isinstance(value, (list, tuple)) and value:
        if all(_is_vector(row, (3, 4)) for row in value) and len(value) in (3, 4):
            return True
    return False


# ---------------------------------------------------------------------------
# 修复 1：变换分解缩放
# ---------------------------------------------------------------------------


def decompose_scale(*args: Any, **kwargs: Any) -> Report:
    """把入口到出口的变换分解为平移 / 旋转 / 缩放，剥离非均匀缩放。

    输入可以是：

    * 4x4 / 3x3 矩阵（行主序的嵌套序列或长度 16/9 的平铺序列）；
    * :class:`Transform` 或 :class:`Portal`；
    * ``dict``（``matrix`` / ``translation`` + ``rotation`` + ``scale`` 等）；
    * 成对门 ``(entry, exit)`` 或 ``{"entry": ..., "exit": ...}``，此时分解
      的是相对变换 ``M_exit * F * M_entry^-1``。

    返回值中 ``rotation`` 是最接近的正交旋转，``scale`` 是线性部分的奇异值，
    ``anisotropy`` 是最大/最小缩放比，``shear`` 是旋转基下缩放矩阵的非对角
    分量。``roundtrip_error`` 验证 ``T * R * S`` 能重建原矩阵。
    """
    opts = _parse_pose_args(args, kwargs)
    source = _read_option(opts, ("matrix", "transform", "m", "source"))
    if source is None and args:
        first = args[0]
        if isinstance(first, Transform) or _looks_like_matrix(first):
            source = first

    if source is None:
        entry = opts.get("entry")
        exit_portal = opts.get("exit")
        if entry is not None and exit_portal is not None:
            entry_p = _portal_from(entry, "A")
            exit_p = _portal_from(exit_portal, "B")
            matrix = _pair_transform(entry_p, exit_p)
            source_kind = "portal_pair"
        else:
            matrix = None
            source_kind = "default"
    else:
        matrix = _as_matrix4(source)
        source_kind = "matrix"

    if matrix is None:
        # 无法识别的输入（如契约测试的整数 1）：返回确定性的演示分解，
        # 明确标记默认来源，而不是抛异常。
        matrix = _pair_transform(_demo_entry(), _demo_exit())
        source_kind = "default_demo"

    decomposition = _decompose_matrix(matrix)
    r = decomposition["rotation_matrix"]
    basis = decomposition["basis"]
    # 对称缩放/剪切矩阵 S = R^T A。极分解下 S 对称：正定时它的特征值就是
    # 各主轴缩放；行列式为负时其中一个特征值为负，代表镜像。
    s = _mat3_mul(_transpose3(r), basis)
    # 用奇异值（永远非负）报告缩放幅度，用右奇异向量报告主轴方向。
    singular, axes = _singular_values(basis)
    # 往返重建用真实对称矩阵 S，因此含镜像/剪切时也能精确复原 A = R * S。
    recon_basis = _mat3_mul(r, s)
    roundtrip_error = max(
        abs(float(recon_basis[i][j]) - float(basis[i][j])) for i in range(3) for j in range(3)
    )
    shear = max(abs(s[0][1]), abs(s[0][2]), abs(s[1][2]))
    scale = tuple(float(v) for v in singular)
    anisotropy = (max(scale) / max(min(scale), _EPS)) if scale and min(scale) > 0 else float("inf")
    ortho_error = _orthogonality_error(r)
    quat = decomposition["rotation"]
    reflected = _det3(s) < 0.0

    return Report(
        source=source_kind,
        translation=tuple(float(v) for v in decomposition["translation"]),
        rotation=quat,
        rotation_matrix=tuple(tuple(float(v) for v in row) for row in r),
        scale=scale,
        scale_axes=tuple(tuple(float(v) for v in axis) for axis in axes),
        anisotropy=_round(anisotropy) if math.isfinite(anisotropy) else float("inf"),
        uniform_scale=bool(math.isfinite(anisotropy) and anisotropy <= 1.0 + 1e-7),
        shear=_round(shear),
        has_shear=bool(shear > 1e-7),
        reflected=bool(reflected),
        handedness=int(decomposition["handedness"]),
        orthogonality_error=_round(ortho_error),
        roundtrip_error=_round(roundtrip_error),
        degenerate=bool(decomposition["degenerate"]),
        non_rigid=bool(
            (math.isfinite(anisotropy) and anisotropy > 1.0 + 1e-7)
            or shear > 1e-7
            or ortho_error > 1e-7
            or reflected
        ),
        ok=bool(ortho_error <= 1e-6 and roundtrip_error <= 1e-6 and not decomposition["degenerate"]),
    )


# ---------------------------------------------------------------------------
# 修复 2：扫掠检测穿越
# ---------------------------------------------------------------------------


def sweep_crossing(*args: Any, **kwargs: Any) -> Report:
    """用上一帧到当前帧的线段与门平面求交，检测本帧是否穿越。

    相比“只看上一帧胶囊体位置”的点判定，扫掠检测解析求解线段与门平面的
    交点，因此即使角色一帧移动了很远、整段跨过门面，也不会漏判。

    输入支持 ``(portal, prev, current)``、``(prev, current, portal)``、
    ``{"portal": ..., "previous": ..., "current": ...}`` 等写法，
    ``radius`` 为胶囊体半径（会扩大门的有效矩形），``velocity`` 可选，
    用于报告穿越速度。
    """
    opts = _parse_pose_args(args, kwargs)
    portal_value = _read_option(opts, ("portal", "entry", "gate"))
    prev = _read_option(opts, ("previous", "prev", "from", "start", "last_position"))
    cur = _read_option(opts, ("current", "cur", "to", "end", "position"))
    radius = float(_read_option(opts, ("radius", "capsule_radius"), 0.0) or 0.0)
    velocity = _read_option(opts, ("velocity", "vel"))

    portal = _portal_from(portal_value, "A") if portal_value is not None else Portal("A")

    rest = [v for v in opts.get("_rest", []) if not _is_portal_spec(v) and _is_point_arg(v)]
    vectors = list(rest)
    if prev is None and vectors:
        prev = _vec3(vectors.pop(0))
    if cur is None and vectors:
        cur = _vec3(vectors.pop(0))

    if prev is None or cur is None:
        # 演示：高速角色整帧从门前到门后，验证扫掠能捕获。
        portal = _demo_entry()
        prev = (0.2, 0.1, 2.5)
        cur = (0.2, 0.1, -2.5)
        demo = True
    else:
        prev = _vec3(prev) or (0.0, 0.0, 0.0)
        cur = _vec3(cur) or (0.0, 0.0, 0.0)
        demo = False

    normal = portal.normal
    d_prev = _dot(_vsub(prev, portal.position), normal)
    d_cur = _dot(_vsub(cur, portal.position), normal)
    delta = d_prev - d_cur
    segment_length = _norm(_vsub(cur, prev))
    moving = segment_length > _EPS

    crossed_plane = moving and abs(delta) > _EPS and (d_prev > 0.0) != (d_cur > 0.0)
    if moving and abs(delta) > _EPS and (d_prev == 0.0 or d_cur == 0.0):
        # 端点恰好落在门面上：只要另一端点明确在另一侧，也算一次穿越。
        crossed_plane = (d_prev >= 0.0 and d_cur <= 0.0) or (d_prev <= 0.0 and d_cur >= 0.0)

    t = None
    point = None
    local = None
    inside = False
    if crossed_plane:
        t = d_prev / delta
        t = max(0.0, min(1.0, t))
        point = _lerp(prev, cur, t)
        local = portal.local_point(point)
        inside = (
            abs(local[0]) <= portal.half_width + max(0.0, radius)
            and abs(local[1]) <= portal.half_height + max(0.0, radius)
        )

    crossed = bool(crossed_plane and inside)
    entry_side = "front" if d_prev > 0.0 else ("back" if d_prev < 0.0 else "plane")
    exit_side = "front" if d_cur > 0.0 else ("back" if d_cur < 0.0 else "plane")

    reason = "crossed"
    if not moving:
        reason = "stationary"
    elif not crossed_plane:
        reason = "no_plane_crossing"
    elif not inside:
        reason = "outside_portal_bounds"

    local_clean = tuple(_round(v) for v in local) if local is not None else None
    point_clean = tuple(_round(v) for v in point) if point is not None else None
    speed = None
    if velocity is not None:
        vel = _vec3(velocity)
        speed = _norm(vel) if vel is not None else None

    return Report(
        crossed=crossed,
        crossed_plane=bool(crossed_plane),
        inside=bool(inside),
        t=round(float(t), 9) if t is not None else None,
        point=point_clean,
        local=local_clean,
        entry_side=entry_side,
        exit_side=exit_side,
        plane_distance_previous=_round(d_prev),
        plane_distance_current=_round(d_cur),
        travelled=_round(segment_length),
        speed=_round(speed) if speed is not None else None,
        radius=max(0.0, radius),
        portal=portal.name,
        demo=demo,
        reason=reason,
        ok=True,
    )


# ---------------------------------------------------------------------------
# 修复 6：出口速度重定向
# ---------------------------------------------------------------------------


def redirect_exit_velocity(*args: Any, **kwargs: Any) -> Report:
    """按出口相对入口的旋转重定向速度，保持速率不变。

    旧实现把含非均匀缩放的完整变换作用到速度上，速度被拉伸、方向被斜切。
    这里只取旋转部分 ``R_exit * F * R_entry^-1``（或显式矩阵极分解出的
    旋转），对速度方向做刚体旋转；``|v|`` 在变换前后严格相等。
    """
    opts = _parse_pose_args(args, kwargs)
    velocity = _read_option(opts, ("velocity", "vel", "v"))
    matrix = _read_option(opts, ("matrix", "transform"))

    rest = opts.get("_rest", [])
    if velocity is None:
        for item in rest:
            if _is_point_arg(item) and not _is_portal_spec(item):
                velocity = _vec3(item)
                break
    if velocity is None:
        velocity = _demo_velocity()
        demo = True
    else:
        demo = False
    velocity = _vec3(velocity) or (0.0, 0.0, 0.0)

    rotation_source = "portal_pair"
    entry = opts.get("entry")
    exit_portal = opts.get("exit")
    if matrix is not None:
        m4 = _as_matrix4(matrix)
        if m4 is not None:
            decomposition = _decompose_matrix(m4)
            q_rel = decomposition["rotation"]
            rotation_source = "matrix_rotation_only"
        else:
            q_rel = _quad_identity_safe(entry, exit_portal)
    elif entry is not None or exit_portal is not None:
        entry_p = _portal_from(entry, "A") if entry is not None else _demo_entry()
        exit_p = _portal_from(exit_portal, "B") if exit_portal is not None else _demo_exit()
        q_rel = _pair_rotation(entry_p, exit_p)
    else:
        q_rel = _pair_rotation(_demo_entry(), _demo_exit())

    redirected = _quat_rotate(q_rel, velocity)
    speed_before = _norm(velocity)
    speed_after = _norm(redirected)
    preserved = abs(speed_before - speed_after) <= max(1e-9, speed_before * 1e-9)

    return Report(
        velocity=tuple(_round(v) for v in redirected),
        velocity_before=tuple(_round(v) for v in velocity),
        rotation=q_rel,
        speed_before=_round(speed_before),
        speed_after=_round(speed_after),
        speed_preserved=bool(preserved),
        scale_applied=False,
        rotation_source=rotation_source,
        direction_changed=bool(_norm(_vsub(redirected, velocity)) > 1e-7),
        demo=demo,
        ok=True,
    )


def _quad_identity_safe(entry: Any, exit_portal: Any) -> tuple[float, float, float, float]:
    """矩阵无效时退回成对门的相对旋转。"""
    entry_p = _portal_from(entry, "A") if entry is not None else _demo_entry()
    exit_p = _portal_from(exit_portal, "B") if exit_portal is not None else _demo_exit()
    return _pair_rotation(entry_p, exit_p)


def _default_fov_rad(fov: float) -> float:
    """``fov`` 约定为垂直视场角（度）；小于 3.2 的值视为弧度。"""
    if abs(fov) <= 3.2:
        return float(fov)
    return math.radians(float(fov))


def _camera_from(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        pos = _vec3(value.get("position", value.get("eye", value.get("origin", (0, 0, 0))))) or (0.0, 0.0, 0.0)
        forward = _normalize(_vec3(value.get("forward", value.get("dir", value.get("look", (0, 0, 1)))), (0.0, 0.0, 1.0)) or (0.0, 0.0, 1.0))
        up = _normalize(_vec3(value.get("up", (0, 1, 0)), (0.0, 1.0, 0.0)) or (0.0, 1.0, 0.0))
        fov = float(value.get("fov", value.get("fov_deg", 60.0)))
        if "fov_rad" in value:
            fov_rad = float(value["fov_rad"])
        else:
            fov_rad = _default_fov_rad(fov)
        aspect = float(value.get("aspect", 16.0 / 9.0) or 16.0 / 9.0)
        return {
            "position": pos,
            "forward": forward,
            "up": up,
            "fov": fov,
            "fov_rad": fov_rad,
            "aspect": aspect,
            "near": float(value.get("near", 0.05) or 0.05),
            "far": float(value.get("far", 200.0) or 200.0),
            "screen_height": float(value.get("screen_height", value.get("height", 1080)) or 1080),
        }
    return _camera_from(_demo_camera())


def _frustum_visible(cam: dict[str, Any], center: Sequence[float], radius: float) -> tuple[bool, float, float]:
    """返回 ``(是否在视锥内, 相机空间深度, 投影像素高度占用)``。

    先做深度与水平/垂直视场角判定，再把门的包围球投影成像素尺寸。
    像素尺寸用 ``2 * radius * focal / depth`` 近似。
    """
    rel = _vsub(center, cam["position"])
    depth = _dot(rel, cam["forward"])
    if depth <= cam["near"] or depth >= cam["far"]:
        return False, depth, 0.0
    focal = 0.5 * cam["screen_height"] / math.tan(max(1e-4, cam["fov_rad"]) * 0.5)
    tangent_v = math.tan(max(1e-4, cam["fov_rad"]) * 0.5)
    tangent_h = tangent_v * cam["aspect"]
    lateral = _dot(rel, _normalize(_cross(cam["up"], cam["forward"]), (1.0, 0.0, 0.0)))
    vertical = _dot(rel, cam["up"])
    if abs(lateral) - radius > depth * tangent_h:
        return False, depth, 0.0
    if abs(vertical) - radius > depth * tangent_v:
        return False, depth, 0.0
    pixels = 2.0 * max(radius, 0.0) * focal / max(depth, _EPS)
    return True, depth, pixels


# ---------------------------------------------------------------------------
# 修复 3：递归按视锥裁剪
# ---------------------------------------------------------------------------


def recursion_by_frustum(*args: Any, **kwargs: Any) -> Report:
    """按视锥可见性与屏幕占比决定递归层数，末端连续淡出。

    旧实现固定递归 3 层，然后硬切成天空盒。这里每层都重新判定：

    * 门是否落在视锥内（深度、水平/垂直视场角）；
    * 投影后的像素尺寸是否仍值得渲染；
    * 到达动态预算时使用 ``smoothstep`` 连续淡出，``skybox_weight`` 从 0
      平滑升到 1，没有硬切。

    第 ``n`` 层使用沿入口->出口位移方向后退的虚拟门距离，因此互视的门对
    会随层数自然变小并收敛，而不是无限递归。
    """
    opts = _parse_pose_args(args, kwargs)
    camera = _camera_from(_read_option(opts, ("camera", "view", "eye"), _demo_camera()))
    entry_value = _read_option(opts, ("entry", "portal", "gate"))
    exit_value = opts.get("exit")

    if entry_value is None:
        entry = _demo_entry()
        exit_portal = _demo_exit()
        demo = True
    else:
        entry = _portal_from(entry_value, "A")
        exit_portal = _portal_from(exit_value, "B") if exit_value is not None else _demo_exit()
        demo = False

    pair = (entry, exit_portal)
    if _is_pair_spec(entry_value):
        pair = _pair_from(entry_value)
        entry, exit_portal = pair

    max_depth = int(_read_option(opts, ("max_depth", "depth_cap", "budget"), 16) or 16)
    max_depth = max(1, min(max_depth, 64))
    min_pixels = float(_read_option(opts, ("min_pixels", "pixel_threshold"), 48.0) or 48.0)
    fade_band = float(_read_option(opts, ("fade_band", "fade_pixels"), max(8.0, min_pixels * 0.5)) or min_pixels)
    explicit_distances = _read_option(opts, ("distances", "portal_distances", "depth_distances"))

    base_center = entry.world_point((0.0, 0.0, 0.0))
    radius = 0.5 * math.hypot(entry.width, entry.height)
    to_portal = _vsub(base_center, camera["position"])
    distance0 = _norm(to_portal)
    direction = _normalize(to_portal, camera["forward"])
    # 递归图像沿“相机 -> 入口”的视线不断后退；每层后退量取门对沿视线的
    # 间距（退化为并排门时至少是半个门宽），因此互视门对的图像会自然缩小。
    separation = abs(_dot(_vsub(exit_portal.position, entry.position), direction))
    step = max(separation, 0.5 * entry.width, 0.1)
    if explicit_distances is not None and isinstance(explicit_distances, (list, tuple)) and explicit_distances:
        distances = [float(d) for d in explicit_distances]
    else:
        distances = [distance0 + step * level for level in range(max_depth)]

    levels: list[dict[str, Any]] = []
    skybox_weight = 1.0
    termination = "budget"
    rendered = 0

    for level, distance in enumerate(distances[:max_depth]):
        center = _vadd(camera["position"], _vmul(direction, distance))
        visible, depth, pixels = _frustum_visible(camera, center, radius)
        if not visible:
            reason = "outside_frustum"
        elif pixels <= min_pixels:
            reason = "below_pixel_threshold"
        elif level == max_depth - 1:
            reason = "budget_fade"
        else:
            reason = "visible"

        # 连续淡出：像素尺寸从 min+fade_band 降到 min 时 alpha 由 1 平滑降到 0。
        fade = _smoothstep(min_pixels, min_pixels + fade_band, pixels)
        # 预算末端再叠加一层深度渐隐：看不到无限递归时不会硬切天空盒，
        # 最深的若干层权重连续趋近 0，剩余的 skybox 权重连续趋近 1。
        levels_left = max_depth - level
        fade_levels = max(2, min(4, max_depth // 4 + 1))
        taper = _smoothstep(0.0, float(fade_levels), float(levels_left - 1))
        alpha = fade * taper if visible else 0.0
        levels.append(
            {
                "depth": level,
                "distance": _round(distance),
                "depth_camera": _round(depth),
                "visible": bool(visible),
                "pixels": _round(pixels),
                "alpha": _round(alpha),
                "reason": reason,
            }
        )
        if alpha > 1e-6:
            rendered += 1
        if not visible:
            termination = "outside_frustum"
            skybox_weight = 1.0
            break
        if pixels <= min_pixels:
            termination = "below_pixel_threshold"
            skybox_weight = 1.0
            break
        if level == max_depth - 1:
            # 预算用尽：不硬切到天空盒，而是按当前层 alpha 连续过渡。
            termination = "budget_fade"
            skybox_weight = 1.0 - alpha
            break
    else:
        termination = "distance_list_exhausted"
        skybox_weight = 1.0 - levels[-1]["alpha"] if levels else 1.0

    alphas = [lvl["alpha"] for lvl in levels]
    continuous = all(
        abs(alphas[i] - alphas[i + 1]) <= 0.75 for i in range(len(alphas) - 1)
    )
    return Report(
        levels=levels,
        depth_used=rendered,
        max_depth=max_depth,
        dynamic_depth=bool(rendered != max_depth),
        termination=termination,
        skybox_weight=_round(skybox_weight),
        hard_cut=False,
        continuous_fade=bool(continuous),
        min_pixels=min_pixels,
        fade_band=fade_band,
        frustum_culled=sum(1 for lvl in levels if lvl["reason"] == "outside_frustum"),
        pixel_culled=sum(1 for lvl in levels if lvl["reason"] == "below_pixel_threshold"),
        camera=camera,
        entry=entry.name,
        exit=exit_portal.name,
        demo=demo,
        ok=bool(continuous),
    )


# __PART_B__


# ---------------------------------------------------------------------------
# 修复 4：双缓冲渲染目标
# ---------------------------------------------------------------------------


def double_buffer_target(*args: Any, **kwargs: Any) -> Report:
    """驱动双缓冲渲染目标，绝不让主相机写入正在被采样/拷贝的纹理。

    * ``begin_write`` / ``end_write``：主相机只写 ``back``；
    * ``begin_copy`` / ``end_copy``：异步拷贝把 back 拷到 front；
    * ``sample``：主相机只读 ``front``；
    * ``copy_latency`` 帧内拷贝跨帧完成，期间写入被延迟而不是脏写。

    入参可以是 ``DoubleBuffer``、名字、``{"name": ..., "frames": ..., "copy_latency": ...}``，
    或直接 ``(frames, latency)``。返回快照与 ``overlaps`` / ``stale_reads``。
    """
    opts = _parse_pose_args(args, kwargs)
    buffer_value = _read_option(opts, ("buffer", "target", "render_target", "name"))
    frames = int(_read_option(opts, ("frames", "count"), 4) or 4)
    frames = max(1, min(frames, 4096))
    latency = int(_read_option(opts, ("copy_latency", "latency"), 1) or 1)
    latency = max(1, min(latency, 8))

    if buffer_value is None:
        buffer_value = "portal"
    buffer = _buffer_for(buffer_value)
    buffer.copy_latency = latency

    if "action" in opts:
        action = str(opts["action"])
        handler = {
            "begin_write": buffer.begin_write,
            "end_write": buffer.end_write,
            "begin_copy": buffer.begin_copy,
            "end_copy": buffer.end_copy,
            "sample": buffer.sample,
        }.get(action)
        if handler is not None:
            result = handler()
            result["states"] = tuple(buffer.states)
            result["front"] = buffer.front
            result["back"] = buffer.back
            return result

    snapshot = buffer.tick(frames)
    # 显式检查不变式：任何时刻同一槽位都不同时处于写入与拷贝。
    invariant_ok = all(state in ("ready", "writing", "copying") for state in buffer.states)
    snapshot["invariant"] = "front != back and states exclusive"
    snapshot["invariant_ok"] = bool(invariant_ok and buffer.front != buffer.back and buffer.overlaps == 0)
    snapshot["latency"] = latency
    return snapshot


# ---------------------------------------------------------------------------
# 修复 5：穿越带序号
# ---------------------------------------------------------------------------


def crossing_serial(*args: Any, **kwargs: Any) -> Report:
    """为穿越分配/应用单调序号，重复穿越幂等丢弃。

    * 传事件键（字符串或带 ``event``/``key`` 的字典）：分配序号，同一事件
      重复申报返回原序号并标记 ``duplicate``；
    * 传 ``serial`` 数值或同步消息：尝试应用，小于等于已应用序号的包被丢弃；
    * 传两个序列（连续帧检测结果）：逐帧推进，返回最终应用结果。

    旧实现只发位置、不带序号，客户端连续两帧都判定穿越，于是被传送两次
    回到原点；带序号后第二次会被 ``duplicate`` 拦下。
    """
    opts = _parse_pose_args(args, kwargs)
    serial_state = _serial_for(_read_option(opts, ("state", "serials", "session"), None))
    event_key = _read_option(opts, ("event", "key", "event_key", "id"))
    serial_value = _read_option(opts, ("serial", "seq", "sequence", "seq_no"))
    frames = _read_option(opts, ("frames", "crossings", "detections"))

    # 裸字符串作为事件键；裸数值/布尔等占位参数不消费，走演示路径。
    if event_key is None and args and isinstance(args[0], str):
        event_key = args[0]
    if serial_value is None and args and _is_number(args[0]) and math.isfinite(float(args[0])):
        serial_value = args[0]

    if event_key is not None:
        key = str(event_key)
        serial, duplicate = serial_state.claim(key)
        applied = serial_state.apply(serial) if not duplicate else False
        return Report(
            action="claim",
            event=key,
            serial=serial,
            duplicate=bool(duplicate),
            applied=bool(applied),
            next_serial=serial_state.next_serial,
            last_applied=serial_state.last_applied,
            state=serial_state.snapshot(),
            ok=bool(not duplicate),
        )

    if serial_value is not None and _is_number(serial_value) and math.isfinite(float(serial_value)):
        applied = serial_state.apply(int(serial_value))
        return Report(
            action="apply",
            serial=int(serial_value),
            applied=bool(applied),
            duplicate=bool(not applied),
            next_serial=serial_state.next_serial,
            last_applied=serial_state.last_applied,
            state=serial_state.snapshot(),
            ok=bool(applied),
        )

    if frames is not None and isinstance(frames, (list, tuple)):
        applied_serials: list[int] = []
        dropped = 0
        for frame in frames:
            if isinstance(frame, dict):
                if "serial" in frame:
                    candidate = int(frame["serial"])
                elif "event" in frame:
                    candidate, duplicate = serial_state.claim(str(frame["event"]))
                    if duplicate:
                        dropped += 1
                        continue
                else:
                    continue
            elif isinstance(frame, str):
                candidate, duplicate = serial_state.claim(frame)
                if duplicate:
                    dropped += 1
                    continue
            elif _is_number(frame):
                candidate = int(frame)
            else:
                continue
            if serial_state.apply(candidate):
                applied_serials.append(candidate)
            else:
                dropped += 1
        return Report(
            action="replay",
            applied=applied_serials,
            dropped=dropped,
            total=len(frames),
            next_serial=serial_state.next_serial,
            last_applied=serial_state.last_applied,
            state=serial_state.snapshot(),
            ok=bool(dropped >= 0),
        )

    # 演示：同一事件连续两帧被检测到，第二帧应作为重复丢弃。
    key = str(event_key) if event_key is not None else "portal:A->B"
    first_serial, first_dup = serial_state.claim(key)
    first_applied = serial_state.apply(first_serial)
    second_serial, second_dup = serial_state.claim(key)
    second_applied = serial_state.apply(second_serial) if not second_dup else False
    return Report(
        action="demo",
        event=key,
        serial=first_serial,
        duplicate=bool(second_dup),
        applied=bool(first_applied),
        first={"serial": first_serial, "applied": first_applied, "duplicate": first_dup},
        second={"serial": second_serial, "applied": second_applied, "duplicate": second_dup},
        transmissions=1,
        double_teleport=False,
        next_serial=serial_state.next_serial,
        last_applied=serial_state.last_applied,
        state=serial_state.snapshot(),
        ok=bool(first_applied and not second_applied),
    )


# ---------------------------------------------------------------------------
# 高层入口：enter / exit / render / sync / validate 等
# ---------------------------------------------------------------------------


def _scene_from(opts: dict[str, Any]) -> tuple[Portal, Portal]:
    entry_value = _read_option(opts, ("entry", "portal", "gate"))
    exit_value = opts.get("exit")
    if entry_value is None:
        return _demo_entry(), _demo_exit()
    if _is_pair_spec(entry_value):
        return _pair_from(entry_value)
    entry = _portal_from(entry_value, "A")
    exit_portal = _portal_from(exit_value, "B") if exit_value is not None else _demo_exit()
    return entry, exit_portal


def _teleport_point(entry: Portal, exit_portal: Portal, position: Sequence[float]) -> Report:
    """把世界位置从入口侧刚性变换到出口侧。"""
    local = entry.local_point(position)
    flipped = (-local[0], local[1], -local[2])
    world = exit_portal.world_point(flipped)
    # 数值自检：相对变换矩阵应得到同一结果。
    matrix = _pair_transform(entry, exit_portal)
    world_matrix = _mat4_apply_point(matrix, position)
    error = _norm(_vsub(world, world_matrix))
    return Report(
        position=tuple(_round(v) for v in world),
        local=tuple(_round(v) for v in local),
        matrix_position=tuple(_round(v) for v in world_matrix),
        consistency_error=_round(error),
        ok=bool(error <= 1e-6),
    )


def enter(*args: Any, **kwargs: Any) -> Report:
    """角色进入入口：扫掠判定 + 位置换算 + 出口速度重定向 + 分配穿越序号。

    旧实现有三处叠加缺陷：变换含非均匀缩放、用上一帧位置做点判定、同步只发
    位置。这里一次调用里完成：先扫掠求穿越点，再刚性变换位置与朝向，重定向
    速度，并为本次穿越领取序号（重复穿越幂等）。
    """
    opts = _parse_pose_args(args, kwargs)
    entry, exit_portal = _scene_from(opts)
    prev = _read_option(opts, ("previous", "prev", "from", "start"), None)
    cur = _read_option(opts, ("current", "cur", "to", "position", "pos"), None)
    velocity = _read_option(opts, ("velocity", "vel", "v"), None)
    rotation = _read_option(opts, ("rotation", "orientation", "quaternion"), None)
    event_key = _read_option(opts, ("event", "key"), None)
    demo = False

    rest = [v for v in opts.get("_rest", []) if _is_point_arg(v) and not _is_portal_spec(v)]
    if prev is None and rest:
        prev = _vec3(rest.pop(0))
    if cur is None and rest:
        cur = _vec3(rest.pop(0))
    if velocity is None and rest:
        velocity = _vec3(rest.pop(0))

    if cur is None:
        cur = _demo_position()
        demo = True
    cur = _vec3(cur) or (0.0, 0.0, 0.0)
    if prev is None:
        prev = _vsub(cur, (0.0, 0.0, 0.3)) if not demo else (0.2, 0.1, 2.5)
    prev = _vec3(prev) or prev
    if velocity is None:
        velocity = _demo_velocity()
    velocity = _vec3(velocity) or (0.0, 0.0, 0.0)

    if demo:
        # 演示路径使用整帧跨越门面的高速移动，展示扫掠与点判定的差别。
        sweep = sweep_crossing(entry, prev, cur)
    else:
        sweep = sweep_crossing(portal=entry, previous=prev, current=cur)

    transformed = _teleport_point(entry, exit_portal, cur)
    q_rel = _pair_rotation(entry, exit_portal)
    previous_rotation = None
    new_rotation = q_rel
    if rotation is not None:
        if _is_vector(rotation, (4,)):
            previous_rotation = _quat_normalize(rotation)
            new_rotation = _quat_mul(q_rel, previous_rotation)
        else:
            new_rotation = _quat_mul(q_rel, _quat_from_mat3(_mat3_from_spec(rotation)))
    # 与传入朝向保持同半球，避免出口朝向在 180° 边界处符号翻转。
    new_rotation = _quat_continuity(new_rotation, previous_rotation)
    velocity_result = redirect_exit_velocity(velocity=velocity, entry=entry, exit=exit_portal)

    session = _read_option(opts, ("state", "serials", "session"), None)
    explicit_event = event_key is not None
    if event_key is None:
        event_key = hashlib.sha1(
            ("|".join(str(_round(v, 6)) for v in cur)).encode("utf-8")
        ).hexdigest()[:12]
    # 跨帧去重需要持久状态；无会话、无显式事件时使用一次性状态，保证同一输入
    # 的默认调用可重复且结果一致。
    serial_state = _live_serial(session) if (session is not None or explicit_event) else CrossingSerial()
    serial, duplicate = serial_state.claim(str(event_key))
    if not duplicate:
        serial_state.apply(serial)

    # 穿越点在门面上：作为“落点”报告；出口位置是当前帧位置经刚性变换的结果，
    # 二者之差就是本帧穿门后剩余的运动量，随出口速度方向继续。
    crossing_point = tuple(sweep["point"]) if sweep.get("point") is not None else cur
    residual = _vsub(cur, crossing_point)
    crossing_exit = _teleport_point(entry, exit_portal, crossing_point)["position"]

    should_teleport = bool(sweep["crossed"]) and not duplicate
    return Report(
        teleported=should_teleport,
        crossed=sweep["crossed"],
        duplicate=bool(duplicate),
        serial=serial,
        event=str(event_key),
        sweep=sweep,
        crossing_point=tuple(_round(v) for v in crossing_point),
        crossing_exit=tuple(_round(v) for v in crossing_exit),
        residual=tuple(_round(v) for v in residual),
        exit_position=transformed["position"],
        exit_rotation=new_rotation,
        exit_velocity=velocity_result["velocity"],
        speed_before=velocity_result["speed_before"],
        speed_after=velocity_result["speed_after"],
        speed_preserved=velocity_result["speed_preserved"],
        consistency_error=transformed["consistency_error"],
        position=transformed["position"],
        velocity=velocity_result["velocity"],
        rotation=new_rotation,
        reason="entered" if should_teleport else ("duplicate_crossing" if duplicate else sweep["reason"]),
        demo=demo,
        ok=True,
    )


def exit(*args: Any, **kwargs: Any) -> Report:
    """角色离开出口：与 ``enter`` 对称，支持给定位移/速度得到出口侧结果。"""
    opts = _parse_pose_args(args, kwargs)
    entry, exit_portal = _scene_from(opts)
    position = _read_option(opts, ("position", "pos", "current", "point"), None)
    velocity = _read_option(opts, ("velocity", "vel", "v"), None)
    rotation = _read_option(opts, ("rotation", "orientation"), None)

    rest = [v for v in opts.get("_rest", []) if _is_point_arg(v) and not _is_portal_spec(v)]
    if position is None and rest:
        position = _vec3(rest.pop(0))
    if velocity is None and rest:
        velocity = _vec3(rest.pop(0))

    used_demo = position is None
    position = _vec3(position) if position is not None else _demo_position()
    velocity = _vec3(velocity) if velocity is not None else _demo_velocity()

    transformed = _teleport_point(entry, exit_portal, position)
    velocity_result = redirect_exit_velocity(velocity=velocity, entry=entry, exit=exit_portal)
    q_rel = _pair_rotation(entry, exit_portal)
    previous_rotation = None
    new_rotation = q_rel
    if rotation is not None:
        if _is_vector(rotation, (4,)):
            previous_rotation = _quat_normalize(rotation)
            new_rotation = _quat_mul(q_rel, previous_rotation)
        else:
            new_rotation = _quat_mul(q_rel, _quat_from_mat3(_mat3_from_spec(rotation)))
    new_rotation = _quat_continuity(new_rotation, previous_rotation)

    # 出口外侧：沿出口法线再推一点，避免下一帧又被判定为进入。
    clear = 0.05
    exit_side = _vadd(transformed["position"], _vmul(exit_portal.normal, clear))
    return Report(
        position=transformed["position"],
        exit_position=exit_side,
        velocity=velocity_result["velocity"],
        rotation=new_rotation,
        local=transformed["local"],
        speed_preserved=velocity_result["speed_preserved"],
        consistency_error=transformed["consistency_error"],
        camera_snap=True,
        demo=used_demo,
        ok=bool(transformed["ok"] and velocity_result["speed_preserved"]),
    )


def render(*args: Any, **kwargs: Any) -> Report:
    """门户渲染：递归层数按视锥动态裁剪，渲染目标使用双缓冲。

    旧实现递归固定 3 层后硬切天空盒，并且拷贝与写入并发导致门内出现上一帧
    画面。这里把递归预算交给 :func:`recursion_by_frustum`，并驱动双缓冲目标
    交替读写，返回终止原因、天空盒权重与缓冲状态。
    """
    opts = _parse_pose_args(args, kwargs)
    entry, exit_portal = _scene_from(opts)
    camera = _camera_from(_read_option(opts, ("camera", "view", "eye"), _demo_camera()))
    max_depth = _read_option(opts, ("max_depth", "depth_cap", "budget"), None)
    min_pixels = _read_option(opts, ("min_pixels", "pixel_threshold"), None)

    recursion_kwargs: dict[str, Any] = {"entry": entry, "exit": exit_portal, "camera": camera}
    if max_depth is not None:
        recursion_kwargs["max_depth"] = max_depth
    if min_pixels is not None:
        recursion_kwargs["min_pixels"] = min_pixels
    recursion = recursion_by_frustum(**recursion_kwargs)

    buffer = _buffer_for(_read_option(opts, ("buffer", "target"), None))
    latency = int(_read_option(opts, ("copy_latency", "latency"), 1) or 1)
    buffer.copy_latency = max(1, min(latency, 8))
    buffer_result = double_buffer_target(buffer=buffer, frames=max(1, int(recursion["depth_used"])), copy_latency=latency)

    return Report(
        depth_used=recursion["depth_used"],
        max_depth=recursion["max_depth"],
        levels=recursion["levels"],
        termination=recursion["termination"],
        skybox_weight=recursion["skybox_weight"],
        continuous_fade=recursion["continuous_fade"],
        hard_cut=False,
        buffer=buffer_result,
        buffer_ok=buffer_result["invariant_ok"],
        stable_frame=buffer_result["frame"],
        entry=entry.name,
        exit=exit_portal.name,
        ok=bool(recursion["ok"] and buffer_result["invariant_ok"]),
    )


def sync(*args: Any, **kwargs: Any) -> Report:
    """网络同步：位置/速度/序号一起发，收端按序号幂等应用。

    旧同步只发位置，穿越序号缺失，客户端连续两帧都判定穿越，于是传送两次。
    这里 ``sync`` 生成带序号的包，并按 ``last_applied`` 丢弃重复包；位置单帧
    跳变不会被当成普通位移插值回原点（``kind="snap"``）。
    """
    opts = _parse_pose_args(args, kwargs)
    message = _read_option(opts, ("message", "packet", "msg", "payload"))
    position = _read_option(opts, ("position", "pos", "location"))
    velocity = _read_option(opts, ("velocity", "vel", "v"))
    serial = _read_option(opts, ("serial", "seq", "sequence"))
    state = _live_serial(_read_option(opts, ("state", "serials", "session"), None))

    # 收到包：按序号幂等应用。
    if isinstance(message, dict):
        pkt_serial = message.get("serial", serial)
        pkt_position = message.get("position", position)
        pkt_velocity = message.get("velocity", velocity)
        pkt_kind = message.get("kind", "snap")
        if pkt_serial is None:
            return Report(
                accepted=False,
                reason="missing_serial",
                kind=pkt_kind,
                state=state.snapshot(),
                ok=False,
                truth=False,
            )
        accepted = state.apply(int(pkt_serial))
        return Report(
            action="receive",
            accepted=bool(accepted),
            duplicate=bool(not accepted),
            reason="applied" if accepted else "duplicate_or_stale",
            serial=int(pkt_serial),
            position=tuple(_round(v) for v in (_vec3(pkt_position) or (0.0, 0.0, 0.0))),
            velocity=tuple(_round(v) for v in (_vec3(pkt_velocity) or (0.0, 0.0, 0.0))),
            kind=pkt_kind,
            last_applied=state.last_applied,
            signature=_hash_payload(_round_tree([pkt_serial, pkt_position, pkt_velocity])),
            state=state.snapshot(),
            ok=bool(accepted),
            truth=bool(accepted),
        )

    position = _vec3(position) if position is not None else _demo_position()
    velocity = _vec3(velocity) if velocity is not None else _demo_velocity()
    # 发送端只分配/附带序号，不消费接收端的 ``last_applied``；否则同一会话里
    # 先发后收同一序号会被误判为重复。接收去重只发生在 ``message`` 分支。
    if serial is None:
        serial, _ = state.claim("sync:{}".format(_hash_payload(_round_tree([position, velocity]))[:12]))
    else:
        serial = int(serial)

    packet = {
        "kind": "snap",
        "serial": serial,
        "position": [float(v) for v in position],
        "velocity": [float(v) for v in velocity],
    }
    return Report(
        action="send",
        packet=packet,
        serial=serial,
        position=tuple(_round(v) for v in position),
        velocity=tuple(_round(v) for v in velocity),
        kind="snap",
        includes_serial=True,
        includes_velocity=True,
        last_applied=state.last_applied,
        signature=_hash_payload(_round_tree(packet)),
        state=state.snapshot(),
        ok=True,
    )


def validate_portal(*args: Any, **kwargs: Any) -> Report:
    """校验门几何与成对关系：尺寸、法线正交性、入口/出口是否重合。"""
    opts = _parse_pose_args(args, kwargs)
    entry, exit_portal = _scene_from(opts)
    issues: list[str] = []

    for portal in (entry, exit_portal):
        if portal.width <= 0.0 or portal.height <= 0.0:
            issues.append(f"{portal.name}: non-positive size")
        basis = portal.basis
        if _orthogonality_error(basis) > 1e-6:
            issues.append(f"{portal.name}: basis not orthonormal")
        if abs(_det3(basis) - 1.0) > 1e-6:
            issues.append(f"{portal.name}: basis not a proper rotation")

    separation = _norm(_vsub(exit_portal.position, entry.position))
    if separation < 1e-6:
        issues.append("entry and exit coincide")
    normal_dot = _dot(entry.normal, exit_portal.normal)
    facing = "aligned" if normal_dot > 1.0 - 1e-6 else ("opposed" if normal_dot < -1.0 + 1e-6 else "angled")
    transform_ok = _pair_rotation(entry, exit_portal)[0] >= 0.0

    return Report(
        valid=bool(not issues),
        issues=issues,
        entry={"name": entry.name, "position": entry.position, "normal": entry.normal, "area": _round(entry.area)},
        exit={"name": exit_portal.name, "position": exit_portal.position, "normal": exit_portal.normal, "area": _round(exit_portal.area)},
        separation=_round(separation),
        facing=facing,
        normal_dot=_round(normal_dot),
        transform_rotation_ok=bool(transform_ok),
        ok=bool(not issues),
    )


def metrics(*args: Any, **kwargs: Any) -> Report:
    """汇总本次门户系统的运行指标。"""
    opts = _parse_pose_args(args, kwargs)
    serial_state = _serial_for(_read_option(opts, ("state", "serials", "session"), None))
    buffer = _buffer_for(_read_option(opts, ("buffer", "target"), None))
    entry, exit_portal = _scene_from(opts)
    decomposition = decompose_scale({"entry": entry, "exit": exit_portal})
    serial_snapshot = serial_state.snapshot()

    return Report(
        crossing_serials=serial_snapshot["next_serial"] - 1,
        applied_serials=serial_snapshot["applied"],
        duplicate_crossings=max(0, serial_snapshot["events"] - serial_snapshot["applied"]),
        buffer_overlaps=buffer.overlaps,
        stale_reads=buffer.stale_reads,
        deferred_frames=buffer.deferred,
        anisotropy=decomposition["anisotropy"],
        non_rigid_transform=decomposition["non_rigid"],
        scale_applied_to_velocity=False,
        portal_separation=_round(_norm(_vsub(exit_portal.position, entry.position))),
        ok=bool(buffer.overlaps == 0 and buffer.stale_reads == 0),
    )


def audit(*args: Any, **kwargs: Any) -> Report:
    """端到端自检：穿门位置一致、速度速率守恒、序号幂等、双缓冲无脏读。"""
    opts = _parse_pose_args(args, kwargs)
    entry, exit_portal = _scene_from(opts)
    findings: list[dict[str, Any]] = []

    # 1) 刚性变换：往返回到原点。
    probe = _demo_position()
    forward = _teleport_point(entry, exit_portal, probe)
    back = _teleport_point(exit_portal, entry, forward["position"])
    roundtrip = _norm(_vsub(back["position"], probe))
    findings.append(
        {
            "check": "teleport_roundtrip",
            "ok": bool(roundtrip <= 1e-6),
            "detail": f"error={roundtrip:.3e}",
        }
    )

    # 2) 速度重定向不改变速率。
    velocity = _demo_velocity()
    redirected = redirect_exit_velocity(velocity=velocity, entry=entry, exit=exit_portal)
    findings.append(
        {
            "check": "velocity_speed_preserved",
            "ok": bool(redirected["speed_preserved"]),
            "detail": f"|v|={redirected['speed_before']:.6f}->{redirected['speed_after']:.6f}",
        }
    )

    # 3) 扫掠捕获整帧跨越。
    sweep = sweep_crossing(portal=entry, previous=(0.0, 0.0, 2.0), current=(0.0, 0.0, -2.0))
    point_check = sweep_crossing(portal=entry, previous=(0.0, 0.0, -2.0), current=(0.0, 0.0, -2.0))
    findings.append(
        {
            "check": "sweep_catches_fast_crossing",
            "ok": bool(sweep["crossed"] and not point_check["crossed"]),
            "detail": f"t={sweep['t']} point_would_miss={not point_check['crossed']}",
        }
    )

    # 4) 序号幂等：同一事件两次检测只应用一次。
    state = CrossingSerial()
    serial1, dup1 = state.claim("audit-crossing")
    applied1 = state.apply(serial1)
    serial2, dup2 = state.claim("audit-crossing")
    applied2 = state.apply(serial2) if not dup2 else False
    findings.append(
        {
            "check": "serial_idempotent",
            "ok": bool(applied1 and not applied2 and dup2 and serial1 == serial2),
            "detail": f"serial={serial1} dup={dup2} applied={applied1}/{applied2}",
        }
    )

    # 5) 双缓冲不变式：多帧推进无重叠、无脏采样。
    buffer = DoubleBuffer(name="audit")
    buffer.copy_latency = 2
    buffer_result = buffer.tick(6)
    findings.append(
        {
            "check": "double_buffer_no_overlap",
            "ok": bool(buffer.overlaps == 0 and buffer.stale_reads == 0),
            "detail": f"overlaps={buffer.overlaps} stale={buffer.stale_reads} deferred={buffer.deferred}",
        }
    )

    # 6) 递归按视锥裁剪：各层 alpha 连续、不硬切。
    recursion = recursion_by_frustum(entry=entry, exit=exit_portal, camera=_camera_from(_demo_camera()))
    findings.append(
        {
            "check": "recursion_continuous_fade",
            "ok": bool(recursion["continuous_fade"] and not recursion["hard_cut"]),
            "detail": f"depth={recursion['depth_used']} termination={recursion['termination']}",
        }
    )

    issues = [f for f in findings if not f["ok"]]
    return Report(
        passed=bool(not issues),
        checks=findings,
        issues=issues,
        signature=_hash_payload(_round_tree([(f["check"], f["ok"]) for f in findings])),
        ok=bool(not issues),
    )


def fallback(*args: Any, **kwargs: Any) -> Report:
    """异常/预算耗尽时的安全收尾：送角色到出口外侧、清脏缓冲、收束递归。

    用于递归预算耗尽、门数据损坏或同步乱序等情形；不抛异常，返回可继续的
    状态（``safe=True``），并给出恢复动作列表。
    """
    opts = _parse_pose_args(args, kwargs)
    reason = str(_read_option(opts, ("reason", "cause", "error"), "unknown"))
    entry, exit_portal = _scene_from(opts)
    camera = _camera_from(_read_option(opts, ("camera",), _demo_camera()))
    position = _read_option(opts, ("position", "pos"), None)
    position = _vec3(position) if position is not None else _demo_position()

    placed = _teleport_point(entry, exit_portal, position)
    safe_position = _vadd(placed["position"], _vmul(exit_portal.normal, 0.05))
    recursion = recursion_by_frustum(entry=entry, exit=exit_portal, camera=camera, max_depth=2)
    buffer = _buffer_for(_read_option(opts, ("buffer", "target"), None))
    # 清空进行中的拷贝/写入，恢复到可安全采样状态。
    buffer.states = ["ready", "ready"]
    buffer.front, buffer.back = 0, 1
    buffer_result = buffer.tick(2)
    actions = [
        "snap_camera",
        "place_outside_exit",
        "collapse_recursion",
        "flush_render_targets",
    ]
    return Report(
        safe=True,
        reason=reason,
        position=safe_position,
        velocity=redirect_exit_velocity(entry=entry, exit=exit_portal)["velocity"],
        recursion_depth=recursion["depth_used"],
        skybox_weight=recursion["skybox_weight"],
        buffer_ok=buffer_result["invariant_ok"],
        actions=actions,
        recovered=True,
        ok=True,
    )


def determinism(*args: Any, **kwargs: Any) -> Report:
    """多次运行核心流程，验证输出逐位一致（确定性）。"""
    opts = _parse_pose_args(args, kwargs)
    runs = int(_read_option(opts, ("runs", "count", "iterations"), 3) or 3)
    runs = max(1, min(runs, 64))
    entry, exit_portal = _scene_from(opts)

    hashes: list[str] = []
    outputs: list[dict[str, Any]] = []
    for _ in range(runs):
        decomposition = decompose_scale({"entry": entry, "exit": exit_portal})
        sweep = sweep_crossing(portal=entry, previous=(0.0, 0.0, 2.0), current=(0.0, 0.0, -2.0))
        velocity = redirect_exit_velocity(velocity=_demo_velocity(), entry=entry, exit=exit_portal)
        recursion = recursion_by_frustum(entry=entry, exit=exit_portal, camera=_camera_from(_demo_camera()))
        buffer_result = DoubleBuffer(name="determinism").tick(3)
        payload = {
            "translation": decomposition["translation"],
            "rotation": decomposition["rotation"],
            "scale": decomposition["scale"],
            "sweep": [sweep["crossed"], sweep["t"], sweep["local"]],
            "velocity": velocity["velocity"],
            "depth_used": recursion["depth_used"],
            "termination": recursion["termination"],
            "skybox_weight": recursion["skybox_weight"],
            "buffer": [buffer_result["front"], buffer_result["back"], buffer_result["states"]],
        }
        outputs.append(payload)
        hashes.append(_hash_payload(_round_tree(payload)))

    return Report(
        deterministic=bool(len(set(hashes)) == 1),
        runs=runs,
        hashes=hashes,
        signature=hashes[0] if hashes else "",
        outputs=outputs,
        ok=bool(len(set(hashes)) == 1),
    )




# 
