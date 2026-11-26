"""六项传送门穿越缺陷的回归测试。

原 ``test_portal_teleport.py`` 只验证公开入口可调用；本文件验证题目描述中的
变换分解缩放、扫掠检测穿越、递归按视锥裁剪、双缓冲渲染目标、穿越带序号和
出口速度重定向。
"""

from __future__ import annotations

import math

import pytest

import gameplay.portal_teleport as mod


def _portal_pair():
    entry = mod.Portal("A", (0.0, 0.0, 0.0))
    exit_portal = mod.Portal("B", (6.0, 0.0, 0.0))
    return entry, exit_portal


# ---------------------------------------------------------------------------
# 1) 变换分解缩放
# ---------------------------------------------------------------------------


def test_decompose_scale_recovers_non_uniform_scale():
    rotation = mod._mat3_from_quat(mod._quat_from_angle_y(30.0))
    scales = (2.0, 0.5, 3.0)
    basis = tuple(tuple(rotation[i][j] * scales[j] for j in range(3)) for i in range(3))
    matrix = (
        (basis[0][0], basis[0][1], basis[0][2], 1.0),
        (basis[1][0], basis[1][1], basis[1][2], 2.0),
        (basis[2][0], basis[2][1], basis[2][2], 3.0),
        (0.0, 0.0, 0.0, 1.0),
    )

    result = mod.decompose_scale(matrix)
    assert sorted(round(v, 6) for v in result["scale"]) == sorted(scales)
    assert result["translation"] == (1.0, 2.0, 3.0)
    assert result["anisotropy"] == pytest.approx(6.0)
    assert result["orthogonality_error"] < 1e-6
    assert result["roundtrip_error"] < 1e-6


def test_decomposed_rotation_is_pure_rotation():
    # 非均匀缩放不得污染旋转：分解出的旋转作用到单位 X 上应与纯旋转一致。
    rotation = mod._mat3_from_quat(mod._quat_from_angle_y(30.0))
    basis = tuple(tuple(rotation[i][j] * (2.0, 0.5, 3.0)[j] for j in range(3)) for i in range(3))
    matrix = [list(row) + [0.0] for row in basis] + [[0.0, 0.0, 0.0, 1.0]]

    result = mod.decompose_scale(matrix)
    rotated = mod._quat_rotate(result["rotation"], (1.0, 0.0, 0.0))
    expected = mod._mat3_apply(rotation, (1.0, 0.0, 0.0))
    assert max(abs(a - b) for a, b in zip(rotated, expected)) < 1e-6


def test_portal_pair_transform_is_rigid():
    entry, exit_portal = _portal_pair()
    result = mod.decompose_scale({"entry": entry, "exit": exit_portal})
    assert result["anisotropy"] == pytest.approx(1.0)
    assert result["has_shear"] is False
    assert result["non_rigid"] is False


def test_decompose_detects_shear_and_reflection():
    shear = mod.decompose_scale((1, 0.5, 0, 0, 1, 0, 0, 0, 1))
    assert shear["has_shear"] is True
    assert shear["shear"] > 0.0

    mirror = mod.decompose_scale((-1, 0, 0, 0, 1, 0, 0, 0, 1))
    assert mirror["reflected"] is True
    assert mirror["handedness"] == -1
    assert mirror["roundtrip_error"] < 1e-6


# ---------------------------------------------------------------------------
# 2) 扫掠检测穿越
# ---------------------------------------------------------------------------


def test_sweep_catches_single_frame_overshoot():
    portal = mod.Portal("A", (0.0, 0.0, 0.0), width=2.0, height=2.0)
    result = mod.sweep_crossing(portal=portal, previous=(0.0, 0.0, 10.0), current=(0.0, 0.0, -10.0))

    assert result["crossed"] is True
    assert result["inside"] is True
    assert result["t"] == pytest.approx(0.5)
    assert result["local"] == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)


def test_point_check_would_miss_but_sweep_does_not():
    portal = mod.Portal("A", (0.0, 0.0, 0.0), width=2.0, height=2.0)
    # 上一帧与当前帧都在门后：点判定不会触发，扫掠段也没跨面。
    stationary = mod.sweep_crossing(portal=portal, previous=(0.0, 0.0, -2.0), current=(0.0, 0.0, -2.0))
    crossing = mod.sweep_crossing(portal=portal, previous=(0.0, 0.0, 2.0), current=(0.0, 0.0, -2.0))
    assert stationary["crossed"] is False
    assert crossing["crossed"] is True


def test_sweep_rejects_crossing_outside_portal_bounds():
    portal = mod.Portal("A", (0.0, 0.0, 0.0), width=2.0, height=2.0)
    result = mod.sweep_crossing(portal=portal, previous=(5.0, 0.0, 5.0), current=(5.0, 0.0, -5.0))
    assert result["crossed_plane"] is True
    assert result["crossed"] is False
    assert result["reason"] == "outside_portal_bounds"


# ---------------------------------------------------------------------------
# 3) 递归按视锥裁剪
# ---------------------------------------------------------------------------


def test_recursion_depth_is_dynamic_not_fixed_three():
    entry, exit_portal = _portal_pair()
    result = mod.recursion_by_frustum(entry=entry, exit=exit_portal, camera=mod._demo_camera())
    assert result["max_depth"] != 3 or result["depth_used"] != 3
    assert result["dynamic_depth"] is True


def test_recursion_fades_continuously_to_skybox():
    entry, exit_portal = _portal_pair()
    result = mod.recursion_by_frustum(entry=entry, exit=exit_portal, camera=mod._demo_camera())
    alphas = [level["alpha"] for level in result["levels"]]
    assert result["hard_cut"] is False
    assert result["continuous_fade"] is True
    assert all(0.0 <= a <= 1.0 for a in alphas)
    # 相邻层透明度平滑变化，没有从 1 直接跳到 0。
    assert all(abs(alphas[i] - alphas[i + 1]) <= 0.75 for i in range(len(alphas) - 1))


def test_recursion_culls_out_of_frustum_portals():
    entry, exit_portal = _portal_pair()
    behind = dict(mod._demo_camera())
    behind["position"] = (0.0, 0.0, -6.0)
    behind["forward"] = (0.0, 0.0, -1.0)
    result = mod.recursion_by_frustum(entry=entry, exit=exit_portal, camera=behind)
    assert result["depth_used"] == 0
    assert result["termination"] == "outside_frustum"


# ---------------------------------------------------------------------------
# 4) 双缓冲渲染目标
# ---------------------------------------------------------------------------


def test_double_buffer_never_overlaps_read_and_write():
    buffer = mod.DoubleBuffer("test-overlap")
    buffer.copy_latency = 2
    snapshot = buffer.tick(12)
    assert snapshot["overlaps"] == 0
    assert snapshot["stale_reads"] == 0
    assert snapshot["front"] != snapshot["back"]


def test_double_buffer_defers_write_during_async_copy():
    buffer = mod.DoubleBuffer("test-defer")
    buffer.begin_write()
    buffer.end_write()
    copy = buffer.begin_copy()
    assert copy["copying"] is True

    # 拷贝进行中：采样 front 仍允许，但不得开始写入同一目标。
    assert buffer.sample()["allowed"] is True
    begin = buffer.begin_write()
    assert begin["deferred"] is True
    assert begin["reason"] == "copy_in_progress"


def test_double_buffer_copy_latency_completes_before_sample():
    buffer = mod.DoubleBuffer("test-latency")
    buffer.copy_latency = 3
    snapshot = buffer.tick(20)
    assert snapshot["deferred"] > 0
    assert all(sample["sample_allowed"] for sample in snapshot["samples"])


# ---------------------------------------------------------------------------
# 5) 穿越带序号
# ---------------------------------------------------------------------------


def test_crossing_serial_is_monotonic_and_deduplicated():
    state = mod.CrossingSerial()
    serial, duplicate = state.claim("portal:A->B")
    assert duplicate is False
    assert state.apply(serial) is True

    again, duplicate_again = state.claim("portal:A->B")
    assert again == serial
    assert duplicate_again is True
    assert state.apply(again) is False

    second, _ = state.claim("portal:A->C")
    assert second > serial


def test_sync_packet_carries_serial_and_drops_duplicates():
    message = {"serial": 7, "position": [1.0, 2.0, 3.0], "velocity": [0.0, 0.0, -1.0]}
    first = mod.sync(message=message, session="dup-check")
    second = mod.sync(message=message, session="dup-check")
    assert first["accepted"] is True
    assert second["accepted"] is False
    assert second["duplicate"] is True

    missing = mod.sync(message={"position": [1.0, 2.0, 3.0]})
    assert missing["accepted"] is False
    assert missing["reason"] == "missing_serial"


def test_enter_teleports_once_for_repeated_frames():
    entry, exit_portal = _portal_pair()
    kwargs = dict(
        entry=entry,
        exit=exit_portal,
        previous=(0.2, 0.1, 2.5),
        current=(0.2, 0.1, -2.5),
        velocity=(0.0, 0.0, -4.0),
        event="crossing-42",
        session="enter-dedup",
    )
    first = mod.enter(**kwargs)
    second = mod.enter(**kwargs)
    assert first["teleported"] is True
    assert second["duplicate"] is True
    assert second["teleported"] is False


# ---------------------------------------------------------------------------
# 6) 出口速度重定向
# ---------------------------------------------------------------------------


def test_exit_velocity_keeps_speed_under_non_uniform_scale():
    # 完整变换含缩放时，速度只随旋转，速率不变。
    scaled = ((2.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 0.5, 0.0), (0.0, 0.0, 0.0, 1.0))
    result = mod.redirect_exit_velocity(velocity=(0.0, 0.0, -3.0), matrix=scaled)
    assert result["speed_before"] == pytest.approx(3.0)
    assert result["speed_after"] == pytest.approx(3.0)
    assert result["scale_applied"] is False


def test_exit_velocity_follows_portal_rotation():
    entry = mod.Portal("A", (0.0, 0.0, 0.0))
    exit_portal = mod.Portal("B", (6.0, 0.0, 0.0), rotation=mod._quat_from_angle_y(90.0))
    result = mod.redirect_exit_velocity(velocity=(0.0, 0.0, -3.0), entry=entry, exit=exit_portal)
    assert result["speed_after"] == pytest.approx(3.0)
    assert result["velocity"][0] == pytest.approx(3.0, abs=1e-6)


def test_portal_flip_reverses_forward_velocity():
    entry = mod.Portal("A", (0.0, 0.0, 0.0))
    exit_portal = mod.Portal("B", (6.0, 0.0, 0.0))
    result = mod.redirect_exit_velocity(velocity=(0.0, 0.0, -3.0), entry=entry, exit=exit_portal)
    assert result["velocity"][2] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# 综合：穿越位置、自检与确定性
# ---------------------------------------------------------------------------


def test_teleport_roundtrip_is_exact():
    entry, exit_portal = _portal_pair()
    position = (0.7, 0.3, 0.1)
    forward = mod._teleport_point(entry, exit_portal, position)["position"]
    back = mod._teleport_point(exit_portal, entry, forward)["position"]
    assert max(abs(a - b) for a, b in zip(position, back)) < 1e-9


def test_audit_and_determinism_pass():
    audit = mod.audit(1)
    assert audit["passed"] is True, audit["issues"]

    result = mod.determinism(1)
    assert result["deterministic"] is True
    assert len(set(result["hashes"])) == 1
