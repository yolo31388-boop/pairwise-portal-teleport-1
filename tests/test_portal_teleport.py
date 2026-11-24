import pytest
import gameplay.portal_teleport as mod

CASES = [
    ("decompose_scale", (1,)),
    ("sweep_crossing", (1,)),
    ("recursion_by_frustum", (1,)),
    ("double_buffer_target", (1,)),
    ("crossing_serial", (1,)),
    ("redirect_exit_velocity", (1,)),
    ("enter", (1,)),
    ("exit", (1,)),
    ("render", (1,)),
    ("sync", (1,)),
    ("validate_portal", (1,)),
    ("metrics", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("determinism", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
