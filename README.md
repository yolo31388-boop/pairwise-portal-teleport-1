# pairwise-portal-teleport

## 领域

游戏、图形与多媒体

## 主题

传送门穿越坐标变换

## 说明

变换分解、扫掠检测与穿越序号。

`gameplay/portal_teleport.py` 已实现六项修复：

- `decompose_scale`：极分解把入口→出口变换拆成平移 / 旋转 / 缩放（含剪切、
  镜像），朝向与速度只取旋转部分，非均匀缩放不会把角色拉扁。
- `sweep_crossing`：用上一帧到当前帧的线段与门面求交，高速整帧跨门也不漏判。
- `recursion_by_frustum`：递归层数按视锥可见性与屏幕占比动态决定，末端连续
  淡出到天空盒，不再固定三层后硬切。
- `double_buffer_target`：front/back 分离，写与采样永不重叠，异步拷贝未完成时
  延迟写入而不是脏读上一帧门内景。
- `crossing_serial`：每次穿越带单调序号，重复检测/重复包幂等丢弃；同步消息
  同时携带位置、速度与序号。
- `redirect_exit_velocity`：速度按出口相对入口的旋转（含门户翻转）重定向，
  速率守恒，且不受门缩放影响。

`enter` / `exit` / `render` / `sync` 组合上述能力，`validate_portal` /
`metrics` / `audit` / `fallback` / `determinism` 提供校验与收尾。公开入口对
非法输入返回 `{"ok": False, ...}` 报告而不是抛出异常。

运行：

```bash
python -m pytest -q
```

`tests/test_portal_fixes.py` 覆盖六项修复的行为回归。
