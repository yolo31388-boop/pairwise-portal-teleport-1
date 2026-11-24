# pairwise-portal-teleport

## 领域

游戏、图形与多媒体

## 主题

传送门穿越坐标变换

## 说明

变换分解、扫掠检测与穿越序号。

初始状态下 `gameplay/portal_teleport.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
