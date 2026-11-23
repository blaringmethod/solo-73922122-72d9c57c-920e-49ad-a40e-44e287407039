"""厘米高程 -> 分米高程 的保流向转换求解器。

核心思路
========
输入整数厘米格网 H，求整数分米格网 D，使得：

1. 每格误差受限：``|10*D[u] - H[u]| <= E``（E 为最大允许误差，单位厘米）；
2. 流向一致：每格的流向 = 四邻中「最低且严格低于本格」的全部邻格
   （并列全部保留；无下坡则流向为空），转换前后该集合必须相同；
3. 锁定格固定为指定分米值。

所有条件都可写成*整数差分约束* ``D[v] - D[u] <= w``：

* 误差上下界（含锁定值）：作为每格的绝对取值区间，以上界为 SPFA
  初始距离（隐式 0 权超级源），求出最大解后再校验下界；
* 邻格比较：H[v] < H[u] 必须保持 D[v] < D[u]（边 u->v，权 -1），
  否则必须保持 D[v] >= D[u]（边 v->u，权 0）；
* 并列最低邻格之间要求 D 相等（两条 0 权边）；
* 最低邻格与其他下坡邻格之间要求严格小于（权 -1 边）。

差分约束系统可行 <=> 约束图无负环且最大解满足全部下界，用 SPFA
判定；无冲突时得到一组合法 D。最小可行误差通过对 E 二分得到。
"""

from __future__ import annotations

import math
from collections import deque

MAX_DIM = 32
NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))


class ValidationError(ValueError):
    """CSV / 参数校验失败。"""


# --------------------------------------------------------------------------- #
# 输入解析
# --------------------------------------------------------------------------- #
def parse_csv(text: str) -> list[list[int]]:
    """解析整数 CSV 文本为二维列表，并校验尺寸与取值。"""
    grid: list[list[int]] = []
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        tokens = [tok.strip() for tok in line.split(",")]
        try:
            row = [int(tok) for tok in tokens]
        except ValueError as exc:
            raise ValidationError(f"第 {lineno} 行包含非整数高程值") from exc
        if not row:
            raise ValidationError(f"第 {lineno} 行为空")
        grid.append(row)

    if not grid:
        raise ValidationError("CSV 内容为空，至少需要 1 行数据")
    width = len(grid[0])
    for i, row in enumerate(grid, 1):
        if len(row) != width:
            raise ValidationError(f"第 {i} 行列数为 {len(row)}，与首行 {width} 不一致")
    if len(grid) > MAX_DIM or width > MAX_DIM:
        raise ValidationError(f"格网尺寸 {len(grid)}×{width} 超过上限 {MAX_DIM}×{MAX_DIM}")
    return grid


def to_csv(grid: list[list[int]]) -> str:
    """把整数格网序列化为 CSV 文本。"""
    return "".join(",".join(str(v) for v in row) + "\n" for row in grid)


# --------------------------------------------------------------------------- #
# 流向
# --------------------------------------------------------------------------- #
def flow_sets(grid: list[list[int]]) -> list[list[list[tuple[int, int]]]]:
    """计算每格流向：四邻中最低且严格低于本格的所有邻格坐标列表。

    无下坡邻格时返回空列表（即“无下坡”）。
    """
    rows, cols = len(grid), len(grid[0])
    flows: list[list[list[tuple[int, int]]]] = [
        [[] for _ in range(cols)] for _ in range(rows)
    ]
    for r in range(rows):
        for c in range(cols):
            h = grid[r][c]
            lower: list[tuple[int, int, int]] = []
            for dr, dc in NEIGHBORS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] < h:
                    lower.append((nr, nc, grid[nr][nc]))
            if not lower:
                continue
            minimum = min(v for _, _, v in lower)
            flows[r][c] = [(nr, nc) for nr, nc, v in lower if v == minimum]
    return flows


# --------------------------------------------------------------------------- #
# 差分约束
# --------------------------------------------------------------------------- #
def _cell_bounds(h: int, error_cm: int) -> tuple[int, int]:
    """满足 |10D - h| <= E 的整数分米 D 的上下界。"""
    return math.ceil((h - error_cm) / 10), (h + error_cm) // 10


def build_system(
    grid: list[list[int]],
    locks: dict[tuple[int, int], int],
    error_cm: int,
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int]]] | None:
    """构造约束系统（无超级源版本）。

    返回 (相对约束边, 每格 (下界, 上界))；若某格在该误差下无整数分米
    取值、或锁定值落在误差区间之外，返回 None。

    边 (u, v, w) 表示 ``D[v] - D[u] <= w``，只包含流向比较约束；
    绝对上下界通过“以上界为 SPFA 初值、求解后校验下界”处理。
    """
    rows, cols = len(grid), len(grid[0])
    n = rows * cols

    def idx(r: int, c: int) -> int:
        return r * cols + c

    bounds: list[tuple[int, int]] = [(0, 0)] * n

    # 1) 误差界 / 锁定值 -------------------------------------------------- #
    for r in range(rows):
        for c in range(cols):
            lo, hi = _cell_bounds(grid[r][c], error_cm)
            if lo > hi:
                return None
            if (r, c) in locks:
                locked = locks[(r, c)]
                if not lo <= locked <= hi:
                    # 锁定值本身就超出该格误差允许范围
                    return None
                lo = hi = locked
            bounds[idx(r, c)] = (lo, hi)

    edges: list[tuple[int, int, int]] = []

    # 2) 流向保持约束 ----------------------------------------------------- #
    for r in range(rows):
        for c in range(cols):
            h = grid[r][c]
            u = idx(r, c)
            lower_nodes: list[tuple[int, int, int]] = []
            for dr, dc in NEIGHBORS:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue
                v = idx(nr, nc)
                hv = grid[nr][nc]
                if hv < h:
                    # 下坡邻格必须保持严格更低：D[v] <= D[u] - 1
                    edges.append((u, v, -1))
                    lower_nodes.append((nr, nc, hv))
                else:
                    # 等高或更高邻格不得变成下坡：D[v] >= D[u]
                    edges.append((v, u, 0))
            if lower_nodes:
                minimum = min(hv for _, _, hv in lower_nodes)
                min_nodes = [idx(nr, nc) for nr, nc, hv in lower_nodes if hv == minimum]
                other_nodes = [idx(nr, nc) for nr, nc, hv in lower_nodes if hv > minimum]
                # 并列最低：分米值必须相等，保证并列关系保留。
                for i in range(len(min_nodes)):
                    for j in range(i + 1, len(min_nodes)):
                        a, b = min_nodes[i], min_nodes[j]
                        edges.append((a, b, 0))
                        edges.append((b, a, 0))
                # 最低邻格必须严格低于其他下坡邻格，保证最低归属不变。
                for low in min_nodes:
                    for other in other_nodes:
                        edges.append((other, low, -1))

    return edges, bounds


def spfa(
    edges: list[tuple[int, int, int]],
    node_count: int,
    initial: list[int],
) -> list[int] | None:
    """SPFA 求最短路并检测负环。

    各节点距离以其上界初始化且全部入队（相当于隐式 0 权超级源），
    得到的是逐点最大的可行解。某节点第 node_count 次入队说明存在
    负环（流向/锁定矛盾）-> 返回 None。
    """
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(node_count)]
    for u, v, w in edges:
        adjacency[u].append((v, w))

    dist = initial[:]
    enqueue_count = [0] * node_count
    in_queue = [True] * node_count
    queue = deque(range(node_count))

    while queue:
        u = queue.popleft()
        in_queue[u] = False
        du = dist[u]
        for v, w in adjacency[u]:
            candidate = du + w
            if candidate < dist[v]:
                dist[v] = candidate
                if not in_queue[v]:
                    enqueue_count[v] += 1
                    if enqueue_count[v] >= node_count:
                        return None
                    queue.append(v)
                    in_queue[v] = True
    return dist


# --------------------------------------------------------------------------- #
# 完整转换流程
# --------------------------------------------------------------------------- #
def _big_error(grid: list[list[int]]) -> int:
    """一个足以容纳全部排序/锁定需求的误差上界（厘米）。"""
    flat = [v for row in grid for v in row]
    span = max(flat) - min(flat)
    return max(200, 10 * span + 100)


def verify(
    grid_cm: list[list[int]],
    grid_dm: list[list[int]],
    locks: dict[tuple[int, int], int],
    error_cm: int,
) -> list[list[int]]:
    """对候选分米格网做完整自检，失败抛 AssertionError，返回逐格误差。"""
    rows, cols = len(grid_cm), len(grid_cm[0])
    assert len(grid_dm) == rows and all(len(r) == cols for r in grid_dm)
    errors: list[list[int]] = [[0] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            diff = 10 * grid_dm[r][c] - grid_cm[r][c]
            assert abs(diff) <= error_cm, f"({r},{c}) 误差 {diff} 超限"
            if (r, c) in locks:
                assert grid_dm[r][c] == locks[(r, c)], f"({r},{c}) 未满足锁定"
            errors[r][c] = diff
    cm_flow = flow_sets(grid_cm)
    dm_flow = flow_sets(grid_dm)
    assert cm_flow == dm_flow, "流向与厘米底图不一致"
    return errors


def solve_at(
    grid: list[list[int]],
    locks: dict[tuple[int, int], int],
    error_cm: int,
) -> list[list[int]] | None:
    """在给定误差下求一组分米格网；不可行返回 None。"""
    rows, cols = len(grid), len(grid[0])
    system = build_system(grid, locks, error_cm)
    if system is None:
        return None
    edges, bounds = system
    initial = [hi for _lo, hi in bounds]
    dist = spfa(edges, len(bounds), initial)
    if dist is None:
        return None  # 负环：流向/锁定矛盾
    # 最大解仍低于下界 -> 差分约束与绝对区间不可兼得
    if any(dist[i] < bounds[i][0] for i in range(len(bounds))):
        return None
    dm = [[dist[r * cols + c] for c in range(cols)] for r in range(rows)]
    verify(grid, dm, locks, error_cm)
    return dm


def min_feasible_error(
    grid: list[list[int]], locks: dict[tuple[int, int], int]
) -> int | None:
    """二分最小可行误差；锁定与流向根本冲突时返回 None。"""
    big = _big_error(grid)
    if solve_at(grid, locks, big) is None:
        return None
    lo, hi = 0, big
    while lo < hi:
        mid = (lo + hi) // 2
        if solve_at(grid, locks, mid) is not None:
            hi = mid
        else:
            lo = mid + 1
    return lo


def convert(
    grid: list[list[int]],
    locks: dict[tuple[int, int], int],
    requested_error_cm: int,
) -> dict:
    """转换主入口。

    返回 dict：
      feasible=True  -> grid_dm / errors / max_abs_error / min_error_cm
      feasible=False -> reason="error_too_tight"（带 min_error_cm）
                        或 reason="hard_conflict"（锁定与流向不可兼得）
    """
    if not isinstance(requested_error_cm, int) or requested_error_cm < 0:
        raise ValidationError("最大允许误差必须是非负整数（厘米）")

    minimum = min_feasible_error(grid, locks)
    if minimum is None:
        return {"feasible": False, "reason": "hard_conflict"}

    if requested_error_cm < minimum:
        return {
            "feasible": False,
            "reason": "error_too_tight",
            "min_error_cm": minimum,
        }

    dm = solve_at(grid, locks, requested_error_cm)
    assert dm is not None  # 上面已证明可行
    errors = verify(grid, dm, locks, requested_error_cm)
    max_abs = max(abs(v) for row in errors for v in row)
    return {
        "feasible": True,
        "reason": None,
        "grid_dm": dm,
        "errors": errors,
        "max_abs_error": max_abs,
        "min_error_cm": minimum,
    }


def naive_round(grid: list[list[int]]) -> list[list[int]]:
    """旧程序式逐格四舍五入到分米，仅用于对比演示。"""
    return [[(h + 5) // 10 if h >= 0 else -((-h + 5) // 10) for h in row] for row in grid]
