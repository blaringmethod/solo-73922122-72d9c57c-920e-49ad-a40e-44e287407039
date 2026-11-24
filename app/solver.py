"""厘米高程 -> 分米高程 的保流向转换求解器。

核心思路
========
输入整数厘米格网 H，求整数分米格网 D，使得：

1. 每格误差受限：``|10*D[u] - H[u]| <= E``（E 为最大允许误差，单位厘米）；
2. 流向一致：每格的流向 = 四邻中「最低且严格低于本格」的全部邻格
   （并列全部保留；无下坡则流向为空），转换前后该集合必须相同；
3. 锁定格固定为指定分米值。

所有条件都写成*整数差分约束* ``D[v] - D[u] <= w``（w 为 -1 或 0）：

* 下坡邻格：``D[v] <= D[u] - 1``；
* 等高邻格必须同值、更高邻格不得变成下坡；
* 并列最低邻格之间分米值相等；最低邻格严格低于其他下坡邻格。

**约束图是类 DAG**：0 权边只连接等高格。先用并查集把双向 0 权边缩成
「必须同值」的分量，剩余有向边全部从较高分量指向较低分量，按高程从高到低
一次拓扑扫描即可得到最大可行解，不再使用 SPFA 反复松弛，也不会出现负环
（无锁定时约束天然相容）。

两种查询
--------
* ``FlowSystem.solve``：给定误差限，一次标量扫描得到最大可行分米解；
* ``FlowSystem.min_error``：一次「11 标签」扫描，直接、精确地得到最小
  可行整数厘米误差。10 个标签对应 E mod 10 的十种余数，另 1 个标签专门
  携带锁定固定值的传播。之后只做 O(10*n) 的逐格汇总，不再对多个候选
  误差逐个跑全图。

最小误差的精确性
----------------
把 E 写成 ``10k + e``（e 固定）。非锁定格在余数 e 下的上界为
``k + floor((h+e)/10)``、下界为 ``ceil((h-e)/10) - k``（单位分米）；锁定格
的值为常数 L。对约束图做一次 11 标签扫描：前 10 个标签各带一种余数 e 的
上界随 k 增长的部分（形如 ``A + k``，A 为整数常量），第 11 个标签只传播
锁定常数 L。汇总时对每格、每个 e 解出所需 k 的闭式下界（非锁定对为
``2k >= c - A``，锁定源/锁定目标分别为常数阈值），取 10 个余数中的最小 E
即为精确最小值。锁定值离测绘高程很远时，锁定自身的误差下界 ``|10L-h|``
直接进入结果；两条锁定路径互相矛盾时第 11 个标签低于锁定固定值，判定为
放宽也无解的硬冲突。

缓存
----
``get_system`` 以厘米格网内容为键缓存流向约束图，``min_feasible_error``
再按（格网, 锁定集合）缓存最小误差结果。同一份格网反复改误差限、增删/
修改锁定都复用同一张图，只重跑标签扫描；完全相同的试算直接命中结果缓存。
更正上传换了厘米高程会按新格网重建图并重新计算误差，不会沿用旧误差。
"""

from __future__ import annotations

from collections import OrderedDict

MAX_DIM = 32
NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _low_dm(h: int, error_cm: int) -> int:
    """满足 |10D-h| <= E 的最小整数分米 D（整数运算，避免浮点误差）。"""
    return -((error_cm - h) // 10)


def _high_dm(h: int, error_cm: int) -> int:
    """满足 |10D-h| <= E 的最大整数分米 D。"""
    return (h + error_cm) // 10


# 远端哨兵：float inf 与整数混用只做比较，有限值运算仍保持精确整数。
INF = float("inf")

# 缓存多少个不同的厘米格网约束图
SYSTEM_CACHE_SIZE = 32


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
# 约束图构造
# --------------------------------------------------------------------------- #
def _build_graph(grid: list[list[int]]):
    """按厘米格网构图，并把必须同值的 0 权边缩成并查集分量。

    返回 ``(level_nodes, level_comps, comps, out_edges)``：

    * ``level_nodes[r]``：高程等级 r（越大越高）的全部格；
    * ``level_comps[r]``：该等级上的同值分量编号；
    * ``comps[c]``：分量的成员格编号；
    * ``out_edges[i]``：格 i 的出边 ``(v, w)``（已去掉分量内部边）。
    """
    rows, cols = len(grid), len(grid[0])
    n = rows * cols
    ordered = sorted({h for row in grid for h in row})
    rank_of = {h: i for i, h in enumerate(ordered)}
    rank = [rank_of[grid[r][c]] for r in range(rows) for c in range(cols)]

    def idx(r: int, c: int) -> int:
        return r * cols + c

    parent = list(range(n))

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    edges: list[tuple[int, int, int]] = []
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
                elif hv == h:
                    # 等高邻格双方都不许成为对方的下坡：D[u] == D[v]
                    union(u, v)
                else:
                    # 更高邻格不得变成下坡：D[u] <= D[v]
                    edges.append((v, u, 0))
            if lower_nodes:
                minimum = min(hv for _, _, hv in lower_nodes)
                min_nodes = [idx(nr, nc) for nr, nc, hv in lower_nodes if hv == minimum]
                other_nodes = [idx(nr, nc) for nr, nc, hv in lower_nodes if hv > minimum]
                # 并列最低：分米值必须相等，保证并列关系保留。
                for i in range(1, len(min_nodes)):
                    union(min_nodes[0], min_nodes[i])
                # 最低邻格必须严格低于其他下坡邻格，保证最低归属不变。
                for low in min_nodes:
                    for other in other_nodes:
                        edges.append((other, low, -1))

    comp_of = [find(i) for i in range(n)]
    comp_index: dict[int, int] = {}
    comps: list[list[int]] = []
    for i in range(n):
        root = comp_of[i]
        ci = comp_index.get(root)
        if ci is None:
            ci = len(comps)
            comp_index[root] = ci
            comps.append([])
        comps[ci].append(i)
        comp_of[i] = ci

    level_nodes: list[list[int]] = [[] for _ in range(len(ordered))]
    for i, rk in enumerate(rank):
        level_nodes[rk].append(i)
    level_comps: list[list[int]] = [[] for _ in range(len(ordered))]
    seen: set[tuple[int, int]] = set()
    for rk, nodes in enumerate(level_nodes):
        for i in nodes:
            pair = (rk, comp_of[i])
            if pair not in seen:
                seen.add(pair)
                level_comps[rk].append(comp_of[i])

    out_edges: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for u, v, w in edges:
        if comp_of[u] != comp_of[v]:
            out_edges[u].append((v, w))

    return level_nodes, level_comps, comps, out_edges


class FlowSystem:
    """一份厘米格网上的流向约束图，可反复用于不同误差限 / 锁定组合。"""

    __slots__ = (
        "rows", "cols", "n", "level_nodes", "level_comps", "comps", "out_edges",
    )

    def __init__(self, grid: list[list[int]]) -> None:
        self.rows, self.cols = len(grid), len(grid[0])
        self.n = self.rows * self.cols
        (
            self.level_nodes,
            self.level_comps,
            self.comps,
            self.out_edges,
        ) = _build_graph(grid)

    def _sweep(self, state: list[list[int]], labels: int) -> None:
        """从高到低做一次拓扑松弛（就地修改 state）。

        每个等级内先取同值分量内标签的最小值（满足 D 相等），再把出边
        推给更低等级。所有剩余有向边严格跨等级，所以单次扫描即得最短路。

        ``labels`` 只取 1 或 11：前者是普通求解，后者是最小误差扫描。
        两个分支各自内联，避免热循环里额外的循环开销。
        """
        if labels not in (1, 11):
            raise ValueError("labels 只支持 1 或 11")
        level_nodes = self.level_nodes
        level_comps = self.level_comps
        comps = self.comps
        out_edges = self.out_edges
        wide = labels > 1
        for rk in range(len(level_nodes) - 1, -1, -1):
            for ci in level_comps[rk]:
                members = comps[ci]
                if len(members) > 1:
                    first = state[members[0]]
                    mins = first[:]
                    if wide:
                        for u in members[1:]:
                            su = state[u]
                            for t in range(labels):
                                if su[t] < mins[t]:
                                    mins[t] = su[t]
                    else:
                        for u in members[1:]:
                            v = state[u][0]
                            if v < mins[0]:
                                mins[0] = v
                    if mins != first:
                        first[:] = mins
                    for u in members[1:]:
                        su = state[u]
                        if su != mins:
                            su[:] = mins
            if wide:
                for u in level_nodes[rk]:
                    du = state[u]
                    d0, d1, d2, d3, d4 = du[0], du[1], du[2], du[3], du[4]
                    d5, d6, d7, d8, d9, d10 = du[5], du[6], du[7], du[8], du[9], du[10]
                    for v, w in out_edges[u]:
                        dv = state[v]
                        c0 = d0 + w
                        if c0 < dv[0]:
                            dv[0] = c0
                        c1 = d1 + w
                        if c1 < dv[1]:
                            dv[1] = c1
                        c2 = d2 + w
                        if c2 < dv[2]:
                            dv[2] = c2
                        c3 = d3 + w
                        if c3 < dv[3]:
                            dv[3] = c3
                        c4 = d4 + w
                        if c4 < dv[4]:
                            dv[4] = c4
                        c5 = d5 + w
                        if c5 < dv[5]:
                            dv[5] = c5
                        c6 = d6 + w
                        if c6 < dv[6]:
                            dv[6] = c6
                        c7 = d7 + w
                        if c7 < dv[7]:
                            dv[7] = c7
                        c8 = d8 + w
                        if c8 < dv[8]:
                            dv[8] = c8
                        c9 = d9 + w
                        if c9 < dv[9]:
                            dv[9] = c9
                        c10 = d10 + w
                        if c10 < dv[10]:
                            dv[10] = c10
            else:
                for u in level_nodes[rk]:
                    du = state[u]
                    du0 = du[0]
                    for v, w in out_edges[u]:
                        cand = du0 + w
                        dv = state[v]
                        if cand < dv[0]:
                            dv[0] = cand

    # -- 给定误差限求解 --------------------------------------------------- #
    def solve(
        self,
        grid: list[list[int]],
        locks: dict[tuple[int, int], int],
        error_cm: int,
    ) -> list[list[int]] | None:
        """给定误差限求一组最大可行分米解；不可行返回 None。"""
        cols, n = self.cols, self.n
        lock_flat: dict[int, int] = {}
        for (r, c), value in locks.items():
            if not (0 <= r < self.rows and 0 <= c < self.cols):
                return None
            lock_flat[r * cols + c] = value

        state: list[list[int]] = [[0] for _ in range(n)]
        lower: list[int] = [0] * n
        for i in range(n):
            h = grid[i // cols][i % cols]
            lo = _low_dm(h, error_cm)
            hi = _high_dm(h, error_cm)
            if i in lock_flat:
                value = lock_flat[i]
                if not isinstance(value, int) or not lo <= value <= hi:
                    return None
                lower[i] = value
                state[i][0] = value
            else:
                if lo > hi:
                    return None
                lower[i] = lo
                state[i][0] = hi

        self._sweep(state, 1)
        dm_flat = [s[0] for s in state]
        for i in range(n):
            if dm_flat[i] < lower[i]:
                return None
        dm = [[dm_flat[r * cols + c] for c in range(cols)] for r in range(self.rows)]
        verify(grid, dm, locks, error_cm)
        return dm

    # -- 精确最小误差 ----------------------------------------------------- #
    def min_error(
        self,
        grid: list[list[int]],
        locks: dict[tuple[int, int], int],
    ) -> int | None:
        """精确最小可行整数厘米误差；锁定与流向硬冲突时返回 None。

        一次「11 标签」拓扑扫描（10 个 E mod 10 余数 + 1 个锁定常数标签）
        后，只需一遍 O(10n) 的逐格汇总即可得到全部余数所需的下界，不针对
        任何候选 E 重跑全图。推导见模块文档。
        """
        cols, n = self.cols, self.n
        lock_flat: dict[int, int] = {}
        for (r, c), value in locks.items():
            if not (0 <= r < self.rows and 0 <= c < self.cols):
                return None
            if not isinstance(value, int):
                raise ValidationError("锁定值必须是整数分米")
            lock_flat[r * cols + c] = value

        LOCK_LABEL = 10
        labels = 11
        state: list[list[float]] = [[INF] * labels for _ in range(n)]
        for i in range(n):
            h = grid[i // cols][i % cols]
            if i in lock_flat:
                state[i][LOCK_LABEL] = lock_flat[i]
            else:
                row = state[i]
                for e in range(10):
                    # E=10k+e 时非锁定格上界 = k + floor((h+e)/10)
                    row[e] = (h + e) // 10

        self._sweep(state, labels)

        # 常数标签 L 与误差无关：若锁定固定值经约束路径低于自身，
        # 说明两条锁定互相矛盾，任何误差限都无解。
        for i, value in lock_flat.items():
            if state[i][LOCK_LABEL] < value:
                return None

        # 一遍逐格汇总，同时对 10 个余数累积所需 k 的下界。
        need_k = [0] * 10
        for i in range(n):
            h = grid[i // cols][i % cols]
            row = state[i]
            locked = lock_flat.get(i)
            cv = row[LOCK_LABEL]
            cv_ok = cv != INF
            if locked is not None:
                # 锁定格：|10L-h| <= 10k+e 的两端 + 上游上界。
                for e in range(10):
                    a = (h + e) // 10          # floor((h+e)/10)
                    c = -((e - h) // 10)       # ceil((h-e)/10)
                    cand = locked - a
                    if c - locked > cand:
                        cand = c - locked
                    av = row[e]
                    if av != INF and locked - av > cand:
                        cand = locked - av
                    if cand > need_k[e]:
                        need_k[e] = cand
            else:
                for e in range(10):
                    c = -((e - h) // 10)
                    cand = 0
                    av = row[e]
                    if av != INF:
                        # low_i = c-k <= upper = k+A  =>  2k >= c-A
                        cand = -((-(c - av)) // 2)   # ceil((c-A)/2)
                    if cv_ok:
                        constant = c - cv
                        if constant > cand:
                            cand = constant
                    if cand > need_k[e]:
                        need_k[e] = cand

        best: int | None = None
        for e in range(10):
            k = need_k[e] if need_k[e] > 0 else 0
            candidate = 10 * k + e
            if best is None or candidate < best:
                best = candidate
        return best


# --------------------------------------------------------------------------- #
# 缓存：同一份厘米格网复用流向约束图
# --------------------------------------------------------------------------- #
_CACHE: "OrderedDict[tuple, FlowSystem]" = OrderedDict()


def get_system(grid: list[list[int]]) -> FlowSystem:
    """按厘米格网内容获取（并缓存）流向约束图。"""
    key = (len(grid), len(grid[0]), tuple(map(tuple, grid)))
    system = _CACHE.get(key)
    if system is not None:
        _CACHE.move_to_end(key)
        return system
    system = FlowSystem(grid)
    _CACHE[key] = system
    if len(_CACHE) > SYSTEM_CACHE_SIZE:
        _CACHE.popitem(last=False)
    return system


def clear_cache() -> None:
    _CACHE.clear()
    _MIN_ERROR_CACHE.clear()


# 最小误差结果缓存：同一厘米格网 + 同一组锁定重复试算（如页面刷新、
# 重复点“立即试算”）时直接命中，不再扫图。键包含完整格网与锁定，
# 任何改锁/改底图都会生成新键，不存在过期结果。
_MIN_ERROR_CACHE: "OrderedDict[tuple, int | None]" = OrderedDict()
MIN_ERROR_CACHE_SIZE = 128
_MISS = object()


def _min_error_key(grid: list[list[int]], locks: dict[tuple[int, int], int]) -> tuple:
    return (
        len(grid),
        len(grid[0]),
        tuple(map(tuple, grid)),
        tuple(sorted(locks.items())),
    )


# --------------------------------------------------------------------------- #
# 自检 / 对外入口
# --------------------------------------------------------------------------- #
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
    if not isinstance(error_cm, int) or error_cm < 0:
        raise ValidationError("最大允许误差必须是非负整数（厘米）")
    return get_system(grid).solve(grid, locks, error_cm)


def min_feasible_error(
    grid: list[list[int]], locks: dict[tuple[int, int], int]
) -> int | None:
    """精确最小可行误差；锁定与流向根本冲突时返回 None。

    结果按（厘米格网, 锁定集合）缓存；改误差限不影响该值，改锁或改底图
    会生成新键。
    """
    key = _min_error_key(grid, locks)
    cached = _MIN_ERROR_CACHE.get(key, _MISS)
    if cached is not _MISS:
        _MIN_ERROR_CACHE.move_to_end(key)
        return cached
    value = get_system(grid).min_error(grid, locks)
    _MIN_ERROR_CACHE[key] = value
    if len(_MIN_ERROR_CACHE) > MIN_ERROR_CACHE_SIZE:
        _MIN_ERROR_CACHE.popitem(last=False)
    return value


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

    dm = get_system(grid).solve(grid, locks, requested_error_cm)
    assert dm is not None  # 最小误差已可行，更大的请求误差必可行
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
