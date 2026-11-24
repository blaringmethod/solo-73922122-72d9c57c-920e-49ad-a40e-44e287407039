"""对比最小误差的「全图检查次数」。

两组格网流向结构相同、高差不同：

* A：``sample/slope_cm.csv`` 原始厘米高程；
* B：A 每格 ``3*h + 37``，四邻比较与并列关系不变（流向结构相同），
  高差放大到 3 倍。

对比两种求最小可行整数厘米误差的做法：

* 「逐候选检查」：在 ``[0, upper]`` 上二分候选误差，每个候选都调用一次
  ``solve_at``（一次完整全图求解），统计调用次数；
* 「一次扫描」：``FlowSystem.min_error``，固定一次 11 标签全图扫描 +
  一遍逐格汇总，统计扫描次数。

运行：``python3 scripts/bench_min_error.py``
"""

import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "app"))

import solver  # noqa: E402
from solver import FlowSystem, parse_csv, solve_at  # noqa: E402

SAMPLE = os.path.join(ROOT, "sample", "slope_cm.csv")


def candidate_checks(grid, locks):
    """逐候选检查：每个候选误差都做一次全图求解，返回 (最小误差, 次数)。"""
    flat = [v for row in grid for v in row]
    big = max(
        200,
        10 * (max(flat) - min(flat)) + 100,
        max([abs(10 * L - grid[r][c]) for (r, c), L in locks.items()] or [0]) + 100,
    )
    calls = 0

    def feasible(limit):
        nonlocal calls
        calls += 1
        return solve_at(grid, locks, limit) is not None

    if not feasible(big):
        return None, calls
    lo, hi = 0, big
    while lo < hi:
        mid = (lo + hi) // 2
        if feasible(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo, calls + 1


class CountingSystem(FlowSystem):
    def __init__(self, grid):
        super().__init__(grid)
        self.sweeps = 0

    def _sweep(self, state, labels):
        self.sweeps += 1
        super()._sweep(state, labels)


def one_scan(grid, locks):
    """一次扫描：返回 (最小误差, 全图扫描次数, 耗时毫秒)。"""
    system = CountingSystem(grid)
    started = time.perf_counter()
    minimum = system.min_error(grid, locks)
    elapsed = (time.perf_counter() - started) * 1000
    return minimum, system.sweeps, elapsed


def main():
    with open(SAMPLE, encoding="utf-8") as fh:
        grid_a = parse_csv(fh.read())
    grid_b = [[3 * h + 37 for h in row] for row in grid_a]

    same_flow = solver.flow_sets(grid_a) == solver.flow_sets(grid_b)
    print(f"流向结构相同：{same_flow}")
    print()
    header = (
        f"{'格网':<22}{'锁定':<14}"
        f"{'逐候选检查':>12}{'一次扫描':>10}{'最小误差cm':>12}{'扫描耗时ms':>12}"
    )
    print(header)
    print("-" * len(header))
    for name, grid in (("A 原始高差", grid_a), ("B ×3+37 高差放大", grid_b)):
        for label, locks in (("无锁定", {}), ("(1,1)=33dm", {(0, 0): 33})):
            ref, checks = candidate_checks(grid, locks)
            minimum, sweeps, elapsed = one_scan(grid, locks)
            assert ref == minimum, (ref, minimum)
            print(
                f"{name:<22}{label:<14}"
                f"{checks:>12}{sweeps:>10}{minimum:>12}{elapsed:>12.2f}"
            )
    print()
    print("「逐候选检查」每次调用 solve_at 都重跑全图；「一次扫描」固定 1 次。")
    print("改误差限或增删/修改锁定后重新试算，仍复用已建约束图，同样只需 1 次扫描。")


if __name__ == "__main__":
    main()
