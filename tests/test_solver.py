"""求解器单元测试：python -m unittest discover -s tests -v"""

import os
import random
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "app"))

import solver  # noqa: E402
from solver import (  # noqa: E402  # noqa: E402
    FlowSystem,
    ValidationError,
    convert,
    flow_sets,
    get_system,
    min_feasible_error,
    naive_round,
    parse_csv,
    solve_at,
    to_csv,
)


class TestParse(unittest.TestCase):
    def test_basic(self):
        grid = parse_csv("1,2\n3,4\n")
        self.assertEqual(grid, [[1, 2], [3, 4]])

    def test_ragged_rejected(self):
        with self.assertRaises(ValidationError):
            parse_csv("1,2,3\n4,5\n")

    def test_non_integer_rejected(self):
        with self.assertRaises(ValidationError):
            parse_csv("1,2.5\n3,4\n")

    def test_empty_rejected(self):
        with self.assertRaises(ValidationError):
            parse_csv("\n  \n")

    def test_size_limit(self):
        big = "\n".join(",".join(["0"] * 33) for _ in range(33))
        with self.assertRaises(ValidationError):
            parse_csv(big)
        ok = "\n".join(",".join(["0"] * 32) for _ in range(32))
        self.assertEqual(len(parse_csv(ok)), 32)

    def test_roundtrip(self):
        grid = [[-5, 0], [10, 15]]
        self.assertEqual(parse_csv(to_csv(grid)), grid)


class TestFlow(unittest.TestCase):
    def test_ties_and_no_downslope(self):
        # 四邻场景：(0,0)=10，右邻 8 与下邻 9，唯一最低为右邻；
        # (0,1)=8 是坑点无下坡；(1,1)=8 邻格都不低于它，也无下坡
        grid = [[10, 8], [9, 8]]
        flows = flow_sets(grid)
        self.assertEqual(flows[0][0], [(0, 1)])
        self.assertEqual(flows[0][1], [])  # 邻格 10、8，无严格更低
        self.assertEqual(flows[1][1], [])  # 最低点，无下坡
        self.assertEqual(flows[1][0], [(1, 1)])

    def test_four_neighbor_tie(self):
        # 十字并列：中心 10，上下左右均为 8 -> 四个流向全部保留
        grid = [
            [9, 8, 9],
            [8, 10, 8],
            [9, 8, 9],
        ]
        flows = flow_sets(grid)
        self.assertEqual(
            sorted(flows[1][1]), [(0, 1), (1, 0), (1, 2), (2, 1)]
        )

    def test_equal_neighbor_not_flow(self):
        grid = [[5, 5], [5, 5]]
        self.assertTrue(all(not cell for row in flow_sets(grid) for cell in row))


class TestConvert(unittest.TestCase):
    def setUp(self):
        # 含缓坡、并列、局部平底的 6x6 厘米底图
        path = os.path.join(os.path.dirname(__file__), "..", "sample", "slope_cm.csv")
        with open(path, encoding="utf-8") as fh:
            self.grid = parse_csv(fh.read())

    def test_naive_rounding_breaks_flow(self):
        # 旧程序逐格取整：至少有一处流向被破坏（演示问题存在）
        self.assertNotEqual(flow_sets(naive_round(self.grid)), flow_sets(self.grid))

    def test_conversion_preserves_flow_within_tolerance(self):
        for limit in (0, 4, 9, 20):
            with self.subTest(limit=limit):
                result = convert(self.grid, {}, limit)
                if result["feasible"]:
                    self.assertEqual(flow_sets(result["grid_dm"]), flow_sets(self.grid))
                    for r, row in enumerate(self.grid):
                        for c, h in enumerate(row):
                            self.assertLessEqual(
                                abs(10 * result["grid_dm"][r][c] - h), limit
                            )

    def test_zero_error_feasible_only_when_multiple_of_ten_compatible(self):
        # 全是 10 的倍数时 0 误差必可行
        grid = [[100, 90], [90, 80]]
        result = convert(grid, {}, 0)
        self.assertTrue(result["feasible"])
        self.assertEqual(result["grid_dm"], [[10, 9], [9, 8]])

    def test_infeasible_tight_reports_minimum(self):
        # 样例底图在 0 cm 误差下必然不可行（含非 10 倍数且有坡度要求）
        result = convert(self.grid, {}, 0)
        self.assertFalse(result["feasible"])
        self.assertEqual(result["reason"], "error_too_tight")
        self.assertGreaterEqual(result["min_error_cm"], 1)
        # 给出的最小误差处必须确实可行
        again = convert(self.grid, {}, result["min_error_cm"])
        self.assertTrue(again["feasible"])

    def test_lock_feasible(self):
        locks = {(5, 5): 8}  # 沟口锁定 8 dm（真值 75 cm，误差 5 cm）
        result = convert(self.grid, locks, 9)
        self.assertTrue(result["feasible"])
        self.assertEqual(result["grid_dm"][5][5], 8)
        self.assertEqual(flow_sets(result["grid_dm"]), flow_sets(self.grid))

    def test_lock_hard_conflict(self):
        # 沟口 (5,5) 是坑点、左邻 (5,4)=100 cm 必须比它高；
        # 把沟口锁到 17 dm、左邻锁到 16 dm，高低颠倒，任何误差都无解
        locks = {(5, 5): 17, (5, 4): 16}
        result = convert(self.grid, locks, 10_000)
        self.assertFalse(result["feasible"])
        self.assertEqual(result["reason"], "hard_conflict")

    def test_lock_reverses_minimum_error(self):
        # 加严格锁定后最小可行误差应不小于无锁定时
        base_min = convert(self.grid, {}, 0)["min_error_cm"]
        locked_min = convert(self.grid, {(0, 0): 33}, 0)
        if locked_min["feasible"]:
            self.assertGreaterEqual(locked_min["min_error_cm"], base_min)
        else:
            self.assertEqual(locked_min["reason"], "error_too_tight")
            self.assertGreaterEqual(locked_min["min_error_cm"], base_min)

    def test_random_grids_invariants(self):
        rng = random.Random(20261123)
        for _ in range(30):
            rows = rng.randint(2, 10)
            cols = rng.randint(2, 10)
            grid = [[rng.randint(0, 500) for _ in range(cols)] for _ in range(rows)]
            result = convert(grid, {}, 50)
            self.assertTrue(result["feasible"], "随机底图在宽误差下应可行")
            self.assertEqual(flow_sets(result["grid_dm"]), flow_sets(grid))
            for r, row in enumerate(grid):
                for c, h in enumerate(row):
                    self.assertLessEqual(abs(10 * result["grid_dm"][r][c] - h), 50)

    def test_negative_elevations(self):
        # 负高程：该小格网三级严格排序需要更宽区间，20 cm 误差下可行
        grid = [[-3, -7], [-5, -12]]
        result = convert(grid, {}, 20)
        self.assertTrue(result["feasible"])
        self.assertEqual(flow_sets(result["grid_dm"]), flow_sets(grid))

    def test_solve_at_returns_none_on_negative_cycle(self):
        # 直接构造不可能的锁定：同一格既 <=5 又 >=6
        grid = [[50, 40], [40, 30]]
        # 通过两次锁定同一格无法构造（字典覆盖），改用边界外锁值制造矛盾
        # 锁 (0,0)=0 且 (0,1)=5：要求 D(0,1)<D(0,0) 与锁定方向矛盾
        result = convert(grid, {(0, 0): 0, (0, 1): 5}, 10_000)
        self.assertFalse(result["feasible"])

    def test_32x32_runtime(self):
        rng = random.Random(7)
        grid = [[rng.randint(0, 1000) for _ in range(32)] for _ in range(32)]
        result = convert(grid, {}, 15)
        self.assertTrue(result["feasible"])
        self.assertEqual(flow_sets(result["grid_dm"]), flow_sets(grid))

    def test_bad_error_value(self):
        with self.assertRaises(ValidationError):
            convert(self.grid, {}, -1)


class TestExactMinError(unittest.TestCase):
    """最小误差精确性、复用与改锁后重算误差。"""

    def setUp(self):
        path = os.path.join(os.path.dirname(__file__), "..", "sample", "slope_cm.csv")
        with open(path, encoding="utf-8") as fh:
            self.grid = parse_csv(fh.read())

    def _reference_min(self, grid, locks):
        """与精确算法无关的独立参考：按 E 从小到大逐个校验。"""
        for limit in range(0, 4000):
            if solve_at(grid, locks, limit) is not None:
                return limit
        return None

    def test_matches_sequential_search(self):
        import random

        rng = random.Random(20261123)
        cases = [(self.grid, {}), (self.grid, {(5, 5): 8}), (self.grid, {(0, 0): 33})]
        for _ in range(12):
            rows, cols = 3, 3
            grid = [[rng.randint(-40, 60) for _ in range(cols)] for _ in range(rows)]
            locks = {}
            for _ in range(rng.randint(1, 3)):
                locks[(rng.randrange(rows), rng.randrange(cols))] = rng.randint(-6, 8)
            cases.append((grid, locks))
        for grid, locks in cases:
            with self.subTest(grid=grid[:1], locks=locks):
                exact = min_feasible_error(grid, locks)
                self.assertEqual(exact, self._reference_min(grid, locks))

    def test_far_lock_reports_exact_value(self):
        # 锁定值远高于测绘高程：最小误差必须至少覆盖锁定自身偏差。
        locks = {(0, 0): 100}  # 该格 317 cm，锁定 100 dm = 1000 cm，偏差 683 cm
        exact = min_feasible_error(self.grid, locks)
        self.assertEqual(exact, 683)
        self.assertEqual(convert(self.grid, locks, 682)["reason"], "error_too_tight")
        self.assertEqual(convert(self.grid, locks, 682)["min_error_cm"], 683)
        self.assertTrue(convert(self.grid, locks, 683)["feasible"])

    def test_far_lock_below_reports_exact_value(self):
        locks = {(0, 0): -50}  # 317 cm 对 -500 cm，偏差 817 cm
        exact = min_feasible_error(self.grid, locks)
        self.assertIsNotNone(exact)
        self.assertGreaterEqual(exact, 817)
        self.assertTrue(convert(self.grid, locks, exact)["feasible"])
        self.assertFalse(convert(self.grid, locks, exact - 1)["feasible"])

    def test_conflict_detected_for_all_errors(self):
        locks = {(5, 5): 17, (5, 4): 16}
        for limit in (0, 50, 10_000, 10**6):
            result = convert(self.grid, locks, limit)
            self.assertEqual(result["reason"], "hard_conflict")

    def test_reuses_system_and_recomputes_error_after_change(self):
        # 同一份厘米格网：改锁复用同一张约束图，但最小误差按新锁定重算。
        s1 = get_system(self.grid)
        s2 = get_system([row[:] for row in self.grid])
        self.assertIs(s1, s2)
        base = min_feasible_error(self.grid, {})
        far = min_feasible_error(self.grid, {(0, 0): 100})
        self.assertNotEqual(base, far)
        self.assertEqual(far, 683)

    def test_correction_same_flow_new_height_recomputes(self):
        # 更正后流向结构可能相同，但误差必须按新的厘米高程计算。
        corrected = [row[:] for row in self.grid]
        corrected[0][0] += 40
        base = convert(self.grid, {}, 9)
        upd = convert(corrected, {}, 9)
        self.assertTrue(base["feasible"] and upd["feasible"])
        self.assertEqual(flow_sets(corrected), flow_sets(upd["grid_dm"]))
        self.assertEqual(upd["grid_dm"][0][0], (corrected[0][0] + 5) // 10)
        # 新格网的最小误差独立重算，与旧值的可行性判断一致
        exact = min_feasible_error(corrected, {})
        self.assertTrue(convert(corrected, {}, exact)["feasible"])


class TestSweepCost(unittest.TestCase):
    """最小误差只做一次全图扫描，不随候选误差个数增长。"""

    def test_min_error_sweeps_once(self):
        path = os.path.join(os.path.dirname(__file__), "..", "sample", "slope_cm.csv")
        with open(path, encoding="utf-8") as fh:
            grid = parse_csv(fh.read())

        class Counting(FlowSystem):
            def __init__(self, g):
                super().__init__(g)
                self.sweeps = 0

            def _sweep(self, state, labels):
                self.sweeps += 1
                return super()._sweep(state, labels)

        system = Counting(grid)
        system.min_error(grid, {})
        self.assertEqual(system.sweeps, 1)
        system.min_error(grid, {(5, 5): 8})
        self.assertEqual(system.sweeps, 2)  # 改锁只再扫一次

    def test_repeated_trial_uses_result_cache(self):
        path = os.path.join(os.path.dirname(__file__), "..", "sample", "slope_cm.csv")
        with open(path, encoding="utf-8") as fh:
            grid = parse_csv(fh.read())

        class Counting(FlowSystem):
            def __init__(self, g):
                super().__init__(g)
                self.sweeps = 0

            def _sweep(self, state, labels):
                self.sweeps += 1
                return super()._sweep(state, labels)

        solver.clear_cache()
        first = Counting(grid)
        # 直接把 Counting 实例放入系统缓存，验证重复试算命中结果缓存
        original_get = solver.get_system
        solver.get_system = lambda g: first
        try:
            a = min_feasible_error(grid, {})
            b = min_feasible_error(grid, {})
            self.assertEqual(first.sweeps, 1)  # 第二次直接命中最小误差结果缓存
            # 改锁后必须重新扫描并更新结果
            c = min_feasible_error(grid, {(5, 5): 11})
        finally:
            solver.get_system = original_get
        self.assertEqual(a, b)
        self.assertEqual(first.sweeps, 2)
        self.assertNotEqual(a, c)


if __name__ == "__main__":
    unittest.main()
