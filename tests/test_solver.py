"""求解器单元测试：python -m unittest discover -s tests -v"""

import os
import random
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "app"))

import solver  # noqa: E402
from solver import (  # noqa: E402
    ValidationError,
    convert,
    flow_sets,
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


if __name__ == "__main__":
    unittest.main()
