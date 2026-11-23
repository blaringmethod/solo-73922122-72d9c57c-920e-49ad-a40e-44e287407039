"""Flask + SQLite 端到端测试。"""

import io
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app import app as webapp  # noqa: E402  (包 app，入口 app/app.py)
from app.solver import flow_sets, parse_csv  # noqa: E402

SAMPLE = os.path.join(os.path.dirname(__file__), "..", "sample", "slope_cm.csv")
CORRECTION = os.path.join(os.path.dirname(__file__), "..", "sample", "correction_cm.csv")


def _upload(client, path, error_limit, name="demo", url=None):
    with open(path, "rb") as fh:
        data = {"name": name, "error_limit": str(error_limit),
                "csv_file": (io.BytesIO(fh.read()), os.path.basename(path))}
        return client.post(url or "/projects", data=data, follow_redirects=True)


class WebFlowTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        webapp.DB_PATH = os.path.join(self.tmp.name, "test.db")
        webapp.app.config.update(TESTING=True)
        webapp.init_db()
        self.client = webapp.app.test_client()

    def tearDown(self):
        self.tmp.cleanup()

    def test_full_lifecycle(self):
        # 1) 初次上传，9 cm 误差 -> 成功，出现下载链接
        page = _upload(self.client, SAMPLE, 9)
        self.assertEqual(page.status_code, 200)
        self.assertIn("转换成功", page.get_data(as_text=True))
        project_url = page.request.path
        pid = int(project_url.rstrip("/").split("/")[-1])

        # 2) 下载分米 CSV，流向与厘米输入一致
        dl = self.client.get(f"/projects/{pid}/download")
        self.assertEqual(dl.status_code, 200)
        dm_grid = parse_csv(dl.get_data(as_text=True))
        with open(SAMPLE, encoding="utf-8") as fh:
            cm_grid = parse_csv(fh.read())
        self.assertEqual(flow_sets(dm_grid), flow_sets(cm_grid))

        # 3) 锁定沟口 (6,6)=8 dm 后重新转换，锁定被遵守
        resp = self.client.post(
            f"/projects/{pid}/locks",
            data={"row": "6", "col": "6", "dm_value": "8"},
            follow_redirects=True,
        )
        self.assertEqual(resp.status_code, 200)
        resp = self.client.post(
            f"/projects/{pid}/convert", data={"error_limit": "9"},
            follow_redirects=True,
        )
        self.assertIn("转换成功", resp.get_data(as_text=True))
        dl = self.client.get(f"/projects/{pid}/download")
        self.assertEqual(parse_csv(dl.get_data(as_text=True))[5][5], 8)

        # 4) 上传同尺寸测绘更正，锁定沿用
        page = _upload(
            self.client, CORRECTION, 9,
            url=f"/projects/{pid}/correction",
        )
        self.assertIn("测绘更正转换成功", page.get_data(as_text=True))
        dl = self.client.get(f"/projects/{pid}/download")
        new_dm = parse_csv(dl.get_data(as_text=True))
        self.assertEqual(new_dm[5][5], 8)
        with open(CORRECTION, encoding="utf-8") as fh:
            new_cm = parse_csv(fh.read())
        self.assertEqual(flow_sets(new_dm), flow_sets(new_cm))

        # 5) 拒绝尺寸不一致的更正
        wrong = io.BytesIO(b"1,2\n3,4\n")
        resp = self.client.post(
            f"/projects/{pid}/correction",
            data={"error_limit": "9", "csv_file": (wrong, "wrong.csv")},
            follow_redirects=True,
        )
        self.assertIn("尺寸", resp.get_data(as_text=True))

        # 6) 快照历史里至少 3 条成功记录
        import sqlite3
        conn = sqlite3.connect(webapp.DB_PATH)
        count = conn.execute(
            "SELECT COUNT(*) FROM snapshots WHERE project_id=? AND status='ok'", (pid,)
        ).fetchone()[0]
        conn.close()
        self.assertGreaterEqual(count, 3)

    def test_tight_error_keeps_previous_basemap(self):
        # 先用 20 cm 得到一份成功底图
        page = _upload(self.client, SAMPLE, 20)
        pid = int(page.request.path.rstrip("/").split("/")[-1])
        good_dl = self.client.get(f"/projects/{pid}/download").get_data(as_text=True)

        # 再用 0 cm 重转，必然失败
        resp = self.client.post(
            f"/projects/{pid}/convert", data={"error_limit": "0"},
            follow_redirects=True,
        )
        text = resp.get_data(as_text=True)
        self.assertIn("最小可行误差", text)
        self.assertIn("上一份可下载底图", text)

        # 下载内容仍是旧的成功底图
        self.assertEqual(
            self.client.get(f"/projects/{pid}/download").get_data(as_text=True),
            good_dl,
        )

    def test_hard_conflict_message(self):
        page = _upload(self.client, SAMPLE, 9)
        pid = int(page.request.path.rstrip("/").split("/")[-1])
        # 制造高低颠倒的锁定（沟口与左邻的高低关系被反转）
        self.client.post(f"/projects/{pid}/locks",
                         data={"row": "6", "col": "6", "dm_value": "17"})
        self.client.post(f"/projects/{pid}/locks",
                         data={"row": "6", "col": "5", "dm_value": "16"})
        resp = self.client.post(
            f"/projects/{pid}/convert", data={"error_limit": "9"},
            follow_redirects=True,
        )
        self.assertIn("无法同时满足", resp.get_data(as_text=True))
        # 放宽到极大误差依然冲突
        resp = self.client.post(
            f"/projects/{pid}/convert", data={"error_limit": "100000"},
            follow_redirects=True,
        )
        self.assertIn("无法同时满足", resp.get_data(as_text=True))

    def test_bad_upload_rejected(self):
        resp = self.client.post(
            "/projects",
            data={"name": "x", "error_limit": "9",
                  "csv_file": (io.BytesIO(b"1,2\n3\n"), "bad.csv")},
            follow_redirects=True,
        )
        self.assertIn("不一致", resp.get_data(as_text=True))

    def test_correction_failure_preserves_basemap(self):
        page = _upload(self.client, SAMPLE, 20)
        pid = int(page.request.path.rstrip("/").split("/")[-1])
        good_dl = self.client.get(f"/projects/{pid}/download").get_data(as_text=True)
        # 锁一个苛刻值，再用 0 误差上传更正 -> 失败：新增失败记录但底图不变
        self.client.post(f"/projects/{pid}/locks",
                         data={"row": "1", "col": "1", "dm_value": "33"})
        before_total = self._snapshot_count(pid)
        before_ok = self._ok_snapshot_count(pid)
        resp = _upload(self.client, CORRECTION, 0,
                       url=f"/projects/{pid}/correction")
        self.assertIn("已保留", resp.get_data(as_text=True))
        self.assertEqual(self._snapshot_count(pid), before_total + 1)
        self.assertEqual(self._ok_snapshot_count(pid), before_ok)
        self.assertEqual(
            self.client.get(f"/projects/{pid}/download").get_data(as_text=True),
            good_dl,
        )

    def _snapshot_count(self, pid):
        import sqlite3
        conn = sqlite3.connect(webapp.DB_PATH)
        n = conn.execute("SELECT COUNT(*) FROM snapshots WHERE project_id=?", (pid,)).fetchone()[0]
        conn.close()
        return n

    def _ok_snapshot_count(self, pid):
        import sqlite3
        conn = sqlite3.connect(webapp.DB_PATH)
        n = conn.execute(
            "SELECT COUNT(*) FROM snapshots WHERE project_id=? AND status='ok'", (pid,)
        ).fetchone()[0]
        conn.close()
        return n


if __name__ == "__main__":
    unittest.main()
