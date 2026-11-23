"""Flask + SQLite 底图转换网页应用。

数据模型（SQLite）
==================
* ``projects``     一个底图转换任务（固定尺寸）
* ``snapshots``    每次成功/失败的转换记录；最新一条 status='ok' 的快照
                   即“上一份可下载的底图”，失败时不会被覆盖
* ``locks``        分米锁定值（沟口等），更正上传时继续沿用
"""

from __future__ import annotations

import io
import json
import os
import sqlite3
from datetime import datetime, timezone

from flask import (
    Flask,
    Response,
    flash,
    g,
    redirect,
    render_template,
    request,
    url_for,
)

from .solver import (
    MAX_DIM,
    ValidationError,
    convert,
    flow_sets,
    parse_csv,
    to_csv,
)

DB_PATH = os.environ.get("DB_PATH", os.path.join(os.path.dirname(__file__), "local_data", "drainage.db"))

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-change-in-prod")

ARROW = {(-1, 0): "↑", (1, 0): "↓", (0, -1): "←", (0, 1): "→"}


# --------------------------------------------------------------------------- #
# 数据库
# --------------------------------------------------------------------------- #
def get_db() -> sqlite3.Connection:
    if "db" not in g:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        g.db = conn
    return g.db


@app.teardown_appcontext
def close_db(_exc: object) -> None:
    conn = g.pop("db", None)
    if conn is not None:
        conn.close()


def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS projects (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            name       TEXT NOT NULL,
            rows       INTEGER NOT NULL,
            cols       INTEGER NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS snapshots (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id   INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            kind         TEXT NOT NULL,            -- initial / correction
            source_name  TEXT,
            input_cm     TEXT NOT NULL,            -- JSON 厘米格网
            error_limit  INTEGER NOT NULL,
            min_error    INTEGER,
            status       TEXT NOT NULL,            -- ok / tight / conflict
            grid_dm      TEXT,                     -- JSON 分米格网，失败时为 NULL
            errors       TEXT,                     -- JSON 逐格误差
            created_at   TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS locks (
            project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            row_idx    INTEGER NOT NULL,
            col_idx    INTEGER NOT NULL,
            dm_value   INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (project_id, row_idx, col_idx)
        );
        """
    )
    conn.commit()
    conn.close()


# --------------------------------------------------------------------------- #
# 辅助
# --------------------------------------------------------------------------- #
def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_locks(conn: sqlite3.Connection, project_id: int) -> dict[tuple[int, int], int]:
    rows = conn.execute(
        "SELECT row_idx, col_idx, dm_value FROM locks WHERE project_id = ?",
        (project_id,),
    ).fetchall()
    return {(r["row_idx"], r["col_idx"]): r["dm_value"] for r in rows}


def _latest_ok_snapshot(conn: sqlite3.Connection, project_id: int) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM snapshots WHERE project_id = ? AND status = 'ok' "
        "ORDER BY id DESC LIMIT 1",
        (project_id,),
    ).fetchone()


def _latest_snapshot(conn: sqlite3.Connection, project_id: int) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM snapshots WHERE project_id = ? ORDER BY id DESC LIMIT 1",
        (project_id,),
    ).fetchone()


def _read_upload(file_storage) -> str:
    if file_storage is None or not file_storage.filename:
        raise ValidationError("请选择要上传的 CSV 文件")
    raw = file_storage.read()
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValidationError("CSV 需为 UTF-8 编码文本") from exc


def _parse_error_limit(raw: str) -> int:
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError) as exc:
        raise ValidationError("最大允许误差必须是整数厘米") from exc
    if value < 0:
        raise ValidationError("最大允许误差不能为负数")
    return value


def _arrow_table(grid: list[list[int]]) -> list[list[str]]:
    flows = flow_sets(grid)
    table: list[list[str]] = []
    for r, row in enumerate(flows):
        line = []
        for c, targets in enumerate(row):
            if not targets:
                line.append("·")
            else:
                line.append("".join(ARROW[(nr - r, nc - c)] for nr, nc in targets))
        table.append(line)
    return table


def _dashboard_context(conn: sqlite3.Connection, project: sqlite3.Row) -> dict:
    snaps = conn.execute(
        "SELECT * FROM snapshots WHERE project_id = ? ORDER BY id DESC LIMIT 10",
        (project["id"],),
    ).fetchall()
    lock_rows = conn.execute(
        "SELECT * FROM locks WHERE project_id = ? ORDER BY row_idx, col_idx",
        (project["id"],),
    ).fetchall()

    latest = _latest_snapshot(conn, project["id"])
    ok = _latest_ok_snapshot(conn, project["id"])
    # 可下载底图始终取最新成功快照；失败尝试不会覆盖它。
    source = ok if ok is not None else latest
    input_cm = json.loads(source["input_cm"])
    grid_dm = json.loads(source["grid_dm"]) if source["grid_dm"] else None
    errors = json.loads(source["errors"]) if source["errors"] else None

    return {
        "project": project,
        "snapshots": snaps,
        "locks": lock_rows,
        "lock_cells": _load_locks(conn, project["id"]),
        "latest": latest,
        "downloadable": ok,
        "input_cm": input_cm,
        "grid_dm": grid_dm,
        "errors": errors,
        "flow_cm": _arrow_table(input_cm),
        "flow_dm": _arrow_table(grid_dm) if grid_dm is not None else None,
        "max_dim": MAX_DIM,
    }


# --------------------------------------------------------------------------- #
# 路由
# --------------------------------------------------------------------------- #
@app.route("/healthz")
def healthz():
    return {"status": "ok"}


@app.route("/")
def index():
    conn = get_db()
    projects = conn.execute(
        """
        SELECT p.*,
               (SELECT COUNT(*) FROM locks l WHERE l.project_id = p.id) AS lock_count,
               (SELECT status FROM snapshots s WHERE s.project_id = p.id
                ORDER BY id DESC LIMIT 1) AS last_status
        FROM projects p ORDER BY id DESC
        """
    ).fetchall()
    return render_template("index.html", projects=projects, max_dim=MAX_DIM)


@app.route("/projects", methods=["POST"])
def create_project():
    conn = get_db()
    name = (request.form.get("name") or "").strip() or "未命名底图"
    try:
        text = _read_upload(request.files.get("csv_file"))
        grid = parse_csv(text)
        error_limit = _parse_error_limit(request.form.get("error_limit", ""))
    except ValidationError as exc:
        flash(str(exc), "error")
        return redirect(url_for("index"))

    cur = conn.execute(
        "INSERT INTO projects (name, rows, cols, created_at) VALUES (?, ?, ?, ?)",
        (name, len(grid), len(grid[0]), _now()),
    )
    project_id = cur.lastrowid

    result = convert(grid, {}, error_limit)
    if result["feasible"]:
        conn.execute(
            "INSERT INTO snapshots (project_id, kind, source_name, input_cm, error_limit,"
            " min_error, status, grid_dm, errors, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                project_id, "initial", request.files["csv_file"].filename,
                json.dumps(grid), error_limit, result["min_error_cm"], "ok",
                json.dumps(result["grid_dm"]), json.dumps(result["errors"]), _now(),
            ),
        )
        flash(
            f"转换成功：实际最大误差 {result['max_abs_error']} cm，"
            f"该底图最小可行误差为 {result['min_error_cm']} cm。",
            "ok",
        )
    else:
        status = "tight" if result["reason"] == "error_too_tight" else "conflict"
        conn.execute(
            "INSERT INTO snapshots (project_id, kind, source_name, input_cm, error_limit,"
            " min_error, status, grid_dm, errors, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                project_id, "initial", request.files["csv_file"].filename,
                json.dumps(grid), error_limit,
                result.get("min_error_cm"), status, None, None, _now(),
            ),
        )
        if status == "tight":
            flash(
                f"误差限内无解。最小可行误差为 {result['min_error_cm']} cm，"
                "请放宽误差后重新转换；锁定与已生成底图均保留。",
                "error",
            )
        else:
            flash("无解：当前锁定值与流向要求无法同时满足（任何误差限都不行）。", "error")
    conn.commit()
    return redirect(url_for("view_project", project_id=project_id))


@app.route("/projects/<int:project_id>")
def view_project(project_id: int):
    conn = get_db()
    project = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
    if project is None:
        flash("底图不存在", "error")
        return redirect(url_for("index"))
    return render_template("project.html", **_dashboard_context(conn, project))


@app.route("/projects/<int:project_id>/convert", methods=["POST"])
def reconvert(project_id: int):
    conn = get_db()
    project = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
    if project is None:
        flash("底图不存在", "error")
        return redirect(url_for("index"))
    try:
        error_limit = _parse_error_limit(request.form.get("error_limit", ""))
    except ValidationError as exc:
        flash(str(exc), "error")
        return redirect(url_for("view_project", project_id=project_id))

    # 重新转换始终基于当前工作底图（最新成功快照的厘米输入）；
    # 若从未成功过，则退回项目最初上传的那份输入。失败尝试不会改变它。
    ok = _latest_ok_snapshot(conn, project_id)
    source = ok if ok is not None else conn.execute(
        "SELECT * FROM snapshots WHERE project_id = ? AND kind = 'initial' "
        "ORDER BY id ASC LIMIT 1", (project_id,)
    ).fetchone()
    grid = json.loads(source["input_cm"])
    locks = _load_locks(conn, project_id)

    result = convert(grid, locks, error_limit)
    if result["feasible"]:
        conn.execute(
            "INSERT INTO snapshots (project_id, kind, source_name, input_cm, error_limit,"
            " min_error, status, grid_dm, errors, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                project_id, "reconvert", None, json.dumps(grid), error_limit,
                result["min_error_cm"], "ok",
                json.dumps(result["grid_dm"]), json.dumps(result["errors"]), _now(),
            ),
        )
        flash(
            f"转换成功：实际最大误差 {result['max_abs_error']} cm，"
            f"最小可行误差 {result['min_error_cm']} cm，锁定值已沿用。",
            "ok",
        )
    else:
        status = "tight" if result["reason"] == "error_too_tight" else "conflict"
        conn.execute(
            "INSERT INTO snapshots (project_id, kind, source_name, input_cm, error_limit,"
            " min_error, status, grid_dm, errors, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                project_id, "reconvert", None, json.dumps(grid), error_limit,
                result.get("min_error_cm"), status, None, None, _now(),
            ),
        )
        if status == "tight":
            flash(
                f"误差限内无解，已保留锁定值和上一份可下载底图。"
                f"最小可行误差为 {result['min_error_cm']} cm。",
                "error",
            )
        else:
            flash(
                "放宽误差仍无解：锁定值与流向要求无法同时满足。"
                "已保留锁定值和上一份可下载底图，请调整或删除锁定。",
                "error",
            )
    conn.commit()
    return redirect(url_for("view_project", project_id=project_id))


@app.route("/projects/<int:project_id>/locks", methods=["POST"])
def add_lock(project_id: int):
    conn = get_db()
    project = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
    if project is None:
        flash("底图不存在", "error")
        return redirect(url_for("index"))
    try:
        row = int(request.form.get("row", "")) - 1  # 页面用 1 基坐标
        col = int(request.form.get("col", "")) - 1
        value = int(request.form.get("dm_value", ""))
    except (TypeError, ValueError):
        flash("行、列、分米值都必须是整数", "error")
        return redirect(url_for("view_project", project_id=project_id))
    if not (0 <= row < project["rows"] and 0 <= col < project["cols"]):
        flash(f"锁定位置超出 {project['rows']}×{project['cols']} 格网范围", "error")
        return redirect(url_for("view_project", project_id=project_id))

    conn.execute(
        "INSERT INTO locks (project_id, row_idx, col_idx, dm_value, created_at) "
        "VALUES (?,?,?,?,?) ON CONFLICT(project_id, row_idx, col_idx) "
        "DO UPDATE SET dm_value = excluded.dm_value",
        (project_id, row, col, value, _now()),
    )
    conn.commit()
    flash(f"已锁定第 {row + 1} 行第 {col + 1} 列 = {value} dm，请重新转换。", "ok")
    return redirect(url_for("view_project", project_id=project_id))


@app.route("/projects/<int:project_id>/locks/<int:row>/<int:col>/delete", methods=["POST"])
def delete_lock(project_id: int, row: int, col: int):
    conn = get_db()
    conn.execute(
        "DELETE FROM locks WHERE project_id = ? AND row_idx = ? AND col_idx = ?",
        (project_id, row, col),
    )
    conn.commit()
    flash("已删除该锁定，请重新转换。", "ok")
    return redirect(url_for("view_project", project_id=project_id))


@app.route("/projects/<int:project_id>/correction", methods=["POST"])
def upload_correction(project_id: int):
    conn = get_db()
    project = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
    if project is None:
        flash("底图不存在", "error")
        return redirect(url_for("index"))
    try:
        text = _read_upload(request.files.get("csv_file"))
        new_grid = parse_csv(text)
        error_limit = _parse_error_limit(request.form.get("error_limit", ""))
    except ValidationError as exc:
        flash(str(exc), "error")
        return redirect(url_for("view_project", project_id=project_id))

    if len(new_grid) != project["rows"] or len(new_grid[0]) != project["cols"]:
        flash(
            f"测绘更正尺寸 {len(new_grid)}×{len(new_grid[0])} 与原底图 "
            f"{project['rows']}×{project['cols']} 不一致，已拒绝。",
            "error",
        )
        return redirect(url_for("view_project", project_id=project_id))

    locks = _load_locks(conn, project_id)
    result = convert(new_grid, locks, error_limit)
    # 无论成败都记录尝试（grid_dm 为 NULL）；可下载底图始终取最新成功快照。
    if result["feasible"]:
        conn.execute(
            "INSERT INTO snapshots (project_id, kind, source_name, input_cm, error_limit,"
            " min_error, status, grid_dm, errors, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                project_id, "correction", request.files["csv_file"].filename,
                json.dumps(new_grid), error_limit, result["min_error_cm"], "ok",
                json.dumps(result["grid_dm"]), json.dumps(result["errors"]), _now(),
            ),
        )
        flash(
            f"测绘更正转换成功：沿用 {len(locks)} 个锁定值，"
            f"实际最大误差 {result['max_abs_error']} cm。",
            "ok",
        )
    else:
        conn.execute(
            "INSERT INTO snapshots (project_id, kind, source_name, input_cm, error_limit,"
            " min_error, status, grid_dm, errors, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                project_id, "correction", request.files["csv_file"].filename,
                json.dumps(new_grid), error_limit,
                result.get("min_error_cm"),
                "tight" if result["reason"] == "error_too_tight" else "conflict",
                None, None, _now(),
            ),
        )
        if result["reason"] == "error_too_tight":
            flash(
                f"更正底图在误差限内无解，已保留锁定和上一份可下载底图。"
                f"最小可行误差为 {result['min_error_cm']} cm。",
                "error",
            )
        else:
            flash(
                "更正底图与锁定值在任何误差限下都无法同时满足流向一致，"
                "已保留锁定和上一份可下载底图。",
                "error",
            )
    conn.commit()
    return redirect(url_for("view_project", project_id=project_id))


@app.route("/projects/<int:project_id>/download")
def download_dm(project_id: int):
    conn = get_db()
    snap = _latest_ok_snapshot(conn, project_id)
    if snap is None:
        flash("当前没有可下载的分米底图", "error")
        return redirect(url_for("view_project", project_id=project_id))
    grid_dm = json.loads(snap["grid_dm"])
    csv_text = to_csv(grid_dm)
    filename = f"basemap_{project_id}_snapshot{snap['id']}_dm.csv"
    return Response(
        csv_text,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.route("/projects/<int:project_id>/delete", methods=["POST"])
def delete_project(project_id: int):
    conn = get_db()
    conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
    conn.commit()
    flash("底图任务已删除", "ok")
    return redirect(url_for("index"))


if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
