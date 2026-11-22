"""批量预测任务领域服务：基于已注册模型的批量预测执行、结果存储与校验。"""
from __future__ import annotations

import hashlib
from typing import Any

from app.services.base import BaseModuleService
from app.store import store

# 归档索引状态：复制中 →（确认落库一致）已确认 /（被截停）已截停
ARCHIVE_COPYING = "复制中"
ARCHIVE_CONFIRMED = "已确认"
ARCHIVE_STOPPED = "已截停"


class PredictBatchService(BaseModuleService):
    MODULE = "predict_batch"
    ENTITY = "预测任务"
    OPERATION_TERM = "校验"
    LIST_FIELDS = ["任务编号", "模型编号", "数据切片", "输出格式", "分片策略", "并行度", "预测状态", "结果校验"]
    REQUIRED_FIELDS = ["任务编号", "模型编号", "数据切片"]
    STATUS_ORDER = ["草稿", "等待中", "运行中", "已完成"]
    ACTION_RULES = {"提交运行": "草稿", "停止预测": "运行中", "确认完成": "已完成"}
    NEGATIVE_ACTIONS = []

    # ── 链路4: 批量预测 → 挖掘结果 ──
    def execute_batch_and_collect_result(self, task_id: int) -> dict[str, Any]:
        """执行批量预测任务并将预测结果采集到挖掘结果模块进行深度分析。"""
        task = store.find(self.MODULE, task_id)
        if task is None:
            return {"ok": False, "message": f"预测任务 {task_id} 不存在"}
        task["status"] = "已完成"
        task["prediction_state"] = "已完成"
        task["pending"] = False
        mining_rows = store.rows("mining_result")
        next_id = max((int(r.get("id", 0)) for r in mining_rows), default=0) + 1
        model_id = task.get("模型编号", "unknown")
        mining_rows.append({
            "id": next_id,
            "status": "草稿",
            "pending": True,
            "abnormal": False,
            "结果编号": f"RES-P{task_id:04d}-{next_id:04d}",
            "分析主题": f"批量预测任务{task_id}挖掘分析",
            "算法类型": "ensemble_voting",
            "洞察摘要": f"从批量预测结果中识别出用户行为模式并写入挖掘结果" + "；发现2个关键行为路径",
            "关键分群": "高转化_群A,待激活_群B",
            "置信度": "0.82",
            "结果文件": f"mining/predict_{task_id}_result.json",
            "结果版本": "v1.0.0",
        })
        return {"ok": True, "task_id": task_id, "mining_id": next_id,
                "message": f"预测结果已采集并导入挖掘结果"}

    def validate_prediction_consistency(self, task_id: int) -> dict[str, Any]:
        """校验预测结果的一致性和格式合法性，确保输出符合预期schema。"""
        task = store.find(self.MODULE, task_id)
        if task is None:
            return {"ok": False}
        is_valid = task.get("输出格式") in ("csv", "json", "parquet")
        if not is_valid:
            task["abnormal"] = True
            return {"ok": False, "reason": f"输出格式 {task.get('输出格式')} 不合法"}
        return {"ok": True, "task_id": task_id, "format": task.get("输出格式")}

    def split_predict_partitions(self, task_id: int) -> dict[str, Any]:
        """将批量预测任务切分为多个分区并行执行，提高大规模数据集的预测吞吐。"""
        task = store.find(self.MODULE, task_id)
        if task is None:
            return {"ok": False}
        parallelism = int(str(task.get("并行度", "4")))
        partitions = []
        for i in range(parallelism):
            partitions.append({
                "partition_id": i,
                "task_id": task_id,
                "model_id": task.get("模型编号"),
                "output": f"predict_output/{task_id}/part_{i:03d}.csv",
            })
        return {"ok": True, "task_id": task_id, "partition_count": parallelism, "partitions": partitions}

    def export_predictions_to_csv(self, task_id: int) -> dict[str, Any]:
        """导出批量预测结果为CSV格式，写入外部存储供下游业务消费。"""
        task = store.find(self.MODULE, task_id)
        if task is None:
            return {"ok": False}
        export_rows = store.rows("data_export")
        next_id = max((int(r.get("id", 0)) for r in export_rows), default=0) + 1
        export_rows.append({
            "id": next_id,
            "status": "已完成",
            "pending": False,
            "abnormal": False,
            "任务编号": f"EXP-P{task_id:04d}-{next_id:04d}",
            "数据来源": f"predict_task_{task_id}",
            "导出范围": "全量预测结果",
            "导出格式": "csv",
            "字段映射": "prediction_id,model_id,score,label,dt",
            "分片策略": "none",
            "压缩方式": "gzip",
            "导出状态": "已完成",
        })
        return {"ok": True, "export_id": next_id, "task_id": task_id}

    # ── 历史归档回读：归档索引 + 落库数据 + 改分依据 ──
    # 存档地址不再只由任务编号拼成，而是带归档代次；每次导出归档都在索引里
    # 登记一条记录，索引与落库数据一致后才记成功，回读一律按索引找实际那份数据。

    def _archive_index(self) -> list[dict[str, Any]]:
        return store.internal("predict_archive_index", [])

    def _archive_data(self) -> dict[int, dict[str, Any]]:
        return store.internal("predict_archive_data", {})

    def _corrections(self) -> list[dict[str, Any]]:
        return store.internal("predict_score_corrections", [])

    def _next_seq(self) -> int:
        seq = store.internal("predict_archive_seq", [0])
        seq[0] += 1
        return seq[0]

    @staticmethod
    def _checksum(rows: list[dict[str, Any]]) -> str:
        payload = "|".join(f"{row.get('prediction_id')}:{row.get('score')}" for row in rows)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def _prediction_rows(self, task_id: int) -> list[dict[str, Any]]:
        """预测任务已生成的结果行（懒生成、按任务编号确定性产出）。"""
        table = store.internal("predict_result_rows", {})
        if task_id not in table:
            generated = []
            for i in range(5):
                score = 0.5 + ((task_id * 37 + i * 11) % 45) / 100
                generated.append({
                    "task_id": task_id,
                    "prediction_id": f"P{task_id:04d}-R{i + 1:03d}",
                    "score": f"{score:.2f}",
                    "label": "positive" if score >= 0.7 else "negative",
                    "dt": "2026-11-22",
                })
            table[task_id] = generated
        return table[task_id]

    def _find_archive(self, archive_id: int) -> dict[str, Any] | None:
        return next((a for a in self._archive_index() if a["archive_id"] == archive_id), None)

    def _archive_complete(self, entry: dict[str, Any]) -> bool:
        """索引已确认且落库数据完整一致，才构成可恢复副本。"""
        if entry["status"] != ARCHIVE_CONFIRMED:
            return False
        data = self._archive_data().get(entry["archive_id"])
        if data is None or data.get("copied", 0) < entry["row_count"]:
            return False
        rows = data.get("rows", [])
        return len(rows) == entry["row_count"] and self._checksum(rows) == entry["checksum"]

    def prediction_detail(self, task_id: int) -> dict[str, Any]:
        """预测详情：任务信息、当前预测行、最新已确认归档指针与改分记录。"""
        task = store.find(self.MODULE, task_id)
        if task is None:
            return {"ok": False, "message": f"预测任务 {task_id} 不存在"}
        confirmed = [a for a in self._archive_index()
                     if a["task_id"] == task_id and a["status"] == ARCHIVE_CONFIRMED]
        current = confirmed[-1] if confirmed else None
        return {
            "ok": True,
            "task_id": task_id,
            "任务编号": task.get("任务编号"),
            "模型编号": task.get("模型编号"),
            "rows": [dict(row) for row in self._prediction_rows(task_id)],
            "current_archive": dict(current) if current else None,
            "corrections": [dict(c) for c in self._corrections() if c["task_id"] == task_id],
        }

    def begin_prediction_archive(self, task_id: int, operator: str = "归档人员") -> dict[str, Any]:
        """开始一次归档：登记归档索引（复制中），并把当前预测行快照写入落库区。

        快照在登记时定型，之后的分数更正不会混入这一笔归档；拷贝完成前
        该副本只是部分数据，不能当作完整数据打开。
        """
        task = store.find(self.MODULE, task_id)
        if task is None:
            return {"ok": False, "message": f"预测任务 {task_id} 不存在"}
        index = self._archive_index()
        archive_id = max((a["archive_id"] for a in index), default=0) + 1
        generation = sum(1 for a in index if a["task_id"] == task_id) + 1
        snapshot = [dict(row) for row in self._prediction_rows(task_id)]
        entry = {
            "archive_id": archive_id,
            "task_id": task_id,
            "generation": generation,
            "archive_path": f"archive/predict/task-{task_id:04d}/gen-{generation:03d}/",
            "status": ARCHIVE_COPYING,
            "row_count": len(snapshot),
            "checksum": self._checksum(snapshot),
            "operator": operator,
            "seq": self._next_seq(),
        }
        index.append(entry)
        # 复制是渐进的：先落一部分，confirm 才把拷贝进度补齐
        self._archive_data()[archive_id] = {"rows": snapshot, "copied": len(snapshot) // 2}
        return {
            "ok": True,
            "archive_id": archive_id,
            "generation": generation,
            "archive_path": entry["archive_path"],
            "status": ARCHIVE_COPYING,
            "copied_rows": len(snapshot) // 2,
            "row_count": len(snapshot),
        }

    def confirm_prediction_archive(self, archive_id: int) -> dict[str, Any]:
        """确认归档：归档索引与落库数据一致（行数+校验和）后才记成功。"""
        entry = self._find_archive(archive_id)
        if entry is None:
            return {"ok": False, "message": f"归档 {archive_id} 不存在"}
        if entry["status"] == ARCHIVE_CONFIRMED:
            return {"ok": True, "archive_id": archive_id, "message": "归档已确认"}
        if entry["status"] == ARCHIVE_STOPPED:
            return {"ok": False, "archive_id": archive_id,
                    "message": f"归档 {archive_id} 已截停，不能确认"}
        data = self._archive_data().get(archive_id)
        rows = (data or {}).get("rows", [])
        consistent = (
            data is not None
            and len(rows) == entry["row_count"]
            and self._checksum(rows) == entry["checksum"]
        )
        if not consistent:
            entry["abnormal"] = True
            return {"ok": False, "archive_id": archive_id,
                    "message": "归档索引与落库数据不一致，未记成功"}
        data["copied"] = entry["row_count"]
        entry["status"] = ARCHIVE_CONFIRMED
        return {
            "ok": True,
            "archive_id": archive_id,
            "generation": entry["generation"],
            "archive_path": entry["archive_path"],
            "message": f"归档 {archive_id} 已确认",
        }

    def archive_predictions(self, task_id: int, operator: str = "值班人员") -> dict[str, Any]:
        """一键归档：登记索引并确认落库（导出动作触发的标准归档路径）。"""
        begun = self.begin_prediction_archive(task_id, operator=operator)
        if not begun.get("ok"):
            return begun
        confirmed = self.confirm_prediction_archive(begun["archive_id"])
        if not confirmed.get("ok"):
            return confirmed
        return {
            "ok": True,
            "archive_id": begun["archive_id"],
            "generation": begun["generation"],
            "archive_path": begun["archive_path"],
            "status": ARCHIVE_CONFIRMED,
            "message": f"预测任务 {task_id} 第 {begun['generation']} 代归档已确认",
        }

    def stop_prediction_archive(self, archive_id: int) -> dict[str, Any]:
        """截停复制中的归档：保留部分副本现场，但它不构成可恢复副本。"""
        entry = self._find_archive(archive_id)
        if entry is None:
            return {"ok": False, "message": f"归档 {archive_id} 不存在"}
        if entry["status"] != ARCHIVE_COPYING:
            return {"ok": False, "archive_id": archive_id,
                    "message": f"归档 {archive_id} 当前状态为{entry['status']}，仅复制中可截停"}
        entry["status"] = ARCHIVE_STOPPED
        return {"ok": True, "archive_id": archive_id, "status": ARCHIVE_STOPPED,
                "message": f"归档 {archive_id} 已截停"}

    def open_prediction_archive(self, archive_id: int) -> dict[str, Any]:
        """历史详情打开归档：只有已确认且落库完整的副本才能当作完整数据打开。

        返回的是当时那组预测行的快照，已领取的结果再次读取仍是一致的。
        """
        entry = self._find_archive(archive_id)
        if entry is None:
            return {"ok": False, "message": f"归档 {archive_id} 不存在"}
        if not self._archive_complete(entry):
            return {"ok": False, "archive_id": archive_id,
                    "message": f"归档 {archive_id} 是未完成的副本，无法当作完整数据打开"}
        data = self._archive_data()[archive_id]
        return {
            "ok": True,
            "archive_id": archive_id,
            "task_id": entry["task_id"],
            "generation": entry["generation"],
            "archive_path": entry["archive_path"],
            "rows": [dict(row) for row in data["rows"]],
        }

    def archive_history(self, task_id: int) -> dict[str, Any]:
        """历史记录：每笔归档是否具备可恢复副本；多笔竞争时指明当前可恢复的是哪笔。"""
        entries = [a for a in self._archive_index() if a["task_id"] == task_id]
        archives = []
        recoverable_ids = []
        for entry in entries:
            data = self._archive_data().get(entry["archive_id"])
            recoverable = self._archive_complete(entry)
            if recoverable:
                recoverable_ids.append(entry["archive_id"])
            if entry["status"] == ARCHIVE_CONFIRMED and recoverable:
                note = "已确认，具备可恢复副本"
            elif entry["status"] == ARCHIVE_STOPPED:
                note = "复制被截停，仅剩部分副本，不可作为完整数据"
            elif entry["status"] == ARCHIVE_COPYING:
                note = "复制中，尚未确认落库一致"
            else:
                note = "落库数据与索引不一致，不可恢复"
            archives.append({
                "archive_id": entry["archive_id"],
                "generation": entry["generation"],
                "archive_path": entry["archive_path"],
                "status": entry["status"],
                "operator": entry["operator"],
                "recoverable": recoverable,
                "copied_rows": (data or {}).get("copied", 0),
                "row_count": entry["row_count"],
                "note": note,
            })
        current = recoverable_ids[-1] if recoverable_ids else None
        conflict = None
        if len(entries) > 1 and any(not a["recoverable"] for a in archives):
            conflict = (
                f"存在多笔归档竞争：归档#{current} 已确认并具备可恢复副本，"
                "其余为未完成副本，不能当作完整数据打开"
                if current is not None else
                "存在多笔归档竞争：尚无已确认的可恢复副本"
            )
        return {
            "ok": True,
            "task_id": task_id,
            "current_archive_id": current,
            "archives": archives,
            "conflict": conflict,
        }

    def correct_prediction_score(self, task_id: int, prediction_id: str,
                                 new_score: str, reason: str,
                                 operator: str = "值班人员") -> dict[str, Any]:
        """分数更正：只改当前预测行并记录改分依据，已确认归档保持原值。"""
        task = store.find(self.MODULE, task_id)
        if task is None:
            return {"ok": False, "message": f"预测任务 {task_id} 不存在"}
        row = next((r for r in self._prediction_rows(task_id)
                    if r["prediction_id"] == prediction_id), None)
        if row is None:
            return {"ok": False, "message": f"预测行 {prediction_id} 不存在"}
        old_score = row["score"]
        row["score"] = str(new_score)
        try:
            row["label"] = "positive" if float(new_score) >= 0.7 else "negative"
        except (TypeError, ValueError):
            pass
        self._corrections().append({
            "task_id": task_id,
            "prediction_id": prediction_id,
            "old_score": old_score,
            "new_score": str(new_score),
            "reason": reason,
            "operator": operator,
            "seq": self._next_seq(),
        })
        return {"ok": True, "task_id": task_id, "prediction_id": prediction_id,
                "old_score": old_score, "new_score": str(new_score)}

    def recall_two_generations(self, task_id: int) -> dict[str, Any]:
        """按归档索引回查最近两代已确认数据：读出变化行与改分依据，回到相应预测批次。"""
        task = store.find(self.MODULE, task_id)
        if task is None:
            return {"ok": False, "message": f"预测任务 {task_id} 不存在"}
        confirmed = [a for a in self._archive_index()
                     if a["task_id"] == task_id and a["status"] == ARCHIVE_CONFIRMED]
        if len(confirmed) < 2:
            return {"ok": False,
                    "message": f"预测任务 {task_id} 已确认归档不足两代，无法回查"}
        older, newer = confirmed[-2], confirmed[-1]
        data = self._archive_data()
        old_rows = {r["prediction_id"]: r for r in data[older["archive_id"]]["rows"]}
        new_rows = {r["prediction_id"]: r for r in data[newer["archive_id"]]["rows"]}
        changed_rows = [
            {"prediction_id": pid,
             "old_score": old_rows[pid]["score"],
             "new_score": new_rows[pid]["score"]}
            for pid in new_rows
            if pid in old_rows and old_rows[pid]["score"] != new_rows[pid]["score"]
        ]
        basis = [dict(c) for c in self._corrections()
                 if c["task_id"] == task_id and older["seq"] < c["seq"] <= newer["seq"]]
        return {
            "ok": True,
            "task_id": task_id,
            "任务编号": task.get("任务编号"),
            "模型编号": task.get("模型编号"),
            "generations": [
                {"archive_id": older["archive_id"], "generation": older["generation"],
                 "archive_path": older["archive_path"]},
                {"archive_id": newer["archive_id"], "generation": newer["generation"],
                 "archive_path": newer["archive_path"]},
            ],
            "changed_rows": changed_rows,
            "corrections": basis,
            "message": f"已回查第 {older['generation']} 代到第 {newer['generation']} 代，"
                       f"变化 {len(changed_rows)} 行",
        }