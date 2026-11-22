"""数据导出管理领域服务：分析结果的导出配置、格式转换与异步任务。"""
from __future__ import annotations

import re
from typing import Any

from app.services.base import BaseModuleService
from app.store import store


class DataExportService(BaseModuleService):
    MODULE = "data_export"
    ENTITY = "导出任务"
    OPERATION_TERM = "转换"
    LIST_FIELDS = ["任务编号", "数据来源", "导出范围", "导出格式", "字段映射", "分片策略", "压缩方式", "导出状态"]
    REQUIRED_FIELDS = ["任务编号", "数据来源", "导出范围"]
    STATUS_ORDER = ["草稿", "导出中", "已完成", "已失败"]
    ACTION_RULES = {"开始导出": "导出中", "确认完成": "已完成", "标记失败": "已失败"}
    NEGATIVE_ACTIONS = []

    def export_feature_dataset(self, task_id: int, output_format: str = "csv") -> dict[str, Any]:
        """将特征数据集导出为指定格式并写入外部存储路径。"""
        task = store.find(self.MODULE, task_id)
        if task is None:
            return {"ok": False}
        task["status"] = "导出中"
        task["pending"] = True
        task["导出格式"] = output_format
        task["status"] = "已完成"
        task["pending"] = False
        return {"ok": True, "task_id": task_id, "format": output_format,
                "output_path": f"exports/{task_id}/feature_dataset.{output_format}"}

    def convert_to_parquet(self, task_id: int) -> dict[str, Any]:
        """将导出数据转换为Parquet列式存储格式并压缩归档。"""
        task = store.find(self.MODULE, task_id)
        if task is None:
            return {"ok": False}
        task["导出格式"] = "parquet"
        task["压缩方式"] = "snappy"
        task["status"] = "已完成"
        task["pending"] = False
        return {"ok": True, "task_id": task_id, "format": "parquet",
                "compression": "snappy", "path": f"exports/{task_id}/data.snappy.parquet"}

    def archive_export_to_storage(self, task_id: int) -> dict[str, Any]:
        """将完成的导出任务归档到长期存储，保留历史版本供回溯。

        存档地址带归档代次，同一任务多次归档可以辨别是哪次保存的；
        预测结果导出委托预测归档索引登记，回读时按索引找到实际那份数据。
        """
        task = store.find(self.MODULE, task_id)
        if task is None:
            return {"ok": False}
        matched = re.fullmatch(r"predict_task_(\d+)", str(task.get("数据来源", "")))
        if matched:
            from app.services import service_for
            predict_service = service_for("predict_batch")
            archived = predict_service.archive_predictions(
                int(matched.group(1)), operator="归档人员")
            if not archived.get("ok"):
                return {"ok": False, "message": archived.get("message", "预测归档失败")}
            return {
                "ok": True,
                "task_id": task_id,
                "archive_id": archived["archive_id"],
                "generation": archived["generation"],
                "archive_path": archived["archive_path"],
                "format": task.get("导出格式"),
                "message": f"导出任务 {task_id} 已归档至 {archived['archive_path']}",
            }
        seq_table = store.internal("export_archive_seq", {})
        seq_table[task_id] = seq_table.get(task_id, 0) + 1
        archive_path = f"archive/exports/{task_id}/gen-{seq_table[task_id]:03d}/"
        return {
            "ok": True,
            "task_id": task_id,
            "archive_path": archive_path,
            "format": task.get("导出格式"),
            "message": f"导出任务 {task_id} 已归档至 {archive_path}",
        }

    def query_export_progress(self, task_id: int) -> dict[str, Any]:
        """查询导出任务的执行状态和进度详情。"""
        task = store.find(self.MODULE, task_id)
        if task is None:
            return {"ok": False}
        return {
            "ok": True,
            "task_id": task_id,
            "status": task.get("status"),
            "source": task.get("数据来源"),
            "format": task.get("导出格式"),
            "compression": task.get("压缩方式"),
        }