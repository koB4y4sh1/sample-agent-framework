import json
import traceback
from datetime import datetime
from typing import Any, Dict, List

from .logging import logger


class ErrorTracker:
    """エラーの追跡と分析のためのクラス
    
    単にエラーをログに出力するだけでなく、エラーの種類ごとの発生頻度を追跡し、同じエラーが頻発している場合にアラートを出す
    """

    def __init__(self):
        self.errors: List[Dict[str, Any]] = []
        self.error_counts: Dict[str, int] = {}

    def log_error(self, error: Exception, context: Dict[str, Any]) -> None:
        """エラーを記録し、統計情報を更新"""
        error_type = type(error).__name__

        # エラー情報の構造化
        error_info = {
            "timestamp": datetime.now().isoformat(),
            "type": error_type,
            "message": str(error),
            "context": context,
            "stack_trace": traceback.format_exc()
        }

        # エラーリストに追加
        self.errors.append(error_info)

        # エラーカウントの更新
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1

        # ログに記録
        logger.error(f"Error tracked: {json.dumps(error_info, indent=2)}")

        # 特定のエラーが頻発している場合は警告
        if self.error_counts[error_type] > 10:
            logger.critical(
                f"Error '{error_type}' has occurred {self.error_counts[error_type]} times!"
            )

    def get_error_summary(self) -> Dict[str, Any]:
        """エラーの統計情報を取得"""
        return {
            "total_errors": len(self.errors),
            "error_counts": self.error_counts,
            "recent_errors": self.errors[-10:]  # 直近10件
        }

# グローバルエラートラッカー
error_tracker = ErrorTracker()