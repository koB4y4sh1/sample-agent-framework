# src/utils/debug_decorators.py
import functools
import os
import time
import traceback
from typing import Any, Callable

import debugpy

from .logging import logger


def log_execution_time(func: Callable) -> Callable:
    """関数の実行時間を測定しログに記録するデコレーター
    
    任意の非同期関数に@log_execution_timeを付けるだけで、自動的に実行時間の測定とログ出力が行われる
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        func_name = func.__name__

        try:
            # 関数の実行
            result = await func(*args, **kwargs)

            # 実行時間の計算とログ
            execution_time = time.time() - start_time
            logger.info(
                f"{func_name} completed successfully in {execution_time:.3f}s"
            )
            logger.info("teset")

            # 実行時間が長い場合は警告
            if execution_time > 5.0:
                logger.warning(
                    f"{func_name} took {execution_time:.3f}s - "
                    "consider optimization"
                )

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                f"{func_name} failed after {execution_time:.3f}s: {e}\n"
                f"Traceback: {traceback.format_exc()}"
            )
            raise

    return wrapper




def setup_remote_debugging():
    """リモートデバッグを有効化
    
    Dockerコンテナ内で動作するMCPサーバーや、リモートサーバー上のMCPサーバーに対して、ローカルのVSCodeからブレークポイントを設定してデバッグできるようになる
    環境変数ENABLE_REMOTE_DEBUG=trueを設定するだけで有効化され、デフォルトではポート5678でデバッガーの接続を待ち受けます。
    """
    if os.getenv("ENABLE_REMOTE_DEBUG", "").lower() == "true":
        debug_port = int(os.getenv("DEBUG_PORT", "5678"))
        debugpy.listen(("0.0.0.0", debug_port))

        print(f"Remote debugging enabled on port {debug_port}")
        print("Waiting for debugger to attach...")

        # デバッガーの接続を待つ（オプション）
        if os.getenv("WAIT_FOR_DEBUGGER", "").lower() == "true":
            debugpy.wait_for_client()
            print("Debugger attached!")
