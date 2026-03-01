import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_logging(level="INFO", log_to_file=True):
    # ディレクトリの作成
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # フォーマッターの設定
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # ルートロガー設定
    logger = logging.getLogger("mcp")
    logger.setLevel(level)

    # コンソールハンドラー
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ファイルハンドラー
    if log_to_file:
        log_file = log_dir/f"mcp-{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
    
logger = setup_logging(level="DEBUG")