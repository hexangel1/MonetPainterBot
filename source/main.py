""" Main """
import os
import signal
import telegram_bot
from aiogram.utils import executor


def signal_handler(signum, frame):
    """Signal handler"""
    if signum == signal.SIGTERM:
        os._exit(0)


def main():
    try:
        os.mkdir("storage", mode=0o755)
    except FileExistsError:
        pass
    signal.signal(signal.SIGTERM, signal_handler)
    executor.start_polling(telegram_bot.dp)


if __name__ == '__main__':
    main()