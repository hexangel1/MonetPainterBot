""" Telegram bot endpoints """
import os
import shutil
import torch

from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.types import InputFile
from model import CycleGAN


device = 'cuda' if torch.cuda.is_available() else 'cpu'
gan = CycleGAN(3, 3, 60, device)
gan.load_from_checkpoint(ckpt_path=os.environ['CHECKPOINT_PATH'])

bot = Bot(token=os.environ['TG_BOT_TOKEN'])
dp = Dispatcher(bot)


@dp.message_handler(commands=['start'])
async def handle_start(msg: types.Message):
    """Processes the start command, creates users' directory and says hi

    :param msg: message from user
    :type msg: aiogram.types.Message
    """
    try:
        os.mkdir(f"storage/{msg.from_user.id}", mode=0o755)
    except FileExistsError:
        pass
    await msg.answer(("Hi, dear {fname}, I am transfer style bot 😎\n"
                     "Use /help command to find out what I can do\n").format(fname=msg.from_user.first_name))


@dp.message_handler(commands=['stop'])
async def handle_stop(msg: types.Message):
    """Processes the stop command, deletes users' directory and says bye

    :param msg: message from user
    :type msg: aiogram.types.Message
    """
    shutil.rmtree(f"storage/{msg.from_user.id}", ignore_errors=True)
    await msg.answer("Goodbye, dear {fname}".format(fname=msg.from_user.first_name))


@dp.message_handler(commands=['help'])
async def handle_help(msg: types.Message):
    """Processes the help command, sends user commands' descriptions

    :param msg: message from user
    :type msg: aiogram.types.Message
    """
    await msg.answer("not implemented")


@dp.message_handler(content_types=['text'])
async def handle_text_message(msg: types.Message):
    """Text handler"""
    
    mesg = msg.text[::-1]
    await msg.answer(mesg)


@dp.message_handler(content_types=['photo'])
async def handle_photo_message(msg: types.Message):
    """Handles photo messages and saves it to the users' directory

    :param msg: message from user
    :type msg: aiogram.types.Message
    """
    file_ID = msg.photo[-1].file_id
    file_info = await bot.get_file(file_ID)
    downloaded_file = await bot.download_file(file_info.file_path)
    image_path = f"storage/{msg.from_user.id}/{file_ID}"
    styled_image_path = f"storage/{msg.from_user.id}/result.png"
    with open(image_path, "wb") as new_file:
        new_file.write(downloaded_file.getvalue())

    gan.transfer_style(image_path, styled_image_path)
    await bot.send_photo(msg.chat.id, photo=InputFile(styled_image_path), caption="Monet styled photo")


@dp.message_handler(content_types=['document'])
async def handle_document_message(msg: types.Message):
    """Handles document messages and saves it to the users' directory

    :param msg: message from user
    :type msg: aiogram.types.Message
    """
    file = msg.document
    file_ID = file.file_id
    file_info = await bot.get_file(file_ID)
    downloaded_file = await bot.download_file(file_info.file_path)
    with open(f"storage/{msg.from_user.id}/{file_ID}={file.file_name}", "wb") as new_file:
        new_file.write(downloaded_file.getvalue())
    await msg.answer("Document uploaded!")