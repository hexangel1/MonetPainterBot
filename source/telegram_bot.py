"""Telegram bot endpoints"""
import os
import sys
import shutil
import torch
from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.dispatcher import FSMContext
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.types import InputFile, ReplyKeyboardMarkup, ReplyKeyboardRemove, KeyboardButton
from model import CycleGAN


device = 'cuda' if torch.cuda.is_available() else 'cpu'
gan = CycleGAN(3, 3, 60, device)
gan.load_from_checkpoint(ckpt_path=os.environ['CHECKPOINT_PATH'])

keyboard = ReplyKeyboardMarkup(resize_keyboard=True)
keyboard.add(KeyboardButton('/transfer_style'))
keyboard.add(KeyboardButton('/help'))

bot = Bot(token=os.environ['TG_BOT_TOKEN'])
dp = Dispatcher(bot, storage=MemoryStorage())


class CustomStates(StatesGroup):
    """Client states"""

    bot_started = State()
    upload_image = State()


def clear_userdir(user_id: int):
    """Delete files in user directory"""
    for file in os.listdir(f"storage/{user_id}/"):
        if os.path.isfile(f"storage/{user_id}/{file}"):
            os.remove(f"storage/{user_id}/{file}")


@dp.message_handler(commands=['start'], state='*')
async def handle_start(msg: types.Message, state: FSMContext):
    """Processes the start command, creates users' directory and says hi

    :param msg: message from user
    :type msg: aiogram.types.Message
    """
    os.makedirs(f"storage/{msg.from_user.id}/", exist_ok=True, mode=0o755)
    await CustomStates.bot_started.set()
    await msg.answer(f"Hi, dear {msg.from_user.first_name}! 😎\n"
                     "I am Monet Painter Bot 🤖\n"
                     "Use /help command to find out what I can do\n", reply_markup=keyboard)


@dp.message_handler(commands=['stop'], state=CustomStates)
async def handle_stop(msg: types.Message, state: FSMContext):
    """Processes the stop command, deletes users' directory and says bye

    :param msg: message from user
    :type msg: aiogram.types.Message
    """
    shutil.rmtree(f"storage/{msg.from_user.id}/", ignore_errors=True)
    await state.finish()
    await msg.answer(f"Goodbye, dear {msg.from_user.first_name}", reply_markup=ReplyKeyboardRemove())


@dp.message_handler(commands=['help'], state='*')
async def handle_help(msg: types.Message, state: FSMContext):
    """Processes the help command, sends user commands' descriptions

    :param msg: message from user
    :type msg: aiogram.types.Message
    """
    await msg.answer(
        "Use commands:\n"
        "/start to launch the bot\n"
        "/stop to stop the bot\n"
        "/help to see this message\n"
        "/transfer_style to pass image to neural network"
    )


@dp.message_handler(commands=['transfer_style'], state=CustomStates)
async def handle_transfer_style(msg: types.Message, state: FSMContext):
    """Processes the transfer_style command

    :param msg: message from user
    :type msg: aiogram.types.Message
    """
    await CustomStates.upload_image.set()
    await msg.answer("Send me the image 🖼")


@dp.message_handler(content_types=['text'], state=CustomStates)
async def handle_text_message(msg: types.Message, state: FSMContext):
    """Text handler"""
    await msg.answer("Sorry, I'm not good at texting yet 😢")


@dp.message_handler(content_types=['photo'], state=CustomStates.upload_image)
async def handle_photo_message(msg: types.Message, state: FSMContext):
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
    try:
        gan.transfer_style(image_path, styled_image_path)
        await bot.send_photo(msg.chat.id, photo=InputFile(styled_image_path), caption="Monet styled image")
        clear_userdir(msg.from_user.id)
    except Exception as error:
        print("Exception caught:", error, file=sys.stderr, flush=True)
        await msg.answer("Internal server error")
    await CustomStates.bot_started.set()


@dp.message_handler(content_types=['document'], state=CustomStates.upload_image)
async def handle_document_message(msg: types.Message, state: FSMContext):
    """Handles document messages and saves it to the users' directory

    :param msg: message from user
    :type msg: aiogram.types.Message
    """
    file = msg.document
    file_ID = file.file_id
    file_info = await bot.get_file(file_ID)
    downloaded_file = await bot.download_file(file_info.file_path)
    image_path = f"storage/{msg.from_user.id}/{file_ID}"
    styled_image_path = f"storage/{msg.from_user.id}/result.png"

    with open(image_path, "wb") as new_file:
        new_file.write(downloaded_file.getvalue())
    try:
        gan.transfer_style(image_path, styled_image_path)
        await bot.send_document(msg.chat.id, document=InputFile(styled_image_path), caption="Monet styled image")
    except Exception as error:
        print("Exception caught:", error, file=sys.stderr, flush=True)
        await msg.answer("Internal server error")
    await CustomStates.bot_started.set()
