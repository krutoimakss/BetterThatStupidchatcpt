import asyncio
import logging
import os
from collections import defaultdict

import httpx
from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command, CommandStart
from aiogram.types import Message
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = os.getenv("MODEL", "anthropic/claude-sonnet-4.5")
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "20"))  # сообщений на пользователя

if not TELEGRAM_TOKEN or not OPENROUTER_API_KEY:
    raise RuntimeError("Заполните TELEGRAM_TOKEN и OPENROUTER_API_KEY в .env")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

bot = Bot(token=TELEGRAM_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN))
dp = Dispatcher()

SYSTEM_PROMPT = (
    "Ты — опытный senior-разработчик и ассистент по программированию. "
    "Отвечай точно, по делу, с рабочим кодом. "
    "Оформляй код в markdown-блоках с указанием языка. "
    "Если задача неоднозначна — делай разумное предположение и говори какое, "
    "не задавай лишних уточняющих вопросов без необходимости. "
    "Объясняй кратко, если пользователь явно не просит подробностей."
)

# История диалога по пользователям (in-memory, сбрасывается при перезапуске)
history: dict[int, list[dict]] = defaultdict(list)


async def ask_ai(user_id: int, user_text: str) -> str:
    history[user_id].append({"role": "user", "content": user_text})
    # ограничиваем историю, чтобы не разрастался контекст
    history[user_id] = history[user_id][-MAX_HISTORY:]

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history[user_id]

    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL,
                "messages": messages,
            },
        )
        response.raise_for_status()
        data = response.json()

    reply = data["choices"][0]["message"]["content"]
    history[user_id].append({"role": "assistant", "content": reply})
    return reply


def split_message(text: str, limit: int = 4000) -> list[str]:
    """Telegram режет сообщения длиннее ~4096 символов — бьём по кускам."""
    if len(text) <= limit:
        return [text]
    parts = []
    while text:
        parts.append(text[:limit])
        text = text[limit:]
    return parts


@dp.message(CommandStart())
async def cmd_start(message: Message):
    history[message.from_user.id].clear()
    await message.answer(
        "Привет! Я ИИ-ассистент по программированию.\n\n"
        "Просто напиши свой вопрос или задачу — помогу с кодом, отладкой, "
        "архитектурой, объясню алгоритм и т.д.\n\n"
        "Команды:\n"
        "/reset — очистить историю диалога\n"
        "/model — показать текущую модель"
    )


@dp.message(Command("reset"))
async def cmd_reset(message: Message):
    history[message.from_user.id].clear()
    await message.answer("История диалога очищена.")


@dp.message(Command("model"))
async def cmd_model(message: Message):
    await message.answer(f"Текущая модель: `{MODEL}`")


@dp.message(F.text)
async def handle_message(message: Message):
    await bot.send_chat_action(message.chat.id, "typing")
    try:
        reply = await ask_ai(message.from_user.id, message.text)
    except httpx.HTTPStatusError as e:
        logger.exception("Ошибка API")
        await message.answer(
            f"Ошибка при обращении к ИИ (HTTP {e.response.status_code}). "
            "Проверьте ключ OpenRouter и баланс на счету."
        )
        return
    except Exception:
        logger.exception("Неожиданная ошибка")
        await message.answer("Произошла ошибка. Попробуйте ещё раз.")
        return

    for chunk in split_message(reply):
        try:
            await message.answer(chunk)
        except Exception:
            # если markdown в ответе битый — отправляем как обычный текст
            await message.answer(chunk, parse_mode=None)


async def main():
    logger.info("Бот запускается...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
