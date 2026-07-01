import os
import logging
import asyncio
from collections import defaultdict

from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
import google.generativeai as genai

# ---------- Настройка ----------

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN не задан в .env")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY не задан в .env")

genai.configure(api_key=GEMINI_API_KEY)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Ты дружелюбный ИИ-ассистент в Telegram-чате. "
    "Отвечай кратко, понятно и по делу. Используй русский язык, "
    "если пользователь не пишет на другом."
)

# Храним историю диалога по каждому чату (в памяти, сбрасывается при рестарте бота)
chat_histories: dict[int, list] = defaultdict(list)
MAX_HISTORY_MESSAGES = 20  # сколько последних сообщений держим в контексте

# ---------- Работа с Gemini ----------

def get_model():
    return genai.GenerativeModel(
        model_name=MODEL_NAME,
        system_instruction=SYSTEM_PROMPT,
    )


async def ask_gemini(chat_id: int, user_text: str) -> str:
    history = chat_histories[chat_id]

    model = get_model()
    chat = model.start_chat(history=history)

    def _send():
        return chat.send_message(user_text)

    # google-generativeai синхронный, запускаем в отдельном потоке,
    # чтобы не блокировать событийный цикл бота
    response = await asyncio.to_thread(_send)

    # Обновляем сохранённую историю и обрезаем её, чтобы не росла бесконечно
    chat_histories[chat_id] = chat.history[-MAX_HISTORY_MESSAGES:]

    return response.text


# ---------- Хендлеры Telegram ----------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_histories[update.effective_chat.id] = []
    await update.message.reply_text(
        "Привет! Я ИИ-ассистент на Gemini. Просто напиши вопрос — отвечу.\n\n"
        "Команды:\n"
        "/reset — очистить историю диалога\n"
        "/help — помощь"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Пиши любой вопрос текстом, я отвечу с помощью Gemini.\n"
        "/reset — забыть предыдущий контекст разговора"
    )


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_histories[update.effective_chat.id] = []
    await update.message.reply_text("Контекст диалога очищен ✅")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_text = update.message.text

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    try:
        answer = await ask_gemini(chat_id, user_text)
    except Exception as e:
        logger.exception("Ошибка при обращении к Gemini")
        await update.message.reply_text(
            f"Упс, что-то пошло не так при обращении к Gemini: {e}"
        )
        return

    # Telegram режет сообщения длиннее 4096 символов — бьём на части
    for i in range(0, len(answer), 4000):
        await update.message.reply_text(answer[i : i + 4000])


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.error("Исключение при обработке update: %s", context.error)


# ---------- Точка входа ----------

def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_error_handler(error_handler)

    logger.info("Бот запущен, ждём сообщений...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
