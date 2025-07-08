# app/bot.py

import json
import logging
import uuid
from datetime import datetime
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from openai import OpenAI

from aiogram import Bot, Dispatcher, Router, types
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup, CallbackQuery
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.fsm.context import FSMContext
from aiogram.filters import CommandStart, Command
from aiogram import F
from dotenv import load_dotenv
import os
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ====== Configuration ======
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CONVERSATIONS_FILE = "conversations.json"

# Инициализация клиента с прокси
client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url="https://api.proxyapi.ru/openai/v1"
)

# ====== AIogram v3 setup ======
from aiogram.client.default import DefaultBotProperties

bot = Bot(token=TELEGRAM_TOKEN, default=DefaultBotProperties(parse_mode="HTML"))

storage = MemoryStorage()
dp = Dispatcher(storage=storage)
router = Router()
dp.include_router(router)

# ====== Load BERT model ======
def load_bert_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(
        "models/bert_head_only_60k"
    )
    model.eval()
    return tokenizer, model

TOKENIZER, BERT_MODEL = load_bert_model()
EMOTION_LABELS = ['positive', 'sad', 'anger', 'neutral']

def classify_emotion(text: str) -> str:
    inputs = TOKENIZER(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = BERT_MODEL(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        label_idx = torch.argmax(probs, dim=1).item()
    return EMOTION_LABELS[label_idx]

# ====== Conversation storage ======
def load_conversations():
    try:
        with open(CONVERSATIONS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def save_conversations(data):
    with open(CONVERSATIONS_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ====== OpenAI GPT interaction ======
def ask_gpt(prompt: str) -> str:
    try:
        chat_completion = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[{'role': 'system', 'content': prompt}],
            max_tokens=150,
            temperature=0.9
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Ошибка при обращении к OpenAI: {e}")
        return 'Извини, произошла ошибка при генерации ответа.'

# ====== Handlers ======

@router.message(CommandStart())
async def cmd_start(message: types.Message, state: FSMContext):
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Начать беседу", callback_data="start_chat")]
        ]
    )
    await message.answer(
        'Привет! Я бот Катя — профессиональный виртуальный психолог. '
        'Я умею распознавать твоё настроение и поддерживать разговор. '
        'Хочешь побеседовать — просто напиши сообщение.',
        reply_markup=keyboard
    )

@router.callback_query(F.data == "start_chat")
async def process_start_chat(callback: CallbackQuery, state: FSMContext):
    session_id = str(uuid.uuid4())
    conversations = load_conversations()
    conversations.append({
        'session_id': session_id,
        'user_id': callback.from_user.id,
        'messages': [],
        'rating': None
    })
    save_conversations(conversations)
    await state.update_data(session_id=session_id)
    await callback.answer()
    await callback.message.answer("Беседа начата! Напиши, что у тебя на уме.")

@router.message()
async def handle_message(message: types.Message, state: FSMContext):
    data = await state.get_data()
    session_id = data.get("session_id")
    if not session_id:
        return await message.reply('Нажми /start и кнопку "Начать беседу" прежде чем писать.')

    try:
        emotion = classify_emotion(message.text)
    except Exception as e:
        logger.error(f"Ошибка при классификации эмоции: {e}")
        emotion = 'neutral'

    conversations = load_conversations()
    session = next((s for s in conversations if s['session_id'] == session_id), None)

    timestamp = datetime.utcnow().isoformat()
    session['messages'].append({'from': 'user', 'text': message.text, 'timestamp': timestamp})
    save_conversations(conversations)

    previous_bot = next((m['text'] for m in reversed(session['messages']) if m['from'] == 'bot'), '')

    prompt = (
        "Ты — доброжелательный и внимательный собеседник с навыками психолога. "
        "Ты умеешь слушать, замечать эмоции собеседника и поддерживать его в трудные моменты. "
        f"Пользователь чувствует себя: {emotion}. "
        f"Он написал: «{message.text}». "
        + (f"Ранее ты ответил: «{previous_bot}». " if previous_bot else "") +
        "Избегай шаблонов вроде «я понимаю» или «это нормально». "
        "Пиши просто, по-человечески, как близкий человек. "
        "Ответ должен быть коротким (1–3 предложения), живым, искренним. "
        "Можно немного юмора или тепла, если это уместно. Не используй формальный тон."
    )
    bot_reply = ask_gpt(prompt)

    session['messages'].append({'from': 'bot', 'text': bot_reply, 'timestamp': datetime.utcnow().isoformat()})
    save_conversations(conversations)

    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[[InlineKeyboardButton(text="Завершить беседу", callback_data="end_chat")]]
    )
    await message.answer(bot_reply, reply_markup=keyboard)

@router.callback_query(F.data == "end_chat")
async def process_end_chat(callback: CallbackQuery):
    await callback.answer()
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[[InlineKeyboardButton(text=f"★{i}", callback_data=f"rate_{i}") for i in range(1, 6)]]
    )
    await callback.message.answer("Оцените беседу:", reply_markup=keyboard)

@router.callback_query(F.data.startswith("rate_"))
async def process_rating(callback: CallbackQuery, state: FSMContext):
    rating = int(callback.data.split('_')[1])
    data = await state.get_data()
    session_id = data.get("session_id")
    conversations = load_conversations()
    session = next((s for s in conversations if s['session_id'] == session_id), None)
    if session:
        session['rating'] = rating
        save_conversations(conversations)
    await state.clear()
    await callback.answer()
    await callback.message.answer("Спасибо за разговор! Хорошего дня!")

# ====== Start Bot ======

async def main():
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
