from datetime import datetime

from telegram import Update, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import ContextTypes

import api

buttons = [[KeyboardButton("/now"), KeyboardButton("/week")]]
keyboard = ReplyKeyboardMarkup(buttons, resize_keyboard=True)


async def greet_user(update: Update, context: ContextTypes):
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Hallo! Ich bin der Wetterbot. Ich kann dir aktuelle Wetterdaten von der Wetterstation in Bühlau senden. \n"
        "Frag mich einfach nach den aktuellen Wetterdaten oder dem Wetter der letzten 7 Tage. \n"
        "Dafür kannst du diese Befehle an mich senden: \n"
        "/now - aktuelle Wetterdaten \n"
        "/week - Wetter der letzten 7 Tage",
        reply_markup=keyboard,
    )


async def respond_to_unknown(update: Update, context: ContextTypes):
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Das ist ein unbekannter Befehl für mich. \n " "/now - aktuelle Wetterdaten \n" "/week - Wetter der letzten 7 Tage",
        reply_markup=keyboard,
    )


async def provide_help(update: Update, context: ContextTypes):
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Ich bin der Wetterbot. Ich kann dir aktuelle Wetterdaten von der Wetterstation in Bühlau senden. Dafür versteht er folgende Befehle: \n"
        "/now - aktuelle Wetterdaten \n"
        "/week - Wetter der letzten 7 Tage",
        reply_markup=keyboard,
    )


async def retrieve_weather_now(update: Update, context: ContextTypes):
    latest_weather = api.get_latest_weather()
    if not latest_weather:
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Keine Wetterdaten verfügbar.", parse_mode="Markdown")
        return
    date = datetime.strptime(latest_weather[0]["date"], "%Y-%m-%dT%H:%M:%S.%f")
    datestr = f"*{date.strftime('%d.%m.%Y')}* um *{date.strftime('%H:%M')}* Uhr"
    text = f"Letzte Wetterdaten vom {datestr}: \n\n"
    for measurement in latest_weather:
        text += f"*{measurement['name']}*: {measurement['value']:.2f} {measurement['unit']} \n"

    await context.bot.send_message(chat_id=update.effective_chat.id, text=text, parse_mode="Markdown")


async def retrieve_last_week_weather(update: Update, context: ContextTypes):
    last_week_weather = api.get_last_week_weather()
    text = "Wetterdaten der letzten 7 Tage: \n\n"
    for series, measurements in last_week_weather.items():
        text += f"*{series}*: \n"
        for measurement in measurements:
            text += f"- {measurement['name']}: {measurement['value']:.2f} {measurement['unit']} \n"
        text += "\n"

    await context.bot.send_message(chat_id=update.effective_chat.id, text=text, parse_mode="Markdown")
