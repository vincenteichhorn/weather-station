from datetime import datetime, timedelta
import os
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from telegram import Update, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import ContextTypes

import api
from plotting import create_forecast_plot, create_weather_plot
from utils import restricted

buttons = [[KeyboardButton("/now"), KeyboardButton("/week")], [KeyboardButton("/24h")]]
buttons_authorized = [*buttons, [KeyboardButton("/forecast")]]
keyboard = ReplyKeyboardMarkup(buttons, resize_keyboard=True)
keyboard_authorized = ReplyKeyboardMarkup(buttons_authorized, resize_keyboard=True)


async def send_plot_image(context: ContextTypes, chat_id: int, figure, caption: str, prefix: str):
    """Save a Matplotlib figure temporarily, send it, and clean it up."""
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".png", prefix=prefix, delete=False
        ) as temporary_file:
            temporary_path = temporary_file.name
        figure.savefig(
            temporary_path,
            format="png",
            dpi=180,
            facecolor=figure.get_facecolor(),
            bbox_inches="tight",
        )
        await context.bot.send_photo(chat_id=chat_id, photo=temporary_path, caption=caption)
    finally:
        plt.close(figure)
        if temporary_path:
            os.unlink(temporary_path)


def get_keyboard(update: Update):
    user_id = str(update.effective_user.id)
    authorized_users = os.environ.get("AUTHORIZED_USERS", "").split(",")
    return keyboard_authorized if user_id in authorized_users else keyboard


async def greet_user(update: Update, context: ContextTypes):
    text = "Hallo! Ich bin der Wetterbot. Ich kann dir aktuelle Wetterdaten von der Wetterstation in Bühlau senden. \n"
    text += (
        "Frag mich einfach nach den aktuellen Wetterdaten oder dem Wetter der letzten 7 Tage. \n"
    )
    text += "Dafür kannst du diese Befehle an mich senden: \n"
    text += "/now - aktuelle Wetterdaten \n"
    text += "/week - Wetter der letzten 7 Tage \n"
    text += "/24h - Wetterdaten der letzten 24 Stunden \n"
    if str(update.effective_user.id) in os.environ.get("AUTHORIZED_USERS", "").split(","):
        text += "/forecast - Wettervorhersage für die nächsten 24 Stunden"
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=text,
        reply_markup=get_keyboard(update),
    )


async def respond_to_unknown(update: Update, context: ContextTypes):
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Das ist ein unbekannter Befehl für mich. \n "
        "/now - aktuelle Wetterdaten \n"
        "/week - Wetter der letzten 7 Tage",
        reply_markup=get_keyboard(update),
    )


async def provide_help(update: Update, context: ContextTypes):
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Ich bin der Wetterbot. Ich kann dir aktuelle Wetterdaten von der Wetterstation in Bühlau senden. Dafür versteht er folgende Befehle: \n"
        "/now - aktuelle Wetterdaten \n"
        "/week - Wetter der letzten 7 Tage",
        reply_markup=get_keyboard(update),
    )


async def retrieve_weather_now(update: Update, context: ContextTypes):
    latest_weather = api.get_latest_weather()
    if not latest_weather:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Keine Wetterdaten verfügbar.",
            parse_mode="Markdown",
        )
        return
    date = datetime.strptime(latest_weather[0]["date"], "%Y-%m-%dT%H:%M:%S.%f")
    datestr = f"*{date.strftime('%d.%m.%Y')}* um *{date.strftime('%H:%M')}* Uhr"
    text = f"Letzte Wetterdaten vom {datestr}: \n\n"
    for measurement in latest_weather:
        text += f"*{measurement['name']}*: {measurement['value']:.2f} {measurement['unit']} \n"

    await context.bot.send_message(
        chat_id=update.effective_chat.id, text=text, parse_mode="Markdown"
    )


async def retrieve_last_week_weather(update: Update, context: ContextTypes):
    last_week_weather = api.get_last_week_weather()
    text = "Wetterdaten der letzten 7 Tage: \n\n"
    for series, measurements in last_week_weather.items():
        text += f"*{series}*: \n"
        for measurement in measurements:
            text += f"- {measurement['name']}: {measurement['value']:.2f} {measurement['unit']} \n"
        text += "\n"

    await context.bot.send_message(
        chat_id=update.effective_chat.id, text=text, parse_mode="Markdown"
    )


async def retrieve_24h_weather_plot(update: Update, context: ContextTypes):
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Die Wetterdaten der letzten 24 Stunden werden geladen...",
    )
    end_date = datetime.now()
    start_date = end_date.replace(microsecond=0) - timedelta(hours=24)
    series = api.get_weather_series()
    plots = []
    plotted_series = set()
    for series_entry in series:
        if series_entry.get("disabled"):
            continue
        series_key = (series_entry["name"], series_entry["unit"])
        if series_key in plotted_series:
            continue
        for series_short in series_entry.get("shorts", []):
            measurements = api.get_series_measurements(series_short, start_date, end_date)
            if measurements:
                plots.append((series_entry["name"], series_entry["unit"], measurements))
                plotted_series.add(series_key)
                break

    if not plots:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Keine Wetterdaten für die letzten 24 Stunden verfügbar.",
        )
        return

    try:
        sun_times = api.get_sunrise_sunset()
    except Exception:
        sun_times = None

    figure = create_weather_plot(
        plots,
        start_date,
        end_date,
        "Wetterdaten der letzten 24 Stunden",
        sun_times=sun_times,
    )

    await send_plot_image(
        context,
        update.effective_chat.id,
        figure,
        "Wetterdaten der letzten 24 Stunden",
        "weather-24h-",
    )


@restricted
async def retrieve_forecast_plot(update: Update, context: ContextTypes):
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Die Wettervorhersage wird berechnet...",
    )
    forecast = api.get_forecast()
    if not forecast:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Keine Wettervorhersage verfügbar.",
        )
        return

    forecast_entries = next(iter(forecast.values()))
    forecast_by_series = {entries[0]["name"]: entries for entries in forecast.values() if entries}
    forecast_start = datetime.fromisoformat(forecast_entries[0]["date"])
    start_date = forecast_start - timedelta(hours=24)
    series = api.get_weather_series()
    plots = []
    for series_entry in series:
        if series_entry.get("disabled"):
            continue
        for series_short in series_entry.get("shorts", []):
            measurements = api.get_series_measurements(series_short, start_date, forecast_start)
            if measurements and series_entry["name"] in forecast_by_series:
                plots.append(
                    (
                        series_entry["name"],
                        series_entry["unit"],
                        measurements,
                        forecast_by_series[series_entry["name"]],
                    )
                )
                break

    if not plots:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Keine Messwerte für die Vorhersage verfügbar.",
        )
        return

    try:
        sun_times = api.get_sunrise_sunset()
    except Exception:
        sun_times = None
    figure = create_forecast_plot(
        plots,
        start_date,
        forecast_start,
        max(
            datetime.fromisoformat(entry["date"])
            for entries in forecast.values()
            for entry in entries
        ),
        "Wettervorhersage",
        sun_times=sun_times,
    )
    await send_plot_image(
        context,
        update.effective_chat.id,
        figure,
        "Wettervorhersage",
        "weather-forecast-",
    )
