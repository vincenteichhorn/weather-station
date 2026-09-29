from datetime import datetime, timedelta
import os
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from telegram import Update, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import ContextTypes

import api
from utils import restricted

buttons = [[KeyboardButton("/now"), KeyboardButton("/week")]]
buttons_authorized = [
    [KeyboardButton("/now"), KeyboardButton("/week")],
    [KeyboardButton("/24h"), KeyboardButton("/forecast")],
]
keyboard = ReplyKeyboardMarkup(buttons, resize_keyboard=True)
keyboard_authorized = ReplyKeyboardMarkup(buttons_authorized, resize_keyboard=True)


def get_keyboard(update: Update):
    user_id = str(update.effective_user.id)
    authorized_users = os.environ.get("AUTHORIZED_USERS", "").split(",")
    return keyboard_authorized if user_id in authorized_users else keyboard


def add_daylight_background(axis, start_date, end_date, sun_times):
    if not sun_times:
        return

    twilight_begin = sun_times["civil_twilight_begin"]
    sunrise = sun_times["sunrise"]
    sunset = sun_times["sunset"]
    twilight_end = sun_times["civil_twilight_end"]
    axis.axvspan(
        start_date,
        twilight_begin,
        facecolor="#657384",
        alpha=0.24,
        zorder=0,
    )
    axis.axvspan(
        twilight_begin,
        sunrise,
        facecolor="#aeb8c4",
        alpha=0.18,
        zorder=0,
    )
    axis.axvspan(
        sunset,
        twilight_end,
        facecolor="#aeb8c4",
        alpha=0.18,
        zorder=0,
    )
    axis.axvspan(
        twilight_end,
        end_date,
        facecolor="#657384",
        alpha=0.24,
        zorder=0,
    )


async def greet_user(update: Update, context: ContextTypes):
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Hallo! Ich bin der Wetterbot. Ich kann dir aktuelle Wetterdaten von der Wetterstation in Bühlau senden. \n"
        "Frag mich einfach nach den aktuellen Wetterdaten oder dem Wetter der letzten 7 Tage. \n"
        "Dafür kannst du diese Befehle an mich senden: \n"
        "/now - aktuelle Wetterdaten \n"
        "/week - Wetter der letzten 7 Tage",
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


@restricted
async def retrieve_24h_weather_plot(update: Update, context: ContextTypes):
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

    figure, base_axis = plt.subplots(figsize=(12, 5.5), facecolor="#f4f7fb")
    base_axis.set_facecolor("#ffffff")
    try:
        sun_times = api.get_sunrise_sunset()
    except Exception:
        sun_times = None
    add_daylight_background(base_axis, start_date, end_date, sun_times)
    axes = [base_axis]
    for _ in range(1, len(plots)):
        axes.append(base_axis.twinx())

    colors = ("#1769aa", "#d97706", "#16803c")
    lines = []
    labels = []
    for index, (axis, (name, unit, measurements)) in enumerate(zip(axes, plots)):
        dates = [datetime.fromisoformat(measurement["date"]) for measurement in measurements]
        values = [measurement["value"] for measurement in measurements]
        axis.set_facecolor("none")
        (line,) = axis.plot(
            dates,
            values,
            color=colors[index % len(colors)],
            linewidth=3,
            solid_capstyle="round",
            solid_joinstyle="round",
        )
        axis.margins(y=0.18)
        lower_axis_limit, upper_axis_limit = axis.get_ylim()
        axis.fill_between(
            dates,
            values,
            lower_axis_limit,
            color=colors[index % len(colors)],
            alpha=0.08,
        )
        axis.set_ylim(lower_axis_limit, upper_axis_limit)
        minimum_index = values.index(min(values))
        maximum_index = values.index(max(values))
        marker_color = colors[index % len(colors)]
        axis.scatter(
            [dates[minimum_index]],
            [values[minimum_index]],
            color="#ffffff",
            edgecolor=marker_color,
            linewidth=2,
            s=80,
            zorder=5,
            clip_on=True,
        )
        axis.scatter(
            [dates[maximum_index]],
            [values[maximum_index]],
            color=marker_color,
            edgecolor="#ffffff",
            linewidth=1.5,
            marker="D",
            s=70,
            zorder=5,
            clip_on=True,
        )
        axis.annotate(
            f"{values[minimum_index]:.2f} {unit}",
            (dates[minimum_index], values[minimum_index]),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=10,
            fontweight="bold",
            color=marker_color,
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.85,
            },
            annotation_clip=True,
        )
        axis.annotate(
            f"{values[maximum_index]:.2f} {unit}",
            (dates[maximum_index], values[maximum_index]),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=10,
            fontweight="bold",
            color=marker_color,
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.85,
            },
            annotation_clip=True,
        )
        lines.append(line)
        labels.append(name)
        axis.set_ylabel(f"{name} ({unit})", color=colors[index % len(colors)], fontsize=14)
        axis.tick_params(axis="y", colors=colors[index % len(colors)], labelsize=12, width=1.5)
        axis.spines["left" if index == 0 else "right"].set_color(colors[index % len(colors)])
        axis.spines["left" if index == 0 else "right"].set_linewidth(2)
        axis.spines["top"].set_visible(False)
        if index == 0:
            axis.spines["right"].set_visible(False)
        else:
            axis.spines["left"].set_visible(False)
        if index > 1:
            axis.spines["right"].set_position(("axes", 1.12 + (index - 2) * 0.1))

    base_axis.set_title(
        "Wetterdaten der letzten 24 Stunden",
        fontsize=20,
        pad=22,
        fontweight="bold",
        loc="left",
        color="#172033",
    )
    base_axis.set_xlabel("Zeit", fontsize=14)
    base_axis.tick_params(axis="x", labelsize=12, width=1.5, colors="#526176")
    base_axis.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    base_axis.xaxis.set_major_formatter(mdates.DateFormatter("%Hh"))
    base_axis.grid(True, axis="y", color="#dbe3ed", alpha=0.8, linewidth=1)
    base_axis.grid(True, axis="x", color="#edf1f5", alpha=0.9, linewidth=0.8)
    base_axis.spines["bottom"].set_linewidth(1.5)
    base_axis.spines["bottom"].set_color("#9aa8b8")
    base_axis.set_xlim(start_date - timedelta(minutes=15), end_date + timedelta(minutes=15))
    figure.legend(
        lines,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.96),
        ncol=len(lines),
        frameon=False,
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.9))

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".png", prefix="weather-24h-", delete=False
        ) as temporary_file:
            temporary_path = temporary_file.name
        figure.savefig(
            temporary_path,
            format="png",
            dpi=180,
            facecolor=figure.get_facecolor(),
            bbox_inches="tight",
        )
        await context.bot.send_photo(
            chat_id=update.effective_chat.id,
            photo=temporary_path,
            caption="Wetterdaten der letzten 24 Stunden",
        )
    finally:
        plt.close(figure)
        if temporary_path:
            os.unlink(temporary_path)


@restricted
async def retrieve_forecast_plot(update: Update, context: ContextTypes):
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Die Wettervorhersage ist noch nicht implementiert.",
        parse_mode="Markdown",
    )
