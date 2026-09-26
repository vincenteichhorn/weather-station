import os

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters
from dotenv import load_dotenv

from replies import greet_user, provide_help, retrieve_weather_now, retrieve_last_week_weather, respond_to_unknown

load_dotenv()

if __name__ == "__main__":
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    application = ApplicationBuilder().token(token).build()

    application.add_handler(CommandHandler("start", greet_user))
    application.add_handler(CommandHandler("help", provide_help))
    application.add_handler(CommandHandler("now", retrieve_weather_now))
    application.add_handler(CommandHandler("week", retrieve_last_week_weather))
    application.add_handler(MessageHandler(filters.COMMAND, respond_to_unknown))

    application.run_polling()
