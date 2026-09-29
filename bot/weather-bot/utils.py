from functools import wraps
import os


def restricted(func):
    @wraps(func)
    async def wrapped(update, context, *args, **kwargs):
        user_id = str(update.effective_user.id)
        authorized_users = os.environ.get("AUTHORIZED_USERS", "").split(",")
        if user_id not in authorized_users:
            await update.bot.send_message(
                chat_id=update.effective_chat.id,
                text="Sorry, du darfst diesen Befehl nicht verwenden.",
            )
            return
        return await func(update, context, *args, **kwargs)

    return wrapped
