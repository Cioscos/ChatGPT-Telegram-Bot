import html
import json
import logging
import re
import traceback
import uuid
from typing import Union, List, Dict, Set, Optional
from warnings import filterwarnings
from itertools import zip_longest

import openai
from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove, InlineKeyboardButton, InlineKeyboardMarkup, User, \
    CallbackQuery
from telegram.constants import ParseMode
from telegram.error import BadRequest
from telegram.ext import (
    Application,
    CommandHandler,
    ConversationHandler,
    MessageHandler,
    ContextTypes,
    filters, CallbackQueryHandler, PicklePersistence
)
from telegram.warnings import PTBUserWarning

from environment_variables_mg import keyring_get, keyring_initialize
from src.openai_lib_wrapper import OpenAiLibWrapper
from utility import format_code_response
from personality import PERSONALITIES

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%y-%m-%d %H:%M:%S',
    filename='chatgptbot.log',
    filemode='w'
)

filterwarnings(action="ignore", message=r".*CallbackQueryHandler", category=PTBUserWarning)

# set higher logging level for httpx to avoid all GET and POST requests being logged
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

MODEL_CHOSE, CHAT, ACTIONS, CHAT_SELECTION, APPROVAL, NOT_APPROVED, INTRODUCTION, PERSONALITY_CHOICE = range(8)

OPENAI_GPT3 = 'gpt-3.5-turbo'
OPENAI_GPT4 = 'gpt-4'
USER_DATA_KEY_HISTORY = 'ai_chat_conversation_history'
USER_DATA_KEY_MODEL = 'model'
USER_DATA_KEY_ID = 'id'
USER_DATA_CURRENT_CHAT_ID = 'current_chat_id'
USER_DATA_TITLE = 'title'
USER_DATA_TITLE_CHOSEN = 'chosen_title'
USER_DATA_COMES_FROM_CHAT = 'comes_from_chat'
USER_DATA_TEMP_MESSAGES = 'temp_messages'
USER_DATA_NOT_APPROVED = 'not_approved'
USER_DATA_PERSONALITY = 'personality'

BOT_DATA_UPDATE_AND_CONTEXT_FROM_USER_TO_APPROVE = 'update_from_user_to_approve'
BOT_DATA_APPROVED_USERS = 'approved_users'
BOT_DATA_APPROVE_MESSAGE = 'approve_message'

CALLBACK_APPROVAL_APPROVE_USER = 'callback_data-approval-ok'
CALLBACK_APPROVAL_NOT_APPROVE_USER = 'callback_data-approval-notok'

TELEGRAM_MAX_MSG_LENGTH = 4096


def markdown_escape(text: str) -> str:
    # Escape special characters in text
    escape_chars = '_[]()~`>#+-=|{}.!'
    for char in escape_chars:
        text = text.replace(char, f'\\{char}')
    return text


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    The callback called when the bot receive the classical start command on a new conversation.
    """

    user: User = update.message.from_user
    user_name = user.username
    name = user.name
    user_id = user.id

    # check if the username is set
    if user_name is None:
        await update.message.reply_text('Hi! To use this  bot you have to set an username.\n'
                                        'Press please /start again after doing so.')
        return ConversationHandler.END

    # check if the user id is already saved in the bot_data
    if str(user_id) in context.bot_data.get(BOT_DATA_APPROVED_USERS, set()):
        return await introduction(update, context)

    await update.message.reply_text("<b>Hi! This is a telegram bot which imitates the behaviour of ChatGPT.</b>\n\n"
                                    "🚦 This bot uses OpenAI API to generates responses for you 🚦\n"
                                    "🚦 so the bot's owner must approve you to use it 🚦\n\n"
                                    "Please wait for approval...", parse_mode=ParseMode.HTML)

    # send a message to the bot developer
    dev_id: str = keyring_get('DevId')

    # Preparing the inline keyboard
    keyboard: List[List[InlineKeyboardButton]] = [
        [InlineKeyboardButton('Approve', callback_data=str(CALLBACK_APPROVAL_APPROVE_USER + ':' + str(user_id))),
         InlineKeyboardButton('Not approve', callback_data=str(CALLBACK_APPROVAL_NOT_APPROVE_USER + ':' + str(user_id)))]
    ]
    markup = InlineKeyboardMarkup(keyboard)

    msg_text: str = f"User {name if name else ''} (user name: @{user_name}) requested access to the bot."
    approve_message = await context.bot.send_message(dev_id, msg_text, reply_markup=markup)

    # save the original update in the bot_data
    context.bot_data[BOT_DATA_UPDATE_AND_CONTEXT_FROM_USER_TO_APPROVE] = (update, context)
    context.bot_data[BOT_DATA_APPROVE_MESSAGE] = approve_message

    return APPROVAL


async def approval(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Handles the approval from the dev
    """
    query: CallbackQuery = update.callback_query
    await query.answer()
    data: str = query.data

    user_id, approval_status = data.split(':')[1], data.split(':')[0]
    original_user_update: Optional[Update]
    original_user_context: Optional[ContextTypes.DEFAULT_TYPE]
    original_user_update, original_user_context = context.bot_data.pop(BOT_DATA_UPDATE_AND_CONTEXT_FROM_USER_TO_APPROVE, tuple())

    # delete approval message
    await original_user_context.bot_data[BOT_DATA_APPROVE_MESSAGE].delete()

    if approval_status == CALLBACK_APPROVAL_APPROVE_USER:
        await context.bot.send_message(chat_id=user_id, text="You have been approved. You can now use the bot.")

        approved_users = context.bot_data.setdefault(BOT_DATA_APPROVED_USERS, set())
        approved_users.add(user_id)

        await introduction(original_user_update, context)

    elif approval_status == CALLBACK_APPROVAL_NOT_APPROVE_USER:
        await context.bot.send_message(chat_id=user_id, text="You have not been approved. Goodbye.")
        original_user_context.user_data[USER_DATA_NOT_APPROVED] = True


async def approval_message_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Works when the dev still didn't make a decision about the approval
    """
    user_id = update.message.from_user.id
    if str(user_id) in context.bot_data.get(BOT_DATA_APPROVED_USERS, set()):
        return await model(update, context)

    if context.user_data.get(USER_DATA_NOT_APPROVED, False):
        await update.message.reply_text("The developer refused your approval. There is no way to use the bot")
    else:
        await update.message.reply_text("The Dev still didn't decide about you approval. Plase wait.")


async def introduction(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    It introduces the user after the approval from the Bot's owner
    """
    reply_keyboard = [['GTP-3.5 Turbo', 'GPT-4']]

    await update.message.reply_text(
        "Hi! This is a telegram bot which imitates the behaviour of ChatGPT."
        "Please choose the model you want to use.",
        reply_markup=ReplyKeyboardMarkup(
            reply_keyboard, resize_keyboard=True, one_time_keyboard=True, input_field_placeholder='GPT-3.5 or GPT4?'
        )
    )

    return MODEL_CHOSE


async def actions(update: Update, context: ContextTypes.DEFAULT_TYPE, comes_from_chat: bool = False) -> Union[int, None]:
    """
    Handles actions based on user input.

    Args:
        update (Update): Update object containing the incoming message.
        context (ContextTypes.DEFAULT_TYPE): Context object that holds runtime data.
        comes_from_chat (bool): True it actions call has been called from chat function

    Returns:
        Union[int, None]: State of the chat session.
    """

    message: str = update.message.text

    if message == 'New chat':
        if comes_from_chat:
            context.user_data[USER_DATA_COMES_FROM_CHAT] = True

            reply_keyboard = [['GTP-3.5 Turbo', 'GPT-4']]
            await update.message.reply_text(
                "Please select a model for the new conversation!",
                reply_markup=ReplyKeyboardMarkup(reply_keyboard, resize_keyboard=True, one_time_keyboard=True)
            )

        structure_single_chat = {
            'title': None,
            'model': None,
            'history': []
        }

        # Generate a new UUID for the new chat
        new_uuid: str = str(uuid.uuid4())
        user_data: Dict = context.user_data
        user_data[USER_DATA_KEY_ID] = new_uuid

        # Initialize or update chat history
        chat_history: Dict = user_data.get(USER_DATA_KEY_HISTORY, {})
        chat_history[new_uuid] = structure_single_chat
        user_data[USER_DATA_KEY_HISTORY] = chat_history

        # Save the model chosen in the chat
        current_chat: Dict = chat_history[new_uuid]
        current_chat[USER_DATA_KEY_MODEL] = user_data.get(USER_DATA_KEY_MODEL)

        # Save the current session to the just created chat
        user_data[USER_DATA_CURRENT_CHAT_ID] = new_uuid

        if not comes_from_chat:
            await update.message.reply_text("New chat created!\nPlease send a message to start the chat!")

        return MODEL_CHOSE if comes_from_chat else CHAT

    elif message == 'Delete chat':
        chat_history: Dict = context.user_data.get(USER_DATA_KEY_HISTORY)
        current_chat_id: str = context.user_data.get(USER_DATA_CURRENT_CHAT_ID)

        if chat_history and current_chat_id and chat_history.get(current_chat_id):
            del chat_history[current_chat_id]

            if chat_history:  # If there are still chats left
                reply_keyboard = [['GTP-3.5 Turbo', 'GPT-4'], ['Select a chat']]
                await update.message.reply_text(
                    "You've deleted the current chat.\nPlease select an old chat or create a new one by selecting the model.",
                    reply_markup=ReplyKeyboardMarkup(reply_keyboard, resize_keyboard=True, one_time_keyboard=True)
                )
                return CHAT_SELECTION
            else:  # If no chats left
                reply_keyboard = [['GTP-3.5 Turbo', 'GPT-4']]
                await update.message.reply_text(
                    "You've deleted the current chat. Please start a new chat by selecting a model.",
                    reply_markup=ReplyKeyboardMarkup(reply_keyboard, resize_keyboard=True, one_time_keyboard=True)
                )
                return MODEL_CHOSE

        else:
            reply_keyboard = [['GTP-3.5 Turbo', 'GPT-4']]
            await update.message.reply_text(
                "You're not in any chat. Please start a new one by selecting a model.",
                reply_markup=ReplyKeyboardMarkup(reply_keyboard, resize_keyboard=True, one_time_keyboard=True)
            )
            return MODEL_CHOSE

    elif message == 'Select a chat':
        chat_history: Dict = context.user_data.get(USER_DATA_KEY_HISTORY)

        if chat_history:
            current_page = context.user_data.get('current_page', 0)
            reply_markup = generate_chat_list_keyboard(chat_history, current_page)
            await update.message.reply_text("Select a chat:", reply_markup=reply_markup)
            return CHAT_SELECTION
        else:
            await update.message.reply_text("No chats available.")
            return ACTIONS
    elif message in ['GTP-3.5 Turbo', 'GPT-4']:
        return await model(update, context)

    elif message == 'Personality':
        """
        Handles the 'Personality' message event to create an inline keyboard with personalities.
        PERSONALITIES: A dictionary containing available personalities.
        message: The incoming message text.
        update: The message update object.

        Returns:
            The next state for the chat.
        """
        personalities = list(PERSONALITIES.keys())
        # Group elements in pairs using zip_longest
        grouped_personality = list(zip_longest(*[iter(personalities)] * 2, fillvalue=None))

        # Build the keyboard
        keyboard = [
            [InlineKeyboardButton(personality, callback_data=f"personality:{personality}") for personality in group if
             personality is not None]
            for group in grouped_personality
        ]

        inline_keyboard = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text("👤 Chosen Personality for ChatGPT", reply_markup=inline_keyboard)

        return PERSONALITY_CHOICE

    else:
        current_chat_id = context.user_data.get(USER_DATA_CURRENT_CHAT_ID)

        if current_chat_id:
            return await chat(update, context)
        else:
            await update.message.reply_text("You are not in any conversation. Please start a new one.")


def generate_chat_list_keyboard(chat_history, page):
    items_per_page = 6
    keyboard = []

    start_index = page * items_per_page
    end_index = start_index + items_per_page

    for chat_id in list(chat_history.keys())[start_index:end_index]:
        # Assuming chat has a 'title' attribute, change it as per your structure
        chat_title = chat_history[chat_id].get('title', chat_id)
        keyboard.append([InlineKeyboardButton(chat_title, callback_data=f"select_chat:{chat_id}")])

    # Add navigation buttons if necessary
    navigation_buttons = []
    if page > 0:
        navigation_buttons.append(InlineKeyboardButton("Previous", callback_data="prev_page"))
    if end_index < len(chat_history):
        navigation_buttons.append(InlineKeyboardButton("Next", callback_data="next_page"))

    if navigation_buttons:
        keyboard.append(navigation_buttons)

    return InlineKeyboardMarkup(keyboard)


async def to_actions(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    # call again action function
    return await actions(update, context, True)


async def personality_choice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    data = query.data
    user_data = context.user_data

    if data.startswith("personality:"):
        personality = data.split(':')[1]
        user_data[USER_DATA_PERSONALITY] = personality

        await query.message.reply_text(f"You chose {personality.lower()} personality")

        return CHAT_SELECTION


async def chat_selection_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> Union[int, None]:
    """
    Handles callback queries for chat selection.

    Args:
        update (Update): Update object containing the incoming callback query.
        context (ContextTypes.DEFAULT_TYPE): Context object that holds runtime data.

    Returns:
        Union[int, None]: Next state in the chat workflow.
    """

    query = update.callback_query
    await query.answer()

    data = query.data

    user_data = context.user_data

    # Initialize current_page if it's not already defined
    if 'current_page' not in user_data:
        user_data['current_page'] = 0

    if data.startswith("select_chat:"):
        chat_id = data.split(":")[1]
        user_data[USER_DATA_CURRENT_CHAT_ID] = chat_id

        # Retrieve title of the selected chat
        chat_history = user_data.get(USER_DATA_KEY_HISTORY, {})
        selected_chat = chat_history.get(chat_id, {})  # Intermediate object for the selected chat
        selected_title = selected_chat.get('title', "Unknown")

        reply_keyboard = [['New chat', 'Delete chat'], ['Select a chat']]

        await context.bot.send_message(chat_id=query.from_user.id,
                                       text=f"You have selected the conversation with title:\n{selected_title}\n"
                                            f"Use /history to read the conversation's message history.",
                                       reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True,
                                                                        resize_keyboard=True))
        return CHAT

    elif data == "prev_page":
        user_data['current_page'] -= 1
    elif data == "next_page":
        user_data['current_page'] += 1

    # Update the InlineKeyboard with the new page data
    chat_history = user_data.get(USER_DATA_KEY_HISTORY, {})
    current_page = user_data.get('current_page', 0)
    reply_markup = generate_chat_list_keyboard(chat_history, current_page)
    await query.edit_message_text("Select a chat:", reply_markup=reply_markup)


async def model(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Stores the model chose by the user
    """
    message = update.message.text
    logger.info("User choosed model: %s", message)

    user_data = context.user_data

    if message in ['GTP-3.5 Turbo', 'GPT-4']:
        reply_keyboard = [['New chat', 'Delete chat'], ['Select a chat', 'Personality']]

        if context.user_data.get(USER_DATA_COMES_FROM_CHAT, False):
            # If model function is called from actions function, means that actions function
            # has been called during a chat.
            # In this way we can manage to directly call action inside model function and not to wait another
            # message which won't arrive from the user
            await update.message.reply_text(f"Ok, I will use {message} as model.\n"
                                            "Please send a message to start the chat!",
                                            reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True,
                                                                             resize_keyboard=True)
                                            )
            current_chatid = user_data[USER_DATA_CURRENT_CHAT_ID]
            conversation = user_data[USER_DATA_KEY_HISTORY]
            user_data[USER_DATA_COMES_FROM_CHAT] = False
            conversation[current_chatid][USER_DATA_KEY_MODEL] = message

            return CHAT
        else:
            # We arrive here basically only from start callback
            if message in ['GTP-3.5 Turbo', 'GPT-4']:
                await update.message.reply_text(f"Ok, I will use {message} as model.\nPlease chose what to do:\n"
                                                f"Do you want to start a new chat or finish an old one?",
                                                reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True,
                                                                                 resize_keyboard=True,
                                                                                 input_field_placeholder='New chat or delete one chat?'))

                user_data[USER_DATA_KEY_MODEL] = OPENAI_GPT3 if message == 'GTP-3.5 Turbo' else OPENAI_GPT4

    return ACTIONS


async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> Union[int, None]:
    """Manage the chat with the model"""
    user_message = update.message.text

    # change status of the chat if the message is new chat or some other command
    if user_message in ['New chat', 'Delete chat', 'Select a chat', 'Personality']:
        # Directly call the actions function and return the resulting state
        return await actions(update, context, comes_from_chat=True)

    # Initialize commonly accessed user data and message history for readability
    user_data = context.user_data
    current_chat_id = user_data[USER_DATA_CURRENT_CHAT_ID]
    current_chat = user_data[USER_DATA_KEY_HISTORY][current_chat_id]
    message_history = current_chat['history']

    # check length of message to understand if it can be potentially a part of a longer text
    temp_messages = current_chat.get(USER_DATA_TEMP_MESSAGES, [])
    if len(user_message) == TELEGRAM_MAX_MSG_LENGTH:
        # get the list of temp messages
        temp_messages.append(user_message)
        # update the list in the user_data dictionary
        current_chat[USER_DATA_TEMP_MESSAGES] = temp_messages

        return CHAT
    else:
        # if the message is not long enough, concatenate the temp messages saved if there are
        if temp_messages:
            final_message = " ".join(temp_messages)
            current_chat[USER_DATA_TEMP_MESSAGES] = []
        else:
            final_message = user_message

    system_message = {
        'role': 'system',
        'content': 'You are a bot in a telegram chat '
                   'So you will formulate the messages taking into account the way telegram handles markdown. '
                   'Also consider that I am using python-telegram-bot as a library for the bot so consider it\'s way '
                   'to format markdown too. '
                   'Answer to the messages in the most completed way possible, creating examples, numbered list if it\'s possible '
                   'and sending pieces of code when it\'s the case'
    }

    # Append to final_message the personality body
    personality = user_data.get(USER_DATA_PERSONALITY)
    personality_body = PERSONALITIES.get(personality, '') if personality is not None else ''
    final_message += personality_body
    new_message_body = {'role': 'user', 'content': final_message}

    if len(message_history) == 0:
        message_history.extend([system_message, new_message_body])
    else:
        message_history.append(new_message_body)

    telegram_message = await update.message.reply_text("Thinking...")

    response = OpenAiLibWrapper.chat_completition(
        model=context.user_data[USER_DATA_KEY_MODEL],
        messages=message_history,
        temperature=0.2
    )

    if response['choices'][0]['finish_reason'] == 'stop':

        # Get the AI response content
        ai_response = response['choices'][0]['message']['content']
        ai_message_body = {'role': 'assistant', 'content': ai_response}

        if not current_chat.get(USER_DATA_TITLE_CHOSEN, None):

            await telegram_message.edit_text('Generating title...')

            title_prompt = ("Conversation:\n"
                            f"User: {final_message}\n"
                            f"GPT: {ai_response}\n\n"
                            "Please provide a succinct and compelling title for the above conversation. "
                            "Use less articles and words possible. Use Maximum 5 words.")

            title_conversation = [
                {'role': 'user',
                 'content': title_prompt}
            ]
            title_response = OpenAiLibWrapper.chat_completition(
                model='gpt-3.5-turbo',
                messages=title_conversation,
                temperature=0
            )

            if title_response['choices'][0]['finish_reason'] == 'stop':
                # Get the AI response content
                title = title_response['choices'][0]['message']['content']

                current_chat['title'] = title

                # Means that the title has been assigned to the conversation
                current_chat[USER_DATA_TITLE_CHOSEN] = True
            else:
                await telegram_message.edit_text("Error with OpenAI API. Please wait some seconds and retry.")
                # Remove the last element from the history since the message won't be read from the API
                message_history.pop(-1)
                return

        # format the message to eventually send pieces of code correctly
        ai_response = format_code_response(ai_response)

        # Append the AI response to user_data
        message_history.append(ai_message_body)

        # Splitting the message into parts if it exceeds the Telegram limit
        start = 0
        messages = []
        while start < len(ai_response):
            end = min(start + 4096, len(ai_response))
            messages.append(ai_response[start:end])
            start = end

        # Send the parts separately via Telegram
        for index, msg in enumerate(messages):
            if index == 0:
                try:
                    # Edit the original message for the first part
                    await telegram_message.edit_text(msg, parse_mode=ParseMode.MARKDOWN_V2)
                except BadRequest:
                    await telegram_message.edit_text(msg)
            else:
                try:
                    # Reply to the original message for the subsequent parts
                    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN_V2)
                except BadRequest:
                    # Reply to the original message for the subsequent parts
                    await update.message.reply_text(msg)
    else:
        logger.warning("Error with OpeanAI request")
        await telegram_message.edit_text('There was an error, please try again')

    return CHAT


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancels and ends the conversation."""
    user = update.message.from_user
    logger.info("User %s canceled the conversation.", user.first_name)

    context.application.drop_user_data(user.id)
    approved_users:Set = context.application.bot_data.pop(BOT_DATA_APPROVED_USERS, set())

    if str(user.id) in approved_users:
        approved_users.remove(str(user.id))

    await update.message.reply_text(
        "Bye! I hope we can talk again some day.", reply_markup=ReplyKeyboardRemove()
    )

    return ConversationHandler.END


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    The error callback function.
    This function is used to handle possible Telegram API errors that aren't handled.

    :param update: The Telegram update.
    :param context: The Telegram context.
    """
    # Log the error before we do anything else, so we can see it even if something breaks.
    logger.error("Exception while handling an update:", exc_info=context.error)

    # traceback.format_exception returns the usual python message about an exception, but as a
    # list of strings rather than a single string, so we have to join them together.
    tb_list = traceback.format_exception(None, context.error, context.error.__traceback__)
    tb_string = "".join(tb_list)

    # Split the traceback into smaller parts
    tb_parts = [tb_string[i: i + 4096] for i in range(0, len(tb_string), 4096)]

    # Build the message with some markup and additional information about what happened.
    update_str = update.to_dict() if isinstance(update, Update) else str(update)
    base_message = (
        f"An exception was raised while handling an update\n"
        f"<pre>update = {html.escape(json.dumps(update_str, indent=2, ensure_ascii=False))}"
        "</pre>\n\n"
        f"<pre>context.chat_data = {html.escape(str(context.chat_data))}</pre>\n\n"
        f"<pre>context.user_data = {html.escape(str(context.user_data))}</pre>\n\n"
    )

    # Send base message
    await context.bot.send_message(
        chat_id=keyring_get('DevId'), text=base_message, parse_mode=ParseMode.HTML
    )

    # Send each part of the traceback as a separate message
    for part in tb_parts:
        await context.bot.send_message(
            chat_id=keyring_get('DevId'), text=f"<pre>{html.escape(part)}</pre>", parse_mode=ParseMode.HTML
        )


async def history_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    conversation_history: str = 'This is the history of this conversation:\n\n'

    message_headers: Dict[str, str] = {
        'user': 'You:',
        'assistant': 'Me:'
    }

    # Initialize commonly accessed user data and message history for readability
    user_data: Dict = context.user_data
    current_chat_id: str = user_data[USER_DATA_CURRENT_CHAT_ID]
    current_chat: Dict = user_data[USER_DATA_KEY_HISTORY][current_chat_id]
    message_history: List[Dict[str, str]] = current_chat['history']

    for message in message_history:
        message_role: str = message.get('role', '')
        message_header = message_headers.get(message_role)
        if message_header:
            conversation_history += f"*{message_headers[message_role]}* {message.get('content')}\n"

    await update.message.reply_text(markdown_escape(conversation_history), parse_mode=ParseMode.MARKDOWN_V2)

    return CHAT


async def personality_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Questa è una prova")


def filter_approval(data: Union[str, object]) -> bool:
    return re.match(r'callback_data-approval-(ok|notok)', data) is not None


def main() -> None:
    # Initialize the keyring
    if not keyring_initialize():
        exit(0xFF)

    # Initialize the Pickle database
    my_persistence = PicklePersistence(filepath='DB')

    OpenAiLibWrapper.set_api_key(keyring_get('OpenAI'))
    OpenAiLibWrapper.set_timeout(60)

    # Initialize Application
    application = Application.builder().token(keyring_get('Telegram')).persistence(
        persistence=my_persistence).concurrent_updates(True).build()

    # Assign an error handler
    application.add_error_handler(error_handler)

    # Add the conversation handler
    chatgpt_handler = ConversationHandler(
        persistent=True,
        name='chatGPT_handler_v3',
        entry_points=[CommandHandler('start', start)],
        states={
            APPROVAL: [MessageHandler(filters.TEXT & ~filters.COMMAND, approval_message_callback)],
            MODEL_CHOSE: [MessageHandler(filters.Regex("^(GTP-3.5 Turbo|GPT-4)$"), model)],
            PERSONALITY_CHOICE: [CallbackQueryHandler(personality_choice)],
            CHAT: [MessageHandler(filters.TEXT & ~filters.COMMAND, chat),
                   CommandHandler('history', history_callback)],
            CHAT_SELECTION: [CallbackQueryHandler(chat_selection_callback),
                             MessageHandler(filters.TEXT & ~filters.COMMAND, to_actions),
                             CommandHandler('history', history_callback)],
            ACTIONS: [MessageHandler(filters.Regex("^(New chat|Delete chat|Select a chat|Personality)$"), actions)]
        },
        fallbacks=[CommandHandler("cancel", cancel)]
    )
    application.add_handler(chatgpt_handler)

    approval_handler = CallbackQueryHandler(approval, pattern=filter_approval)
    application.add_handler(approval_handler)

    # Start the bot polling
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == '__main__':
    main()
