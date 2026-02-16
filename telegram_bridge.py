"""
Agent Zero <-> Telegram Bot Bridge
Bridges Telegram messages to Agent Zero's /api_message HTTP API.

Usage:
    docker exec -it agent-zero /opt/venv/bin/python3 /a0/usr/workdir/telegram_bridge.py

Requirements (inside container):
    /opt/venv/bin/pip install aiohttp python-telegram-bot python-dotenv
"""

import sys
import os
import asyncio
import logging
import traceback

# Insert A0 path so we can import settings to auto-discover the API key
sys.path.insert(0, "/a0")

import aiohttp
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

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Load environment from A0's .env file
load_dotenv("/a0/usr/.env")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")

# Agent Zero API configuration
A0_API_URL = os.getenv("A0_API_URL", "http://127.0.0.1:80/api_message")
A0_TIMEOUT = int(os.getenv("A0_TIMEOUT", "300"))  # seconds (agent can be slow)

# Optional: restrict the bot to specific chat IDs (comma-separated).
# If empty, the bot responds in ALL chats.
ALLOWED_CHATS = os.getenv("TELEGRAM_CHAT_IDS", "")
ALLOWED_CHAT_SET = (
    set(ALLOWED_CHATS.split(",")) if ALLOWED_CHATS.strip() else set()
)

# Telegram message length limit
TELEGRAM_MAX_LEN = 4096

# ---------------------------------------------------------------------------
# Auto-discover Agent Zero API key from runtime settings
# ---------------------------------------------------------------------------


def get_a0_api_key() -> str:
    """
    Try to read the mcp_server_token from Agent Zero's settings module.
    Falls back to A0_API_KEY env var if import fails.
    """
    # First check env var
    env_key = os.getenv("A0_API_KEY", "")
    if env_key:
        return env_key

    # Auto-discover from A0 settings
    try:
        from python.helpers.settings import get_settings

        token = get_settings().get("mcp_server_token", "")
        if token:
            return token
    except Exception as e:
        print(f"[WARN] Could not auto-discover API key from A0 settings: {e}")

    return ""


A0_API_KEY = get_a0_api_key()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("telegram_bridge")

# ---------------------------------------------------------------------------
# Conversation context mapping: Telegram chat ID -> Agent Zero context_id
# This gives each chat its own persistent conversation with the agent.
# ---------------------------------------------------------------------------

chat_contexts: dict[str, str] = {}


def split_message(text: str, limit: int = TELEGRAM_MAX_LEN) -> list[str]:
    """Split a long message into chunks that fit Telegram's character limit."""
    if len(text) <= limit:
        return [text]

    chunks = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break

        # Try to split at a newline
        split_pos = text.rfind("\n", 0, limit)
        if split_pos == -1:
            # Try to split at a space
            split_pos = text.rfind(" ", 0, limit)
        if split_pos == -1:
            # Hard split
            split_pos = limit

        chunks.append(text[:split_pos])
        text = text[split_pos:].lstrip("\n")

    return chunks


async def send_to_agent(message_text: str, context_id: str = "") -> dict:
    """
    Send a message to Agent Zero's /api_message endpoint.
    Returns the parsed JSON response dict.
    """
    payload = {
        "message": message_text,
        "context_id": context_id,
    }
    headers = {
        "Content-Type": "application/json",
        "X-API-KEY": A0_API_KEY,
    }

    timeout = aiohttp.ClientTimeout(total=A0_TIMEOUT)

    async with aiohttp.ClientSession() as session:
        async with session.post(
            A0_API_URL, json=payload, headers=headers, timeout=timeout
        ) as resp:
            if resp.status == 200:
                return await resp.json()
            else:
                error_text = await resp.text()
                raise RuntimeError(
                    f"Agent Zero returned HTTP {resp.status}: {error_text[:500]}"
                )


# ---------------------------------------------------------------------------
# Command Handlers
# ---------------------------------------------------------------------------


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command."""
    await update.message.reply_text(
        "🤖 *Agent Zero Bridge*\n\n"
        "Send me any message and I'll forward it to Agent Zero.\n\n"
        "*Commands:*\n"
        "/reset — Start a new conversation\n"
        "/status — Show connection status\n"
        "/help — Show this message",
        parse_mode="Markdown",
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help command."""
    await cmd_start(update, context)


async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /reset command — clears conversation context."""
    chat_id = str(update.effective_chat.id)
    chat_contexts.pop(chat_id, None)
    await update.message.reply_text("🔄 Conversation reset. Starting fresh.")
    log.info(f"Context reset for chat {chat_id}")


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /status command."""
    chat_id = str(update.effective_chat.id)
    ctx = chat_contexts.get(chat_id, "(none)")
    await update.message.reply_text(
        f"🤖 *Bot Status*\n"
        f"• API: `{A0_API_URL}`\n"
        f"• Context: `{ctx}`\n"
        f"• Timeout: {A0_TIMEOUT}s",
        parse_mode="Markdown",
    )


# ---------------------------------------------------------------------------
# Message Handler
# ---------------------------------------------------------------------------


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Forward user messages to Agent Zero and send back the response."""
    # Ignore if no text
    if not update.message or not update.message.text:
        return

    chat_id = str(update.effective_chat.id)

    # Chat filter
    if ALLOWED_CHAT_SET and chat_id not in ALLOWED_CHAT_SET:
        return

    content = update.message.text.strip()
    if not content:
        return

    context_id = chat_contexts.get(chat_id, "")
    user = update.effective_user
    user_display = user.username or user.first_name or str(user.id)

    log.info(
        f"[{user_display}] → Agent Zero: {content[:100]}{'...' if len(content) > 100 else ''}"
    )

    # Show typing indicator
    try:
        await update.effective_chat.send_action(ChatAction.TYPING)

        # Start a typing keep-alive task (Telegram typing expires after ~5s)
        typing_active = True

        async def keep_typing():
            while typing_active:
                try:
                    await asyncio.sleep(4)
                    if typing_active:
                        await update.effective_chat.send_action(ChatAction.TYPING)
                except Exception:
                    break

        typing_task = asyncio.create_task(keep_typing())

        try:
            data = await send_to_agent(content, context_id)
        finally:
            typing_active = False
            typing_task.cancel()
            try:
                await typing_task
            except asyncio.CancelledError:
                pass

        # Store context ID for conversation continuity
        new_context = data.get("context_id", "")
        if new_context:
            chat_contexts[chat_id] = new_context

        reply = data.get("response", "")
        if not reply:
            reply = "(Agent returned an empty response)"

        log.info(
            f"Agent Zero → [{user_display}]: {reply[:100]}{'...' if len(reply) > 100 else ''}"
        )

        # Send response, splitting if needed
        chunks = split_message(reply)
        for chunk in chunks:
            await update.message.reply_text(chunk)

    except asyncio.TimeoutError:
        log.warning(f"Timeout waiting for Agent Zero (>{A0_TIMEOUT}s)")
        await update.message.reply_text(
            f"⏳ Agent Zero took too long to respond (timeout: {A0_TIMEOUT}s). "
            f"Try again or use /reset to start fresh."
        )

    except aiohttp.ClientConnectorError as e:
        log.error(f"Connection error: {e}")
        await update.message.reply_text(
            f"🔌 Cannot connect to Agent Zero API. Is the server running?\n"
            f"Target: `{A0_API_URL}`",
            parse_mode="Markdown",
        )

    except Exception as e:
        log.error(f"Error: {traceback.format_exc()}")
        await update.message.reply_text(f"❌ Error: {str(e)[:500]}")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not TELEGRAM_TOKEN:
        print("ERROR: TELEGRAM_BOT_TOKEN not found in /a0/usr/.env")
        sys.exit(1)

    if not A0_API_KEY:
        print("ERROR: Could not determine Agent Zero API key.")
        print("Set A0_API_KEY in /a0/usr/.env or ensure A0 settings are accessible.")
        sys.exit(1)

    print("=" * 60)
    print("  Agent Zero <-> Telegram Bridge")
    print("=" * 60)
    print(f"  API URL:  {A0_API_URL}")
    print(f"  API Key:  {A0_API_KEY[:4]}****")
    print(f"  Timeout:  {A0_TIMEOUT}s")
    print("=" * 60)

    # Build and run the Telegram bot
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    # Register handlers
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("reset", cmd_reset))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    log.info("Starting Telegram bot...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)
