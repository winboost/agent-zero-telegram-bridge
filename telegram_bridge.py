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
import mimetypes
import uuid
import re
from pathlib import Path

# Insert A0 path so we can import settings to auto-discover the API key
sys.path.insert(0, "/a0")

import aiohttp
from dotenv import load_dotenv
from telegram import Update, Message
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
ALLOWED_CHAT_SET = set(ALLOWED_CHATS.split(",")) if ALLOWED_CHATS.strip() else set()

# Optional: restrict the bot to specific Telegram user IDs (comma-separated).
# User IDs are numeric (e.g. 123456789) and never change, unlike usernames.
# Find your user ID by messaging @userinfobot on Telegram.
# If empty, the bot responds to ALL users (subject to TELEGRAM_CHAT_IDS above).
ALLOWED_USERS = os.getenv("TELEGRAM_USER_IDS", "")
ALLOWED_USER_SET = set(ALLOWED_USERS.split(",")) if ALLOWED_USERS.strip() else set()

# Telegram message length limit
TELEGRAM_MAX_LEN = 4096

# Max file size to forward to Agent Zero (default 20 MB)
MAX_FILE_BYTES = int(os.getenv("A0_MAX_FILE_BYTES", str(20 * 1024 * 1024)))

# Directory inside the Agent Zero container where uploaded files are saved.
# Must be writable and accessible to the agent's working context.
# Default matches A0's typical workdir; override via A0_UPLOAD_DIR in .env.
UPLOAD_DIR = Path(os.getenv("A0_UPLOAD_DIR", "/a0/usr/workdir/uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

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

# ---------------------------------------------------------------------------
# Active task tracking: Telegram chat ID -> currently running asyncio Task
# A new message cancels any in-flight task for the same chat.
# ---------------------------------------------------------------------------

active_tasks: dict[str, asyncio.Task] = {}


async def cancel_active_task(chat_id: str):
    """Cancel any currently running agent task for this chat."""
    task = active_tasks.get(chat_id)
    if task and not task.done():
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
        log.info(f"Cancelled in-flight task for chat {chat_id}")


def is_authorized(update: Update) -> bool:
    """Check if the incoming update passes chat and user ID filters."""
    chat_id = str(update.effective_chat.id)
    user_id = str(update.effective_user.id)

    if ALLOWED_CHAT_SET and chat_id not in ALLOWED_CHAT_SET:
        return False

    if ALLOWED_USER_SET and user_id not in ALLOWED_USER_SET:
        log.warning(f"Blocked unauthorized user {user_id} in chat {chat_id}")
        return False

    return True


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


def markdown_to_telegram(text: str) -> str:
    """
    Convert common AI markdown to Telegram MarkdownV2 format.
    Telegram MarkdownV2 requires escaping many special characters, so we:
      1. Convert the markdown constructs we want to keep
      2. Escape everything else that MarkdownV2 treats as special
    """
    import html

    # We'll use Telegram's simpler HTML parse mode instead of MarkdownV2
    # because HTML is much easier to generate correctly without double-escaping issues.

    # Escape HTML special chars first (we're targeting parse_mode="HTML")
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # Headers → <b>text</b> with a newline (Telegram has no heading support)
    text = re.sub(r'^#{1,6}\s+(.+)$', r'<b>\1</b>', text, flags=re.MULTILINE)

    # Bold: **text** or __text__ → <b>text</b>
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'__(.+?)__', r'<b>\1</b>', text)

    # Italic: *text* or _text_ → <i>text</i>
    text = re.sub(r'\*(?!\*)(.+?)(?<!\*)\*', r'<i>\1</i>', text)
    text = re.sub(r'_(?!_)(.+?)(?<!_)_', r'<i>\1</i>', text)

    # Strikethrough: ~~text~~ → <s>text</s>
    text = re.sub(r'~~(.+?)~~', r'<s>\1</s>', text)

    # Inline code: `code` → <code>code</code>
    text = re.sub(r'`([^`\n]+)`', r'<code>\1</code>', text)

    # Code blocks: ```lang\ncode\n``` → <pre>code</pre>
    text = re.sub(r'```(?:\w+)?\n?(.*?)```', r'<pre>\1</pre>', text, flags=re.DOTALL)

    # Links: [label](url) → <a href="url">label</a>
    # Skip file:// and img:// links (handled by extract_file_paths)
    text = re.sub(
        r'\[([^\]]+)\]\(((?!file://|img://)[^\)]+)\)',
        r'<a href="\2">\1</a>',
        text
    )

    return text


def extract_file_paths(text: str) -> tuple[list[str], str]:
    """
    Scan agent response text for file references and return:
      - list of absolute paths to send as Telegram files
      - cleaned text with file link/tag markers removed

    Patterns detected:
      ![alt](img:///absolute/path.jpg)          ← A0 image markdown
      [label](file:///absolute/path.xlsx)       ← markdown link with file:// scheme
      [FILE path=/absolute/path.ext]            ← our own upload tag (round-trip)
      **`/a0/usr/workdir/file.ext`**            ← bold-backtick absolute path
      `/a0/usr/workdir/file.ext`                ← backtick absolute path
    """
    paths: list[str] = []
    cleaned = text

    def add(p: str):
        p = p.strip().rstrip(')')
        if p and p not in paths:
            paths.append(p)

    # 1. Markdown image with img:// scheme  e.g. ![x](img:///a0/usr/workdir/out.jpg)
    for m in re.finditer(r'!\[.*?\]\(img://(/[^\)]+)\)', text):
        add(m.group(1))
    cleaned = re.sub(r'!\[.*?\]\(img://[^\)]+\)', '', cleaned)

    # 2. Markdown link with file:// scheme  e.g. [label](file:///a0/usr/workdir/out.xlsx)
    for m in re.finditer(r'\[.*?\]\(file:///([^\)]+)\)', text):
        add('/' + m.group(1))
    cleaned = re.sub(r'\[.*?\]\(file:///[^\)]+\)', '', cleaned)

    # 3. Our own [FILE ... path=...] tags
    for m in re.finditer(r'\[FILE\b[^\]]*\bpath=([^\]\s]+)', text):
        add(m.group(1))
    cleaned = re.sub(r'\[FILE\b[^\]]*\]', '', cleaned)

    # 4. Absolute paths in backticks (with optional surrounding bold **)
    #    e.g. **`/a0/usr/workdir/out.xlsx`**  or  `/a0/usr/workdir/out.jpg`
    for m in re.finditer(r'\*{0,2}`(/[^`]+\.[a-zA-Z0-9]{1,5})`\*{0,2}', text):
        add(m.group(1))
    # Leave these in the text — they're readable context for the user.

    # Filter to paths that actually exist on disk
    valid = [p for p in paths if Path(p).is_file()]
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()
    return valid, cleaned


async def send_files_to_telegram(update: Update, file_paths: list[str]):
    """
    Send a list of on-disk files back to the user via Telegram.
    Images/video are sent as native media; everything else as a document.
    """
    IMAGE_MIMES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
    VIDEO_MIMES = {"video/mp4", "video/mpeg"}
    AUDIO_MIMES = {"audio/mpeg", "audio/ogg", "audio/wav", "audio/mp4"}

    for path_str in file_paths:
        path = Path(path_str)
        if not path.is_file():
            log.warning(f"Skipping missing file: {path_str}")
            continue

        mime, _ = mimetypes.guess_type(path_str)
        mime = mime or "application/octet-stream"

        try:
            with open(path, "rb") as fh:
                if mime in IMAGE_MIMES:
                    await update.message.reply_photo(photo=fh, filename=path.name)
                elif mime in VIDEO_MIMES:
                    await update.message.reply_video(video=fh, filename=path.name)
                elif mime in AUDIO_MIMES:
                    await update.message.reply_audio(audio=fh, filename=path.name)
                else:
                    await update.message.reply_document(document=fh, filename=path.name)
            log.info(f"Sent file to Telegram: {path_str} ({mime})")
        except Exception as e:
            log.error(f"Failed to send {path_str} to Telegram: {e}")
            await update.message.reply_text(
                f"⚠️ Could not send file `{path.name}` — it may have been deleted.",
                parse_mode="Markdown",
            )


async def send_to_agent(message_text: str, context_id: str = "",
                        saved_files: list[dict] | None = None) -> dict:
    """
    Send a message to Agent Zero's /api_message endpoint.
    Returns the parsed JSON response dict.

    saved_files is an optional list of dicts, each with:
        {"path": str, "filename": str, "mime_type": str}
    Files must already be saved to disk before calling this function.
    Their absolute paths are appended to the message so the agent can
    read them directly without any decoding step.
    """
    if saved_files:
        parts = [message_text] if message_text else []
        for f in saved_files:
            parts.append(
                f"\n[FILE name={f['filename']} mime={f['mime_type']} path={f['path']}]"
            )
        message_text = "\n".join(parts)

    payload = {
        "message": message_text,
        "context_id": context_id,
    }
    headers = {
        "Content-Type": "application/json",
        "X-API-KEY": A0_API_KEY,
        # Identify the request as coming from localhost.
        # Required when a reverse proxy (e.g. Cloudflare tunnel) is active and
        # SearXNG's bot-detection middleware sits in front of port 80.
        "X-Forwarded-For": "127.0.0.1",
        "X-Real-IP": "127.0.0.1",
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
# File save helper
# ---------------------------------------------------------------------------


async def save_attachments(message: Message, bot) -> list[dict]:
    """
    Download all attachments from a Telegram message, save them to UPLOAD_DIR
    on disk, and return a list of dicts with their on-disk paths.

    Each dict contains:
        {"path": str, "filename": str, "mime_type": str}

    Files exceeding MAX_FILE_BYTES are skipped with a warning.
    """
    saved = []

    async def process(file_id: str, filename: str, forced_mime: str | None = None):
        try:
            tg_file = await bot.get_file(file_id)
            raw = bytes(await tg_file.download_as_bytearray())

            if len(raw) > MAX_FILE_BYTES:
                log.warning(
                    f"Skipping {filename}: {len(raw)} bytes exceeds limit {MAX_FILE_BYTES}"
                )
                return

            # Detect MIME type
            mime = forced_mime or "application/octet-stream"
            if not forced_mime and tg_file.file_path:
                guessed, _ = mimetypes.guess_type(tg_file.file_path)
                if guessed:
                    mime = guessed

            # Build a unique filename to avoid collisions
            stem, _, ext = filename.rpartition(".")
            ext = f".{ext}" if ext else ""
            unique_name = f"{stem}_{uuid.uuid4().hex[:8]}{ext}" if stem else f"{uuid.uuid4().hex}{ext}"
            dest = UPLOAD_DIR / unique_name

            dest.write_bytes(raw)
            log.info(f"Saved {filename} → {dest} ({mime}, {len(raw)} bytes)")

            saved.append({
                "path": str(dest),
                "filename": filename,
                "mime_type": mime,
            })

        except Exception as e:
            log.error(f"Failed to save {filename}: {e}")

    # Document (any file type)
    if message.document:
        doc = message.document
        filename = doc.file_name or f"document_{doc.file_id[:8]}"
        await process(doc.file_id, filename, doc.mime_type or None)

    # Photos — Telegram sends multiple resolutions; pick the largest
    if message.photo:
        best = max(message.photo, key=lambda p: p.file_size or 0)
        await process(best.file_id, f"photo_{best.file_id[:8]}.jpg", "image/jpeg")

    # Audio file
    if message.audio:
        audio = message.audio
        filename = audio.file_name or f"audio_{audio.file_id[:8]}"
        await process(audio.file_id, filename, audio.mime_type or None)

    # Voice message
    if message.voice:
        await process(
            message.voice.file_id,
            f"voice_{message.voice.file_id[:8]}.ogg",
            "audio/ogg",
        )

    # Video
    if message.video:
        video = message.video
        filename = video.file_name or f"video_{video.file_id[:8]}.mp4"
        await process(video.file_id, filename, video.mime_type or "video/mp4")

    # Video note (round video)
    if message.video_note:
        await process(
            message.video_note.file_id,
            f"videonote_{message.video_note.file_id[:8]}.mp4",
            "video/mp4",
        )

    # Sticker
    if message.sticker:
        ext = ".webm" if (message.sticker.is_animated or message.sticker.is_video) else ".webp"
        await process(
            message.sticker.file_id,
            f"sticker_{message.sticker.file_id[:8]}{ext}",
        )

    return saved


# ---------------------------------------------------------------------------
# Command Handlers
# ---------------------------------------------------------------------------


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command."""
    if not is_authorized(update):
        return
    await update.message.reply_text(
        "🤖 *Agent Zero Bridge*\n\n"
        "Send me any message and I'll forward it to Agent Zero.\n"
        "You can also send *files, photos, audio, and video* — they will be forwarded too.\n\n"
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
    if not is_authorized(update):
        return
    chat_id = str(update.effective_chat.id)
    chat_contexts.pop(chat_id, None)
    await update.message.reply_text("🔄 Conversation reset. Starting fresh.")
    log.info(f"Context reset for chat {chat_id}")


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /status command."""
    if not is_authorized(update):
        return
    chat_id = str(update.effective_chat.id)
    ctx = chat_contexts.get(chat_id, "(none)")
    await update.message.reply_text(
        f"🤖 *Bot Status*\n"
        f"• API: `{A0_API_URL}`\n"
        f"• Context: `{ctx}`\n"
        f"• Timeout: {A0_TIMEOUT}s\n"
        f"• Max file size: {MAX_FILE_BYTES // (1024*1024)} MB\n"
        f"• Upload dir: `{UPLOAD_DIR}`",
        parse_mode="Markdown",
    )


# ---------------------------------------------------------------------------
# Core send-with-typing helper (shared by text and file handlers)
# ---------------------------------------------------------------------------


async def forward_to_agent_and_reply(
    update: Update,
    text: str,
    saved_files: list[dict],
    context_id: str,
    chat_id: str,
    user_display: str,
):
    """Send content to Agent Zero with a typing indicator and reply to user."""
    log.info(
        f"[{user_display}] → Agent Zero: {text[:100]}{'...' if len(text) > 100 else ''} "
        f"({len(saved_files)} file(s))"
    )

    typing_active = True

    async def keep_typing():
        while typing_active:
            try:
                await asyncio.sleep(4)
                if typing_active:
                    await update.effective_chat.send_action(ChatAction.TYPING)
            except Exception:
                break

    await update.effective_chat.send_action(ChatAction.TYPING)
    typing_task = asyncio.create_task(keep_typing())

    try:
        data = await send_to_agent(text, context_id, saved_files or None)
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

    # Extract any file paths embedded in the response and send them as real files
    file_paths, reply_text = extract_file_paths(reply)

    # Send text reply (skip if it's empty after stripping file markers)
    if reply_text.strip():
        converted = markdown_to_telegram(reply_text)
        chunks = split_message(converted)
        for chunk in chunks:
            try:
                await update.message.reply_text(chunk, parse_mode="HTML")
            except Exception:
                # Fallback to plain text if HTML parsing fails
                await update.message.reply_text(reply_text)

    # Send each detected file back to the user
    if file_paths:
        await send_files_to_telegram(update, file_paths)


# ---------------------------------------------------------------------------
# Message Handlers
# ---------------------------------------------------------------------------


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Forward user text messages (with optional caption) to Agent Zero."""
    if not update.message or not update.message.text:
        return
    if not is_authorized(update):
        return

    chat_id = str(update.effective_chat.id)
    content = update.message.text.strip()
    if not content:
        return

    context_id = chat_contexts.get(chat_id, "")
    user = update.effective_user
    user_display = user.username or user.first_name or str(user.id)

    # Cancel any in-flight request for this chat
    await cancel_active_task(chat_id)

    # Prepend user identity so the agent knows who is writing and can use
    # the chat_id in any proactive scripts it generates
    user_prefix = f"[User: {user_display} | chat_id: {chat_id}] "
    content = user_prefix + content

    async def run():
        try:
            await forward_to_agent_and_reply(
                update, content, [], context_id, chat_id, user_display
            )
        except asyncio.CancelledError:
            log.info(f"Request cancelled for chat {chat_id}")
            await update.message.reply_text("⚡ Request cancelled.")
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
        finally:
            active_tasks.pop(chat_id, None)

    task = asyncio.create_task(run())
    active_tasks[chat_id] = task


async def handle_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handle messages that contain files (documents, photos, audio, voice,
    video, stickers). The caption (if any) is used as the text message.
    """
    if not update.message:
        return
    if not is_authorized(update):
        return

    chat_id = str(update.effective_chat.id)
    context_id = chat_contexts.get(chat_id, "")
    user = update.effective_user
    user_display = user.username or user.first_name or str(user.id)

    # Cancel any in-flight request for this chat
    await cancel_active_task(chat_id)

    # Use the caption as the accompanying text, or a default prompt
    caption = (update.message.caption or "").strip()
    if not caption:
        caption = "I am sending you a file. Please analyse it."

    # Prepend user identity
    user_prefix = f"[User: {user_display} | chat_id: {chat_id}] "
    caption = user_prefix + caption

    # Notify user we're downloading
    processing_msg = await update.message.reply_text("📎 Downloading file(s)…")

    async def run():
        try:
            saved_files = await save_attachments(update.message, context.bot)
            if not saved_files:
                await processing_msg.edit_text(
                    "⚠️ Could not save any attachments. "
                    "The file may be too large or unsupported."
                )
                return

            await processing_msg.delete()

            await forward_to_agent_and_reply(
                update, caption, saved_files, context_id, chat_id, user_display
            )
        except asyncio.CancelledError:
            log.info(f"File request cancelled for chat {chat_id}")
            try:
                await processing_msg.delete()
            except Exception:
                pass
            await update.message.reply_text("⚡ Request cancelled.")
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
        finally:
            active_tasks.pop(chat_id, None)

    task = asyncio.create_task(run())
    active_tasks[chat_id] = task


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
    print(f"  API URL:      {A0_API_URL}")
    print(f"  API Key:      {A0_API_KEY[:4]}****")
    print(f"  Timeout:      {A0_TIMEOUT}s")
    print(f"  Max file:     {MAX_FILE_BYTES // (1024*1024)} MB")
    print(f"  Upload dir:   {UPLOAD_DIR}")
    if ALLOWED_USER_SET:
        print(f"  Users:        {', '.join(sorted(ALLOWED_USER_SET))}")
    else:
        print("  Users:        (all)")
    print("=" * 60)

    # Build and run the Telegram bot
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    # File filter — matches any message that carries an attachment
    file_filter = (
        filters.Document.ALL
        | filters.PHOTO
        | filters.AUDIO
        | filters.VOICE
        | filters.VIDEO
        | filters.VIDEO_NOTE
        | filters.Sticker.ALL
    )

    # Register handlers
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("reset", cmd_reset))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(file_filter, handle_file))  # ← NEW: file handler

    log.info("Starting Telegram bot (with file support)…")
    app.run_polling(allowed_updates=Update.ALL_TYPES)
