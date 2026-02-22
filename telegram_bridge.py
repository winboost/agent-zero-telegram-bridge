"""
Agent Zero <-> Telegram Bot Bridge (REST transport)
Uses REST /api_message for message delivery with per-user project isolation.
Each user is automatically assigned a project (e.g. tg_mo) via resolve_project().

Usage:
    docker exec -it agent-zero /opt/venv/bin/python3 /a0/usr/workdir/telegram_bridge.py

Requirements (inside container):
    /opt/venv/bin/pip install aiohttp python-telegram-bot python-dotenv
"""

import sys
import os
import asyncio
import base64
import logging
import traceback
import mimetypes
import uuid
import re
from pathlib import Path

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

load_dotenv("/a0/usr/.env")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
A0_BASE_URL    = os.getenv("A0_API_URL", "http://127.0.0.1:80").rstrip("/api_message").rstrip("/")
A0_API_URL     = f"{A0_BASE_URL}/api_message"
A0_RESET_URL   = f"{A0_BASE_URL}/api_reset_chat"
A0_TERMINATE_URL = f"{A0_BASE_URL}/api_terminate_chat"
A0_LOG_URL     = f"{A0_BASE_URL}/api_log_get"
A0_TIMEOUT     = int(os.getenv("A0_TIMEOUT", "300"))

ALLOWED_CHATS    = os.getenv("TELEGRAM_CHAT_IDS", "")
ALLOWED_CHAT_SET = set(ALLOWED_CHATS.split(",")) if ALLOWED_CHATS.strip() else set()
ALLOWED_USERS    = os.getenv("TELEGRAM_USER_IDS", "")
ALLOWED_USER_SET = set(ALLOWED_USERS.split(",")) if ALLOWED_USERS.strip() else set()

TELEGRAM_MAX_LEN = 4096
MAX_FILE_BYTES   = int(os.getenv("A0_MAX_FILE_BYTES", str(20 * 1024 * 1024)))
UPLOAD_DIR       = Path(os.getenv("A0_UPLOAD_DIR", "/a0/usr/workdir/uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_LENGTH   = int(os.getenv("A0_HISTORY_LENGTH", "20"))

AUTO_PROJECT_ENABLED      = os.getenv("AUTO_PROJECT_ENABLED", "true").lower() == "true"
AUTO_PROJECT_PREFIX       = os.getenv("AUTO_PROJECT_PREFIX", "tg_")
AUTO_PROJECT_USE_USERNAME = os.getenv("AUTO_PROJECT_USE_USERNAME", "true").lower() == "true"

# ---------------------------------------------------------------------------
# API key discovery
# ---------------------------------------------------------------------------

def get_a0_api_key() -> str:
    env_key = os.getenv("A0_API_KEY", "")
    if env_key:
        return env_key
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
# State
# ---------------------------------------------------------------------------

chat_contexts: dict[str, str]          = {}  # chat_id -> A0 context_id
active_tasks:  dict[str, asyncio.Task] = {}  # chat_id -> asyncio.Task
user_projects: dict[str, str]          = {}  # user_id -> manual project override

# ---------------------------------------------------------------------------
# Project helpers
# ---------------------------------------------------------------------------

def get_auto_project(user_id: str, username: str | None = None) -> str:
    if AUTO_PROJECT_USE_USERNAME and username:
        slug = re.sub(r"[^a-zA-Z0-9_-]", "_", username)
    else:
        slug = user_id
    return f"{AUTO_PROJECT_PREFIX}{slug}"


def resolve_project(user_id: str, username: str | None = None) -> str | None:
    if user_id in user_projects:
        return user_projects[user_id]
    if AUTO_PROJECT_ENABLED:
        return get_auto_project(user_id, username)
    return None

# ---------------------------------------------------------------------------
# REST helpers
# ---------------------------------------------------------------------------

def _headers() -> dict:
    return {
        "Content-Type": "application/json",
        "X-API-KEY": A0_API_KEY,
        "X-Forwarded-For": "127.0.0.1",
        "X-Real-IP": "127.0.0.1",
    }


async def a0_post(url: str, payload: dict, timeout: int = 30) -> dict:
    async with aiohttp.ClientSession() as session:
        async with session.post(
            url, json=payload, headers=_headers(),
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            if resp.status == 200:
                return await resp.json()
            raise RuntimeError(f"REST {url} HTTP {resp.status}: {(await resp.text())[:300]}")


async def send_to_agent(
    message_text: str,
    context_id: str = "",
    saved_files: list[dict] | None = None,
    project: str | None = None,
) -> dict:
    """
    Send a message to Agent Zero via /api_message.
    Files are sent as base64 attachments (correct A0 API format).
    project_name is only sent when context_id is empty (first message of a new context).
    """
    # Build attachments array from saved files
    attachments = []
    if saved_files:
        for f in saved_files:
            try:
                raw = Path(f["path"]).read_bytes()
                attachments.append({
                    "filename": f["filename"],
                    "base64": base64.b64encode(raw).decode(),
                })
            except Exception as e:
                log.warning(f"Could not encode {f['filename']} for API: {e}")
                # Fallback: path reference in message text
                message_text += f"\n[FILE name={f['filename']} mime={f['mime_type']} path={f['path']}]"

    payload: dict = {"message": message_text, "context_id": context_id}

    if attachments:
        payload["attachments"] = attachments

    if project:
        payload["project_name"] = project  # A0 API uses project_name, not project
        log.info(f"Starting new context with project_name: {project}")

    return await a0_post(A0_API_URL, payload, timeout=A0_TIMEOUT)


async def reset_context(context_id: str) -> bool:
    try:
        await a0_post(A0_RESET_URL, {"context_id": context_id})
        log.info(f"Reset context {context_id}")
        return True
    except Exception as e:
        log.warning(f"Reset failed: {e}")
        return False


async def terminate_context(context_id: str) -> bool:
    try:
        await a0_post(A0_TERMINATE_URL, {"context_id": context_id})
        log.info(f"Terminated context {context_id}")
        return True
    except Exception as e:
        log.warning(f"Terminate failed: {e}")
        return False


async def get_log(context_id: str, length: int = HISTORY_LENGTH) -> list[dict]:
    try:
        data = await a0_post(A0_LOG_URL, {"context_id": context_id, "length": length})
        return data.get("log", {}).get("items", [])
    except Exception as e:
        log.warning(f"Log fetch failed: {e}")
        return []

# ---------------------------------------------------------------------------
# Task cancellation
# ---------------------------------------------------------------------------

async def cancel_active_task(chat_id: str):
    task = active_tasks.get(chat_id)
    if task and not task.done():
        context_id = chat_contexts.get(chat_id, "")
        if context_id:
            await terminate_context(context_id)
            chat_contexts.pop(chat_id, None)
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
        log.info(f"Cancelled task for chat {chat_id}")

# ---------------------------------------------------------------------------
# Auth / formatting
# ---------------------------------------------------------------------------

def is_authorized(update: Update) -> bool:
    chat_id = str(update.effective_chat.id)
    user_id = str(update.effective_user.id)
    if ALLOWED_CHAT_SET and chat_id not in ALLOWED_CHAT_SET:
        return False
    if ALLOWED_USER_SET and user_id not in ALLOWED_USER_SET:
        log.warning(f"Blocked user {user_id} in chat {chat_id}")
        return False
    return True


def split_message(text: str, limit: int = TELEGRAM_MAX_LEN) -> list[str]:
    if len(text) <= limit:
        return [text]
    chunks = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        pos = text.rfind("\n", 0, limit)
        if pos == -1:
            pos = text.rfind(" ", 0, limit)
        if pos == -1:
            pos = limit
        chunks.append(text[:pos])
        text = text[pos:].lstrip("\n")
    return chunks


def markdown_to_telegram(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r'```(?:\w+)?\n?(.*?)```', r'<pre>\1</pre>', text, flags=re.DOTALL)
    text = re.sub(r'^#{1,6}\s+(.+)$', r'<b>\1</b>', text, flags=re.MULTILINE)
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'__(.+?)__', r'<b>\1</b>', text)
    text = re.sub(r'\*(?!\*)(.+?)(?<!\*)\*', r'<i>\1</i>', text)
    text = re.sub(r'_(?!_)(.+?)(?<!_)_', r'<i>\1</i>', text)
    text = re.sub(r'~~(.+?)~~', r'<s>\1</s>', text)
    text = re.sub(r'`([^`\n]+)`', r'<code>\1</code>', text)
    text = re.sub(r'\[([^\]]+)\]\(((?!file://|img://)[^\)]+)\)', r'<a href="\2">\1</a>', text)
    return text


def extract_file_paths(text: str) -> tuple[list[str], str]:
    paths: list[str] = []
    cleaned = text

    def add(p: str):
        p = p.strip().rstrip(")")
        if p and p not in paths:
            paths.append(p)

    for m in re.finditer(r'!\[.*?\]\(img://(/[^\)]+)\)', text):
        add(m.group(1))
    cleaned = re.sub(r'!\[.*?\]\(img://[^\)]+\)', '', cleaned)
    for m in re.finditer(r'\[.*?\]\(file:///([^\)]+)\)', text):
        add("/" + m.group(1))
    cleaned = re.sub(r'\[.*?\]\(file:///[^\)]+\)', '', cleaned)
    for m in re.finditer(r'\[FILE\b[^\]]*\bpath=([^\]\s]+)', text):
        add(m.group(1))
    cleaned = re.sub(r'\[FILE\b[^\]]*\]', '', cleaned)
    for m in re.finditer(r'\*{0,2}`(/[^`]+\.[a-zA-Z0-9]{1,5})`\*{0,2}', text):
        add(m.group(1))
    valid = [p for p in paths if Path(p).is_file()]
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()
    return valid, cleaned

# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

async def save_attachments(message: Message, bot) -> tuple[list[dict], list[str]]:
    saved = []
    errors = []

    async def process(file_id: str, filename: str, forced_mime: str | None = None):
        try:
            tg_file = await bot.get_file(file_id)
            raw = bytes(await tg_file.download_as_bytearray())
            if len(raw) > MAX_FILE_BYTES:
                mb = len(raw) / (1024 * 1024)
                errors.append(f"{filename}: {mb:.1f} MB exceeds configured limit of {MAX_FILE_BYTES // (1024*1024)} MB")
                log.warning(f"Skipping {filename}: too large ({mb:.1f} MB)")
                return
            mime = forced_mime or "application/octet-stream"
            if not forced_mime and tg_file.file_path:
                guessed, _ = mimetypes.guess_type(tg_file.file_path)
                if guessed:
                    mime = guessed
            stem, _, ext = filename.rpartition(".")
            ext = f".{ext}" if ext else ""
            unique_name = f"{stem}_{uuid.uuid4().hex[:8]}{ext}" if stem else f"{uuid.uuid4().hex}{ext}"
            dest = UPLOAD_DIR / unique_name
            dest.write_bytes(raw)
            saved.append({"path": str(dest), "filename": filename, "mime_type": mime})
        except Exception as e:
            err_str = str(e)
            if "too big" in err_str.lower() or "file is too big" in err_str.lower():
                errors.append(f"{filename}: Telegram Bot API limits file downloads to 20 MB.")
            else:
                errors.append(f"{filename}: {e}")
            log.error(f"Failed to save {filename}: {e}")

    if message.document:
        doc = message.document
        await process(doc.file_id, doc.file_name or f"document_{doc.file_id[:8]}", doc.mime_type or None)
    if message.photo:
        best = max(message.photo, key=lambda p: p.file_size or 0)
        await process(best.file_id, f"photo_{best.file_id[:8]}.jpg", "image/jpeg")
    if message.audio:
        audio = message.audio
        await process(audio.file_id, audio.file_name or f"audio_{audio.file_id[:8]}", audio.mime_type or None)
    if message.voice:
        await process(message.voice.file_id, f"voice_{message.voice.file_id[:8]}.ogg", "audio/ogg")
    if message.video:
        video = message.video
        await process(video.file_id, video.file_name or f"video_{video.file_id[:8]}.mp4", video.mime_type or "video/mp4")
    if message.video_note:
        await process(message.video_note.file_id, f"videonote_{message.video_note.file_id[:8]}.mp4", "video/mp4")
    if message.sticker:
        ext = ".webm" if (message.sticker.is_animated or message.sticker.is_video) else ".webp"
        await process(message.sticker.file_id, f"sticker_{message.sticker.file_id[:8]}{ext}")
    return saved, errors


async def send_files_to_telegram(update: Update, file_paths: list[str]):
    IMAGE_MIMES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
    VIDEO_MIMES = {"video/mp4", "video/mpeg"}
    AUDIO_MIMES = {"audio/mpeg", "audio/ogg", "audio/wav", "audio/mp4"}
    for path_str in file_paths:
        path = Path(path_str)
        if not path.is_file():
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
        except Exception as e:
            log.error(f"Failed to send {path_str}: {e}")
            await update.message.reply_text(
                f"⚠️ Could not send file <code>{path.name}</code>.", parse_mode="HTML"
            )

# ---------------------------------------------------------------------------
# Core send
# ---------------------------------------------------------------------------

async def forward_to_agent_and_reply(
    update: Update,
    text: str,
    saved_files: list[dict],
    context_id: str,
    chat_id: str,
    user_display: str,
    project: str | None,
):
    log.info(
        f"[{user_display}] → REST (project={project or 'none'}): "
        f"{text[:80]}{'...' if len(text) > 80 else ''} ({len(saved_files)} file(s))"
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
        data = await send_to_agent(text, context_id, saved_files, project)
    finally:
        typing_active = False
        typing_task.cancel()
        try:
            await typing_task
        except asyncio.CancelledError:
            pass

    # Persist context for conversation continuity
    new_context_id = data.get("context_id", "")
    if new_context_id:
        chat_contexts[chat_id] = new_context_id

    reply = (data.get("response") or "").strip() or "(Agent returned an empty response)"
    log.info(f"REST → [{user_display}]: {reply[:100]}{'...' if len(reply) > 100 else ''}")

    file_paths, reply_text = extract_file_paths(reply)

    if reply_text.strip():
        converted = markdown_to_telegram(reply_text)
        for chunk in split_message(converted):
            try:
                await update.message.reply_text(chunk, parse_mode="HTML")
            except Exception:
                await update.message.reply_text(reply_text[:4096])

    if file_paths:
        await send_files_to_telegram(update, file_paths)

# ---------------------------------------------------------------------------
# Command Handlers
# ---------------------------------------------------------------------------

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        return
    await update.message.reply_text(
        "🤖 <b>Agent Zero Bridge</b>\n\n"
        "Send me any message and I'll forward it to Agent Zero.\n"
        "You can also send <b>files, photos, audio, and video</b>.\n\n"
        "<b>Commands:</b>\n"
        "/reset — Start a new conversation\n"
        "/history — Show recent conversation history\n"
        "/project set &lt;name&gt; — Assign an A0 project\n"
        "/project clear — Remove manual project assignment\n"
        "/status — Show connection status\n"
        "/help — Show this message",
        parse_mode="HTML",
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await cmd_start(update, context)


async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        return
    chat_id = str(update.effective_chat.id)
    await cancel_active_task(chat_id)
    context_id = chat_contexts.get(chat_id, "")
    if context_id:
        await reset_context(context_id)
        chat_contexts.pop(chat_id, None)
    await update.message.reply_text("🔄 Conversation reset. Starting fresh.")


async def cmd_history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        return
    chat_id = str(update.effective_chat.id)
    context_id = chat_contexts.get(chat_id, "")
    if not context_id:
        await update.message.reply_text("No active conversation yet.")
        return
    await update.effective_chat.send_action(ChatAction.TYPING)
    entries = await get_log(context_id, length=HISTORY_LENGTH)
    if not entries:
        await update.message.reply_text("No history found.")
        return
    lines = [f"<b>Last {len(entries)} messages:</b>\n"]
    for entry in entries:
        role = entry.get("role", entry.get("type", "?")).capitalize()
        content = entry.get("content", entry.get("message", ""))
        if isinstance(content, list):
            content = " ".join(b.get("text", "") for b in content if isinstance(b, dict))
        content = str(content).strip()
        if content:
            preview = content[:300] + "…" if len(content) > 300 else content
            lines.append(f"<b>{role}:</b> {preview}\n")
    for chunk in split_message(markdown_to_telegram("\n".join(lines))):
        try:
            await update.message.reply_text(chunk, parse_mode="HTML")
        except Exception:
            await update.message.reply_text(chunk)


async def cmd_project(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        return
    chat_id = str(update.effective_chat.id)
    user_id = str(update.effective_user.id)
    user = update.effective_user
    args = context.args or []

    if not args:
        current = user_projects.get(user_id)
        auto = get_auto_project(user_id, user.username) if AUTO_PROJECT_ENABLED else None
        effective = current or auto or "(none)"
        note = " <i>(manual)</i>" if current else " <i>(auto)</i>" if auto else ""
        await update.message.reply_text(
            f"🗂 <b>Project</b>\n"
            f"Effective: <code>{effective}</code>{note}\n\n"
            f"• <code>/project set &lt;name&gt;</code>\n"
            f"• <code>/project clear</code>",
            parse_mode="HTML",
        )
        return

    sub = args[0].lower()

    if sub == "set":
        if len(args) < 2:
            await update.message.reply_text("Usage: <code>/project set &lt;name&gt;</code>", parse_mode="HTML")
            return
        name = args[1]
        user_projects[user_id] = name
        await cancel_active_task(chat_id)
        ctx = chat_contexts.pop(chat_id, "")
        if ctx:
            await reset_context(ctx)
        await update.message.reply_text(
            f"🗂 Project set to <code>{name}</code>. Conversation reset.", parse_mode="HTML"
        )

    elif sub == "clear":
        removed = user_projects.pop(user_id, None)
        if removed:
            await cancel_active_task(chat_id)
            ctx = chat_contexts.pop(chat_id, "")
            if ctx:
                await reset_context(ctx)
            auto = get_auto_project(user_id, user.username) if AUTO_PROJECT_ENABLED else None
            fb = f" Falling back to <code>{auto}</code>." if auto else ""
            await update.message.reply_text(
                f"🗂 Removed <code>{removed}</code>.{fb} Conversation reset.", parse_mode="HTML"
            )
        else:
            await update.message.reply_text("No manual project assigned.")
    else:
        await update.message.reply_text(
            "Use <code>/project set &lt;name&gt;</code> or <code>/project clear</code>.", parse_mode="HTML"
        )


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        return
    chat_id = str(update.effective_chat.id)
    user_id = str(update.effective_user.id)
    user = update.effective_user
    project = resolve_project(user_id, user.username)
    ctx = chat_contexts.get(chat_id, "(none)")
    active = "yes" if chat_id in active_tasks and not active_tasks[chat_id].done() else "no"
    await update.message.reply_text(
        f"🤖 <b>Bot Status</b>\n"
        f"• API: <code>{A0_API_URL}</code>\n"
        f"• Context: <code>{ctx}</code>\n"
        f"• Project: <code>{project or '(none)'}</code>\n"
        f"• Processing: {active}\n"
        f"• Timeout: {A0_TIMEOUT}s\n"
        f"• Max file: {MAX_FILE_BYTES // (1024 * 1024)} MB\n"
        f"• Upload dir: <code>{UPLOAD_DIR}</code>",
        parse_mode="HTML",
    )

# ---------------------------------------------------------------------------
# Message Handlers
# ---------------------------------------------------------------------------

async def _dispatch(
    update: Update,
    chat_id: str,
    context_id: str,
    user_display: str,
    text: str,
    saved_files: list[dict],
    project: str | None,
    processing_msg=None,
):
    async def run():
        try:
            if processing_msg:
                try:
                    await processing_msg.delete()
                except Exception:
                    pass
            await forward_to_agent_and_reply(
                update, text, saved_files, context_id, chat_id, user_display, project
            )
        except asyncio.CancelledError:
            if processing_msg:
                try:
                    await processing_msg.delete()
                except Exception:
                    pass
            await update.message.reply_text("⚡ Request cancelled.")
        except asyncio.TimeoutError:
            await update.message.reply_text(
                f"⏳ Agent Zero timed out (>{A0_TIMEOUT}s). Try again or /reset."
            )
        except aiohttp.ClientConnectorError as e:
            log.error(f"Connection error: {e}")
            await update.message.reply_text(
                f"🔌 Cannot connect to Agent Zero.\nTarget: <code>{A0_API_URL}</code>",
                parse_mode="HTML",
            )
        except Exception as e:
            log.error(traceback.format_exc())
            await update.message.reply_text(f"❌ Error: {str(e)[:500]}")
        finally:
            active_tasks.pop(chat_id, None)

    task = asyncio.create_task(run())
    active_tasks[chat_id] = task


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return
    if not is_authorized(update):
        return
    chat_id = str(update.effective_chat.id)
    content = update.message.text.strip()
    if not content:
        return
    user = update.effective_user
    user_display = user.username or user.first_name or str(user.id)
    user_id = str(user.id)
    await cancel_active_task(chat_id)
    context_id = chat_contexts.get(chat_id, "")
    project = resolve_project(user_id, user.username)
    user_prefix = f"[User: {user_display} | chat_id: {chat_id}] "
    await _dispatch(update, chat_id, context_id, user_display,
                    user_prefix + content, [], project)


async def handle_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message:
        return
    if not is_authorized(update):
        return
    chat_id = str(update.effective_chat.id)
    user = update.effective_user
    user_display = user.username or user.first_name or str(user.id)
    user_id = str(user.id)
    await cancel_active_task(chat_id)
    context_id = chat_contexts.get(chat_id, "")
    project = resolve_project(user_id, user.username)
    caption = (update.message.caption or "").strip() or "I am sending you a file. Please analyse it."
    user_prefix = f"[User: {user_display} | chat_id: {chat_id}] "
    processing_msg = await update.message.reply_text("📎 Downloading file(s)…")
    saved_files, errors = await save_attachments(update.message, context.bot)
    if not saved_files:
        error_detail = "\n• " + "\n• ".join(errors) if errors else ""
        await processing_msg.edit_text(f"⚠️ Could not save any attachments.{error_detail}")
        return
    await _dispatch(update, chat_id, context_id, user_display,
                    user_prefix + caption, saved_files, project, processing_msg)

# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not TELEGRAM_TOKEN:
        print("ERROR: TELEGRAM_BOT_TOKEN not found in /a0/usr/.env")
        sys.exit(1)
    if not A0_API_KEY:
        print("ERROR: Could not determine Agent Zero API key.")
        sys.exit(1)

    print("=" * 60)
    print("  Agent Zero <-> Telegram Bridge  [REST transport]")
    print("=" * 60)
    print(f"  API URL:      {A0_API_URL}")
    print(f"  API Key:      {A0_API_KEY[:4]}****")
    print(f"  Timeout:      {A0_TIMEOUT}s")
    print(f"  Max file:     {MAX_FILE_BYTES // (1024 * 1024)} MB")
    print(f"  Upload dir:   {UPLOAD_DIR}")
    print(f"  History len:  {HISTORY_LENGTH}")
    print(f"  Auto-project: {'enabled' if AUTO_PROJECT_ENABLED else 'disabled'}", end="")
    if AUTO_PROJECT_ENABLED:
        id_source = "username" if AUTO_PROJECT_USE_USERNAME else "user_id"
        print(f" (prefix={AUTO_PROJECT_PREFIX!r}, id={id_source})")
    else:
        print()
    if ALLOWED_USER_SET:
        print(f"  Users:        {', '.join(sorted(ALLOWED_USER_SET))}")
    else:
        print("  Users:        (all)")
    print("=" * 60)

    app = Application.builder().token(TELEGRAM_TOKEN).build()
    file_filter = (
        filters.Document.ALL | filters.PHOTO | filters.AUDIO |
        filters.VOICE | filters.VIDEO | filters.VIDEO_NOTE | filters.Sticker.ALL
    )
    app.add_handler(CommandHandler("start",   cmd_start))
    app.add_handler(CommandHandler("help",    cmd_help))
    app.add_handler(CommandHandler("reset",   cmd_reset))
    app.add_handler(CommandHandler("history", cmd_history))
    app.add_handler(CommandHandler("project", cmd_project))
    app.add_handler(CommandHandler("status",  cmd_status))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(file_filter, handle_file))

    log.info("Starting Telegram bot (REST transport)…")
    app.run_polling(allowed_updates=Update.ALL_TYPES)
