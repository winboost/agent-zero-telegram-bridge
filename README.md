# Agent Zero Telegram Bot Bridge

A Telegram bot that bridges messages to [Agent Zero](https://github.com/frdel/agent-zero)'s HTTP API, enabling you to chat with your AI agent directly from Telegram.

![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## How It Works

This bot bridges Telegram to Agent Zero's HTTP API running inside the Docker container. Here are the key technical details:

**API Endpoint:** The bot sends messages to `POST /api_message` on port 80 (the Agent Zero web UI server, powered by Flask + Socket.IO via uvicorn). This is the dedicated external API endpoint — it does not use the web UI's WebSocket protocol or the tunnel API on port 55520.

**Authentication:** The `/api_message` endpoint requires an API key via the `X-API-KEY` header. The key (`mcp_server_token`) is deterministically generated at runtime from `sha256(runtime_id:username:password)`, truncated to 16 characters. The bot auto-discovers this key by importing Agent Zero's settings module at startup — no need to hardcode or manually update it.

**Conversation Continuity:** Agent Zero uses `context_id` to track conversations. The bot maps each Telegram chat to a unique `context_id`, so conversations persist across messages within the same chat. Use `/reset` to start a fresh context.

**Response Format:** The API returns `{"context_id": "...", "response": "..."}` synchronously — the request blocks until the agent finishes thinking (up to 5 minutes timeout). The bot shows a typing indicator during this wait.

**Runtime Environment:** The bot must run with `/opt/venv/bin/python3` (Agent Zero's virtual environment) so it can import the settings module for API key discovery. The script lives at `/a0/usr/workdir/telegram_bridge.py`.

---

## Prerequisites

1. Open Telegram and search for **@BotFather**
2. Send `/newbot` and follow the prompts to create your bot
3. Copy the bot token that BotFather gives you

You can utilize the respective section of [this tutorial](https://memu.bot/tutorial/telegram).

---

## How to Implement

### 1. Add your Telegram bot token

Copy the bot token and add `TELEGRAM_BOT_TOKEN=your_token_here` to `/a0/usr/.env` (no quotes around the token).

You can do this via the Files browser in the Agent Zero GUI.

### 2. Download and upload the bot script

Download [`telegram_bridge.py`](https://raw.githubusercontent.com/winboost/agent-zero-telegram-bridge/main/telegram_bridge.py) and upload it into `/a0/usr/workdir`.

Again, you can utilize the Files browser in Agent Zero.

### 3. Install dependencies

```bash
docker exec agent-zero /opt/venv/bin/pip install aiohttp python-telegram-bot python-dotenv
```

### 4. Launch the bot

I'd recommend launching with `-it` first to confirm everything starts cleanly:

```bash
docker exec -it agent-zero /opt/venv/bin/python3 /a0/usr/workdir/telegram_bridge.py
```

You should see:

```
============================================================
  Agent Zero <-> Telegram Bridge
============================================================
  API URL:  http://127.0.0.1:80/api_message
  API Key:  _tyX****
  Timeout:  300s
============================================================
Starting Telegram bot...
```

### 5. Test it

Open your Telegram bot and send `/start`, then send a message. The agent should reply!

### 6. Stop the test instance

Once you've confirmed the bot responds, kill the `-it` instance (closing the terminal doesn't always kill it):

```bash
docker exec agent-zero pkill -f telegram_bridge.py
```

If you ever need an instant kill, `-9` sends SIGKILL which can't be caught — the process dies immediately:

```bash
docker exec agent-zero pkill -9 -f telegram_bridge.py
```

### 7. Run in background (optional)

> Skip this step if you plan to use auto-start / supervisord (step 8) instead.

Switch to the `-d` approach to run it in background (since `-d` detaches immediately, you won't see the startup banner or any errors):

```bash
docker exec -d agent-zero /opt/venv/bin/python3 /a0/usr/workdir/telegram_bridge.py
```

### 8. Auto-start with container (recommended)

If you have a running bot instance (from step 7 or otherwise), kill it first:

```bash
docker exec agent-zero pkill -9 -f telegram_bridge.py
```

To make it auto-start with the container, add it to the container's supervisord config (which Agent Zero already uses to manage its processes). **Pick one of the two options below — not both.**

#### Option A: Paste a command (quick)

Copy and paste this single command into your terminal — it appends the config block automatically:

```bash
docker exec agent-zero bash -c 'printf "\n[program:telegram_bridge]\ncommand=/opt/venv/bin/python3 /a0/usr/workdir/telegram_bridge.py\nenvironment=\nuser=root\ndirectory=/a0\nstopwaitsecs=10\nstdout_logfile=/dev/stdout\nstdout_logfile_maxbytes=0\nstderr_logfile=/dev/stderr\nstderr_logfile_maxbytes=0\nautorestart=true\nstartretries=3\nstopasgroup=true\nkillasgroup=true\n" >> /etc/supervisor/conf.d/supervisord.conf'
```

#### Option B: Edit the file manually

Open `/etc/supervisor/conf.d/supervisord.conf` via the Files browser in the Agent Zero GUI and add this block at the end of the file:

```ini
[program:telegram_bridge]
command=/opt/venv/bin/python3 /a0/usr/workdir/telegram_bridge.py
environment=
user=root
directory=/a0
stopwaitsecs=10
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
stderr_logfile=/dev/stderr
stderr_logfile_maxbytes=0
autorestart=true
startretries=3
stopasgroup=true
killasgroup=true
```

---

> **⚠️ Important:** Whichever option you chose, make sure the block only appears **once** in the file. A duplicate `[program:telegram_bridge]` block will cause supervisor to error. You can verify the file looks correct with:
> ```bash
> docker exec agent-zero cat /etc/supervisor/conf.d/supervisord.conf
> ```


### 9. Reload and start

```bash
docker exec agent-zero supervisorctl reread && docker exec agent-zero supervisorctl update
```

### 10. Verify (optional)

```bash
docker exec agent-zero supervisorctl status telegram_bridge
```

From now on the bot will auto-start with the container.

---

## Useful Commands

```bash
# Stop the bot
docker exec agent-zero supervisorctl stop telegram_bridge

# Start the bot
docker exec agent-zero supervisorctl start telegram_bridge

# Restart the bot
docker exec agent-zero supervisorctl restart telegram_bridge

# Status of all services
docker exec agent-zero supervisorctl status

# Kill the bot (any running instance)
docker exec agent-zero pkill -f telegram_bridge.py

# Instant kill
docker exec agent-zero pkill -9 -f telegram_bridge.py

# Verify it's gone
docker exec agent-zero pgrep -f telegram_bridge.py

# Check if the process is alive
docker exec agent-zero ps aux | grep telegram_bridge

# View live logs (supervisord-managed)
docker logs agent-zero --tail 50
```

## Restrict by User ID (optional)

You can limit the bot to specific Telegram users so that only you (or a set of people) can interact with it.

**1. Get your user ID** — message [@userinfobot](https://t.me/userinfobot) on Telegram. It will reply with your numeric user ID.

**2. Add it to `.env`** — open `/a0/usr/.env` and add:

```
TELEGRAM_USER_IDS=123456789
```

For multiple users, separate with commas: `TELEGRAM_USER_IDS=123456789,987654321`

**3. Restart the bot:**

```bash
docker exec agent-zero supervisorctl restart telegram_bridge
```

When the bot starts, the banner will show the allowed user IDs. If `TELEGRAM_USER_IDS` is not set or empty, all users are allowed (the default). Unauthorized users are silently ignored — they receive no response.

> This filter works alongside the existing `TELEGRAM_CHAT_IDS` filter. Both can be used at the same time.

---

## Telegram Bot Commands

| Command | Description |
|---------|-------------|
| `/start` | Welcome message and usage info |
| `/reset` | Start a new conversation (clears context) |
| `/status` | Show connection status |
| `/help` | Show available commands |

---

## Cost

The idle bot is effectively costless:

- **No LLM credits** — the bot only calls Agent Zero's `/api_message` when a Telegram message arrives. No messages = no API calls = no tokens used.
- **No cron jobs** — the bot script has no scheduled tasks or periodic pings. It uses long polling to wait for Telegram updates.
- **Negligible resources** — the idle process uses ~30–50MB of RAM and essentially 0% CPU.

You only "pay" (in LLM credits) when someone actually sends a message through Telegram.

---

## Uninstall / Cleanup

To fully remove the bot from your Agent Zero container (no container restart needed):

```bash
# 1. Stop and remove the supervisord entry (if configured)
docker exec agent-zero supervisorctl stop telegram_bridge

# 2. Remove the supervisord config block
#    Open the file and delete the [program:telegram_bridge] block:
docker exec agent-zero nano /etc/supervisor/conf.d/supervisord.conf
#    Alternatively, you can do this via the Files browser in the Agent Zero GUI.
#    Then reload:
docker exec agent-zero supervisorctl reread && docker exec agent-zero supervisorctl update

# 3. Kill any running instance
docker exec agent-zero pkill -9 -f telegram_bridge.py

# 4. Delete the bot script and log file
docker exec agent-zero rm -f /a0/usr/workdir/telegram_bridge.py /a0/usr/workdir/telegram_bridge.log

# 5. Remove the TELEGRAM_BOT_TOKEN line from /a0/usr/.env
docker exec agent-zero sed -i '/TELEGRAM_BOT_TOKEN/d' /a0/usr/.env
```

Optionally, you can also delete the bot via [@BotFather](https://t.me/BotFather) on Telegram (send `/deletebot`).

---

## See Also

- [Agent Zero Discord Bridge](https://github.com/winboost/agent-zero-discord-bridge) — Same concept for Discord

---

## License

MIT
