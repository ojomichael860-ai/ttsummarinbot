import os
import asyncio
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

# --- Sumy Natural Language Processing Libraries ---
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# --- Web Server for Render Health Checks ---
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(b"Text Summarizer Engine is Live!")

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()

def run_health_server():
    port = int(os.environ.get("PORT", 10000))
    server = HTTPServer(("0.0.0.0", port), HealthCheckHandler)
    print(f"Health check server running on port {port}")
    server.serve_forever()

# --- Summarization Core Logic ---
def summarize_text(input_text: str, sentences_count: int = 3) -> str:
    """Extracts the most important sentences from a body of text using LSA."""
    # Parse text structure using standard English tokenizer rule sets
    parser = PlaintextParser.from_string(input_text, Tokenizer("english"))
    stemmer = Stemmer("english")
    
    summarizer = LsaSummarizer(stemmer)
    summarizer.stop_words = get_stop_words("english")
    
    # Calculate and capture structural rankings
    summary_sentences = summarizer(parser.document, sentences_count)
    
    # Reassemble strings
    return " ".join([str(sentence) for sentence in summary_sentences])

# --- Bot Commands and Event Handlers ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📝 **Welcome to the AI Text Summarizer Bot!**\n\n"
        "Send me any long article, document text, or essay paragraph, "
        "and I will instantly analyze the contents and extract the key takeaways "
        "into a short, powerful summary for you! ⚡\n\n"
        "👉 *Simply paste your long text below to begin.*",
        parse_mode="Markdown"
    )

async def handle_text_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text

    # Guard clause against short single lines
    if len(user_text.strip()) < 200:
        await update.message.reply_text(
            "⚠️ *The text you provided is a bit too short to summarize.* "
            "Please send an article or paragraph containing at least 200–300 characters.",
            parse_mode="Markdown"
        )
        return

    # Trigger processing state visual indicator
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    status_msg = await update.message.reply_text("⏳ *Analyzing context structure and generating summary...*", parse_mode="Markdown")

    try:
        # Determine reading length adaptation
        # For extremely long text blocks, scale the summary size to 4 sentences instead of 3
        target_sentences = 4 if len(user_text) > 1500 else 3
        
        summary_result = summarize_text(user_text, sentences_count=target_sentences)
        
        if not summary_result.strip():
            summary_result = "Could not cleanly isolate focal sentences. Please verify structural paragraph inputs."

        # Send back the clean text summary results
        await update.message.reply_text(
            f"📋 **Content Summary:**\n\n"
            f"_{summary_result}_\n\n"
            f"📉 *Shortened down to {target_sentences} key sentences.*",
            parse_mode="Markdown"
        )
    except Exception as e:
        print(f"Summarizer Exception Trace: {e}")
        await update.message.reply_text("❌ An error occurred while parsing your document text profile.")
    finally:
        await status_msg.delete()

async def main():
    TOKEN = os.environ.get("TELEGRAM_TOKEN")
    if not TOKEN:
        raise ValueError("Missing TELEGRAM_TOKEN environment target variable.")

    # Run the background port listener for Render health verification routines
    threading.Thread(target=run_health_server, daemon=True).start()

    # Build the Application framework mapping rules
    app = Application.builder().token(TOKEN).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_input))
    
    print("Text Summarization engine polling sequence active...")
    
    async with app:
        await app.initialize()
        await app.start()
        await app.updater.start_polling()
        while True:
            await asyncio.sleep(3600)

if __name__ == "__main__":
    asyncio.run(main())
