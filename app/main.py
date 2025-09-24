from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from textblob import TextBlob

app = FastAPI(title="Cognitive Twin API")

# ---------- Request Models ----------
class TextInput(BaseModel):
    text: str

# ---------- Endpoints ----------

@app.get("/")
def root():
    return {"message": "🚀 Cognitive Twin API is running!"}

@app.get("/hello/{name}")
def hello(name: str):
    return {"message": f"Hello {name}, welcome to your Cognitive Twin!"}

@app.post("/analyze")
def analyze_sentiment(data: TextInput):
    blob = TextBlob(data.text)
    sentiment = blob.sentiment.polarity
    if sentiment > 0:
        label = "positive"
    elif sentiment < 0:
        label = "negative"
    else:
        label = "neutral"
    return {"original": data.text, "sentiment": label}

@app.post("/summarize")
def summarize(data: TextInput):
    # simple summary (first 30 words)
    words = data.text.split()
    if len(words) > 30:
        summary = " ".join(words[:30]) + "..."
    else:
        summary = data.text
    return {"original": data.text, "summary": summary}

@app.post("/translate")
def translate(data: TextInput, lang: str = "es"):
    # Translate to another language (default Spanish)
    try:
        blob = TextBlob(data.text)
        translated = str(blob.translate(to=lang))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"original": data.text, "translated": translated, "lang": lang}

@app.post("/classify")
def classify(data: TextInput):
    text = data.text.lower()
    if "ai" in text or "machine learning" in text:
        label = "Technology"
    elif "health" in text or "medicine" in text:
        label = "Health"
    elif "finance" in text or "money" in text:
        label = "Finance"
    else:
        label = "General"
    return {"original": data.text, "category": label}

# ---------- NEW: Upload + Summarize ----------
@app.post("/upload-summarize")
async def upload_summarize(file: UploadFile = File(...)):
    """
    Upload a .txt file and get a summary of its contents.
    """
    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are supported.")

    contents = await file.read()
    text = contents.decode("utf-8")

    # same summary logic as above
    words = text.split()
    if len(words) > 30:
        summary = " ".join(words[:30]) + "..."
    else:
        summary = text

    return {
        "filename": file.filename,
        "summary": summary,
        "original_length": len(text.split())
    }
