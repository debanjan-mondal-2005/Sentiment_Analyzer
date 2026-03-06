import re
import emoji

slang_dict = {
    "tbh": "to be honest",
    "ngl": "not gonna lie",
    "idk": "i do not know",
    "lol": "laughing",
    "omg": "oh my god",
    "bro": "brother",
    "highkey": "really"
}

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    for slang, full in slang_dict.items():
        text = re.sub(r"\b" + slang + r"\b", full, text)
    text = emoji.demojize(text)
    text = re.sub(r"[^a-zA-Z\s:]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text