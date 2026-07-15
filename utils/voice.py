import tempfile
from gtts import gTTS
def text_to_speech(text):
    temp=tempfile.NamedTemporaryFile(delete=False,suffix=".mp3"); temp.close()
    gTTS(text=text[:5000],lang="en").save(temp.name); return temp.name
