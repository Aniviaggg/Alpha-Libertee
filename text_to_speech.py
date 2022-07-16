from gtts import gTTS
import os

path = "/test/TTS_test.txt"
with open(os.path.join(os.path.dirname(__file__), path), 'r') as input_file:
    myText = input_file.read()

language = 'en'

output = gTTS(text=myText, lang = language, slow = False)

output.save("output.mp3")

os.system("start output.mp3")
