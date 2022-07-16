from gtts import gTTS
import os

myText = open('TTS_test.txt', 'r').read()

language = 'en'

output = gTTS(text=myText, lang = language, slow = False)

output.save("output.mp3")

os.system("start output.mp3")
