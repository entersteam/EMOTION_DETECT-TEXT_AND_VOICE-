from gtts import gTTS
from playsound import playsound

def TTS(text, path='test.mp3'):
    tts_ko = gTTS(text, lang='ko')
    tts_ko.save(path)
    
    playsound(path)
    
if __name__=='__main__':
    TTS('안녕')
    print(1)