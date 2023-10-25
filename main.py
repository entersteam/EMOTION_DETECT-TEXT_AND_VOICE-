import cv2
import numpy as np
from keras.models import load_model
from statistics import mode
from multiprocessing import Process
import time
import speech_recognition as sr
import openai
from gtts import gTTS
from playsound import playsound


recognizer = sr.Recognizer()
OPENAI_API_KEY = "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"

openai.api_key = OPENAI_API_KEY

model = "gpt-3.5-turbo"



cap = cv2.VideoCapture(0)

def TTS(text, path='.//output.mp3'):
    try:
        tts_ko = gTTS(text, lang='ko')
        tts_ko.save(path)
        playsound(path)
    except:
        print('tts error')

# 이전 대화 히스토리를 저장할 변수
conversation_history = [{"role": "system", "content": "너는 상담사야. 너는 앞으로 말할때 사람처럼 말해야 해. 대답을 짧게 해야해."}]

# ChatGPT와 대화하기 위한 함수 정의
def chat_with_gpt(prompt):
    # 이전 대화 내용과 새로운 입력을 합치기
    conversation_history.append({"role": "user", "content": prompt})

    # ChatGPT와 대화하기 위해 API 호출
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=conversation_history
    )

    # 모델의 응답 가져오기
    message = response['choices'][0]['message']['content']
    
    # 대화 기록에 현재 대화 추가
    conversation_history.append({"role": "assistant", "content": message})
    
    # 모델의 응답 반환
    return message

# # ChatGPT와 상호 작용하기
def get_emotion():
    emot = None
    return emot


if __name__=='__main__':
    while True:
        query = ''
        with sr.Microphone() as source:
            print("음성을 입력하세요...")
            recognizer.adjust_for_ambient_noise(source)     
            audio = recognizer.listen(source)
        try:
            # 음성을 텍스트로 변환
            query = recognizer.recognize_google(audio, language="ko-KR")
            
            with open("input.wav", "wb") as f:
                f.write (audio.get_wav_data())
        except sr.UnknownValueError:
            print("음성을 인식하지 못했습니다.")
        except sr.RequestError as e:
            print(f"오류 발생: {e}")

        if (query == "상담해 줘서 고마워") or (cv2.waitKey(1) & 0xFF == ord('q')):
            break
        
                
        answer = chat_with_gpt(query)
        print(conversation_history)
        TTS(answer)    