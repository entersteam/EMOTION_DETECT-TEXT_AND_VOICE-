import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from konlpy.tag import Okt
import time
import speech_recognition as sr
import openai
from gtts import gTTS
from playsound import playsound
from ultralytics import YOLO
import math
import pickle

recognizer = sr.Recognizer()
okt = Okt()

OPENAI_API_KEY = "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"

openai.api_key = OPENAI_API_KEY

model = "gpt-3.5-turbo"
model_YOLO = YOLO("//Users//minjae//Desktop//col//EMOTION_DETECT-TEXT_AND_VOICE--main//weights//best.pt") #가중치 파일 경로

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
    
#모델 불러오기
model = tf.keras.Sequential([
  tf.keras.layers.Embedding(10000, 300, input_length=45), 
  tf.keras.layers.LSTM(units=64, return_sequences=True), 
  tf.keras.layers.LSTM(units=64), 
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(4, activation='softmax')
])

model.load_weights('./lstm/my_checkpoint')

cap = cv2.VideoCapture(0)

classNames = ['angry','sad','neatural','happy']


def TTS(text, path='.//output.mp3'):
    try:
        tts_ko = gTTS(text, lang='ko')
        tts_ko.save(path)
        playsound(path)
    except:
        print('tts error')

# 이전 대화 히스토리를 저장할 변수
conversation_history = [{"role": "system", "content": "너는 상담사야. 너는 앞으로 말할때 사람처럼 말해야 해. 대답을 짧게 해야해. 인사로 시작해줘"}]

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
def text_to_emotion(text:str):
    seq = [okt.morphs(text)]
    seq = tokenizer.texts_to_sequences(seq)
    seq = pad_sequences(seq, padding='post', maxlen=45)
    return classNames[np.argmax(model.predict(seq))]

def emotion_Video():
    
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    emotion = ""
    start_t = time.time() # 같은 감정이 반복되서 감지되는 횟수 t 가 5가 되면 영상 감정인식 끝
    
    while True:
        success, img = cap.read()
        results = model_YOLO(img, stream=True)

        detected_objects = [] 

        for r in results:
            boxes = r.boxes

            for box in boxes:
                print(box)
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                confidence = math.ceil((box.conf[0] * 100)) /  100
                cls = int(box.cls[0])

                detected_objects.append(classNames[cls])  

                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)
                
        if emotion != classNames[cls]:
            emotion = classNames[cls]
            start_t = time.time()
        elif emotion == classNames[cls] and emotion != 'neatural':
            if time.time() - start_t > 5:
                cap.release()
                cv2.destroyAllWindows()
                return emotion
                
        cv2.imshow('Webcam', img)
        if cv2.waitKey(1) == ord('q'):
            break
    

    

if __name__=='__main__':
    while True:
        
        result_emotion = emotion_Video()
        
        query = ''
        
        while True:
            with sr.Microphone() as source:
                print("음성을 입력하세요...")
                recognizer.adjust_for_ambient_noise(source)     
                audio = recognizer.listen(source)
            try:
                # 음성을 텍스트로 변환
                query = recognizer.recognize_google(audio, language="ko-KR")
                
                with open("input.wav", "wb") as f:
                    f.write (audio.get_wav_data())
                break
            except sr.UnknownValueError:
                print("음성을 인식하지 못했습니다.")
            except sr.RequestError as e:
                print(f"오류 발생: {e}")
                break

            if (query == "상담해 줘서 고마워") or (cv2.waitKey(1) & 0xFF == ord('q')):
                break
            
                
        answer = chat_with_gpt(query + f"(기분 : {result_emotion})") 
        print(conversation_history)
        TTS(answer) 
        
         