import cv2
import numpy as np
from keras.models import load_model
from statistics import mode
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input
from multiprocessing import Process
import time
import speech_recognition as sr
import openai
from gtts import gTTS
from playsound import playsound

def TTS(text, path='.//test.mp3'):
    try:
        tts_ko = gTTS(text, lang='ko')
        tts_ko.save(path)
        playsound(path)
    except:
        print('tts error')
    

recognizer = sr.Recognizer()

OPENAI_API_KEY = "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"

# openai API 키 인증
openai.api_key = OPENAI_API_KEY

# 모델 - GPT 3.5 Turbo 선택
model = "gpt-3.5-turbo"

# 메시지 설정하기
messages = [
        {"role": "system", "content": "You are a Counselor."}
] 

expresssion = {'happy':['오늘 기분이 좋아 보이시네요 좋은일 이라도 있으셨나요?',
                        '오늘 무슨 좋은 일이 있으신가요?',
                        '표정이 좋아 보이시네요 좋은 일이 있으신가요?'],
                'sad' : ['오늘 기분이 안좋아 보이시네요 안좋은 일이라도 있으셨나요?',
                         '오늘 무슨 안좋은 일이 있으셨나요?',
                         '표정이 좋지 않아 보이네요 무슨 일이 있으셨나요?'],
                'angry' : ['많이 화나 보이시네요 누구 때문에 그렇게 화나 계시죠?',
                           '오늘 많이 화나 계신것 같네요 무슨 일이시죠?',
                           '표정이 많이 화나 보이시는데 무슨 일이시죠?']}



# parameters for loading data and images
emotion_model_path = './models/emotion_model.hdf5'
emotion_labels = get_labels('fer2013')

# hyper-parameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)

# loading models
face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
emotion_classifier = load_model(emotion_model_path)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# starting lists for calculating modes
emotion_window = []
# starting video streaming

cap = cv2.VideoCapture(0)


def cam_emotion():
    while cap.isOpened(): # True:
        ret, bgr_image = cap.read()

        #bgr_image = video_capture.read()[1]

        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5,
                minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

        for face_coordinates in faces:

            x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
            gray_face = gray_image[y1:y2, x1:x2]
            try:
                gray_face = cv2.resize(gray_face, (emotion_target_size))
            except:
                continue

            gray_face = preprocess_input(gray_face, True)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)
            emotion_prediction = emotion_classifier.predict(gray_face,  verbose = 0)
            emotion_probability = np.max(emotion_prediction)
            emotion_label_arg = np.argmax(emotion_prediction)
            emotion_text = emotion_labels[emotion_label_arg]
            emotion_window.append(emotion_text)

            if len(emotion_window) > frame_window:
                emotion_window.pop(0)
            try:
                emotion_mode = mode(emotion_window)
            except:
                continue

            if emotion_text == 'angry':
                color = emotion_probability * np.asarray((255, 0, 0))
            elif emotion_text == 'sad':
                color = emotion_probability * np.asarray((0, 0, 255))
            elif emotion_text == 'happy':
                color = emotion_probability * np.asarray((255, 255, 0))
            elif emotion_text == 'surprise':
                color = emotion_probability * np.asarray((0, 255, 255))
            else:
                color = emotion_probability * np.asarray((0, 255, 0))

            color = color.astype(int)
            color = color.tolist()

            draw_bounding_box(face_coordinates, rgb_image, color)
            draw_text(face_coordinates, rgb_image, emotion_mode,
                    color, 0, -45, 1, 1)

        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        cv2.imshow('window_frame', cv2.flip(bgr_image, 1))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def gpt_anwser():
    while True:
        query = ''
        with sr.Microphone() as source:
            print("음성을 입력하세요...")
            audio = recognizer.listen(source)
        try:
            # 음성을 텍스트로 변환
            text = recognizer.recognize_google(audio, language="ko-KR")
            query = text
        except sr.UnknownValueError:
            print("음성을 인식하지 못했습니다.")
        except sr.RequestError as e:
            print(f"오류 발생: {e}")

        if (query == "상담해 줘서 고마워") or (cv2.waitKey(1) & 0xFF == ord('q')):
            break
        
        messages.append({"role": "user", "content": query})
        # ChatGPT API 호출하기
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages
        )
        
        answer = response['choices'][0]['message']['content']
        print(answer)
        TTS(answer)


if __name__ == '__main__':
    cam_p = Process(target=cam_emotion)
    mike_p = Process(target=gpt_anwser)
    cam_p.start()
    mike_p.start()
    cam_p.join()