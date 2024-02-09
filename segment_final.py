import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import numpy as np
import torch
from moviepy.editor import VideoFileClip
import threading



if torch.cuda.is_available():
    # CUDA 장치 정보 출력
    print("CUDA 사용 가능")
    print("GPU 장치 수:", torch.cuda.device_count())
    print("현재 사용 중인 GPU:", torch.cuda.current_device())
else:
    print("CUDA를 사용할 수 없습니다.")


# yolov8s-seg.pt 모델 로드
model = YOLO('yolov8s-seg.pt')

if torch.cuda.is_available():
    model.cuda()


# 웹캠으로부터 실시간 영상 캡처
webcam = cv2.VideoCapture(0)

# 입력받은 동영상 읽기
video = cv2.VideoCapture('long.mp4')


video_file = 'long.mp4'
audio_file = 'long_audio.wav'
video_clip = VideoFileClip(video_file)


# 웹캠과 동영상의 프레임 크기와 프레임 레이트 설정
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
#webcam.set(cv2.CAP_PROP_FPS, 30)

video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
#video.set(cv2.CAP_PROP_FPS, 30)
#writer = cv2.VideoWriter('output.mp4',cv2.VideoWriter_fourcc(*'mp4v'),30,(1280,720))

# 마스크 외곽선을 채우는 함수 정의
def fill_contour(frame, xyn):
    # 빈 이미지 생성
    contour = np.zeros_like(frame)

    # 정규화된 좌표를 픽셀 좌표로 변환
    xy = [(int(x * frame.shape[1]), int(y * frame.shape[0])) for x, y in xyn]

    # 마스크 외곽선을 흰색으로 채우기
    cv2.fillPoly(contour, [np.array(xy)], (255, 255, 255))

    # 채운 이미지 반환
    return contour


def play_audio():
    video_clip.audio.preview()

# 음성 재생을 위한 쓰레드 생성
audio_thread = threading.Thread(target=play_audio)
audio_thread.daemon = True
audio_thread.start()

# 합성한 영상을 실시간으로 재생하고 저장하는 반복문
while True:
    frames = []
    for _ in range(3):
        ret2, frame2 = video.read()
        if not ret2:
            print('Cam Error')
            break
        frames.append(frame2)
    ret1, frame1 = webcam.read()
    frame2 = frames[-1]
    if not ret1:
        print('Cam Error')
        break

    results = model.predict(frame1)
    masks = results[0].masks
  
    xyn = masks.xyn
    sizes = []
    for xy in xyn:
        xy = [(x*frame1.shape[1], y*frame1.shape[0]) for x, y in xy]
        xs = [x for x, _ in xy]
        ys = [y for _ ,y in xy]
        
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        w = x_max - x_min
        h = y_max - y_min
        size = w*h
        sizes.append(size)
    sizes = np.array(sizes)
    max_idx = np.argmax(sizes)


    # 각 객체의 마스크 외곽선을 채운 이미지를 생성합니다.
    contour = np.zeros_like(frame1)
    xyn = masks.xyn[max_idx]
    contour += fill_contour(frame1, xyn)
    # 마스크 외곽선을 8비트 단일 채널 이미지로 변환합니다.
    contour = cv2.cvtColor(contour, cv2.COLOR_BGR2GRAY)
    
    
    ncontour = np.zeros_like(frame1)
    ncontour += cv2.bitwise_not(fill_contour(frame1, xyn))
    ncontour = cv2.cvtColor(ncontour, cv2.COLOR_BGR2GRAY)




    # 웹캠 프레임과 동영상 프레임을 마스크 외곽선으로 합성합니다.
    segment = cv2.bitwise_and(frame1, frame1, mask=contour)
    frame2 = cv2.bitwise_or(frame2, frame2, mask=ncontour)
    new = cv2.bitwise_or(segment, frame2)

    # 합성한 프레임을 재생합니다.
    cv2.imshow('Blended', new)
    #writer.write(new)

    if cv2.waitKey(1) == ord('q'):
        break


# 웹캠, 동영상, 비디오 라이터, 창을 닫습니다.
webcam.release()
video.release()
#writer.release()
cv2.destroyAllWindows()
video_clip.close()