import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import numpy as np

# yolov8s-seg.pt 모델 로드
model = YOLO('yolov8n-seg.pt')

# 모델의 클래스 이름 가져오기 
names = model.model.names

# 웹캠으로부터 실시간 영상 캡처
webcam = cv2.VideoCapture(0)

# 입력받은 동영상 읽기
video = cv2.VideoCapture('C:\\Users\\user\\anaconda3\\envs\\Yolov8\\banana.mp4')

# 웹캠과 동영상의 프레임 크기와 프레임 레이트 설정
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
webcam.set(cv2.CAP_PROP_FPS, 30)

video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
video.set(cv2.CAP_PROP_FPS, 30)

# 합성한 영상을 저장할 비디오 라이터 생성
writer = cv2.VideoWriter('output.mp4',cv2.VideoWriter_fourcc(*'mp4v'),30,(1280,720))

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

# 합성한 영상을 실시간으로 재생하고 저장하는 반복문
while True:
    # 웹캠에서 프레임을 읽어옵니다.
    ret1, frame1 = webcam.read()
    # 만약 읽어오는데 실패하면, 오류 메시지를 출력하고 프로그램을 종료합니다.
    if not ret1:
        print('Cam Error')
        break

    # model.predict 함수를 사용하여 웹캠 프레임에 대해 객체를 인식하고 분할합니다.
    results = model.predict(frame1)

    # Result 객체에서 masks 속성을 가져옵니다.
    masks = results[0].masks
    #print("---------------------------------------")
    #print(results[0])
    #print(masks)
    
    
    
    xyn = masks.xyn
    sizes = []
    for xy in xyn:
        xy = [(x*frame1.shape[1], y*frame1.shape[0]) for x, y in xy]
        xs = [x for x, _ in xy]
        ys = [y for _ ,y in xy]
        
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        #x_min, y_min = min(xy)
        #x_min, _ = min(xy)
        #_, y_min = min(xy, key=lambda x:x[1])
        #x_max, y_max = max(xy)
        w = x_max - x_min
        h = y_max - y_min
        size = w*h
        sizes.append(size)
    sizes = np.array(sizes)
    max_idx = np.argmax(sizes)

    # 비디오 파일에서 프레임을 읽어옵니다.
    ret2, frame2 = video.read()
    # 만약 읽어오는데 실패하면, 오류 메시지를 출력하고 프로그램을 종료합니다.
    if not ret2:
        print('Video Error')
        break

    # 각 객체의 마스크 외곽선을 채운 이미지를 생성합니다.
    contour = np.zeros_like(frame1)
    #for xyn in masks.xyn:
    xyn = masks.xyn[max_idx]
    contour += fill_contour(frame1, xyn)
    # 마스크 외곽선을 8비트 단일 채널 이미지로 변환합니다. # 수정된 부분
    contour = cv2.cvtColor(contour, cv2.COLOR_BGR2GRAY)
    #
    
    
    ncontour = np.zeros_like(frame1)
    ncontour += cv2.bitwise_not(fill_contour(frame1, xyn))
    ncontour = cv2.cvtColor(ncontour, cv2.COLOR_BGR2GRAY)




    # 웹캠 프레임과 동영상 프레임을 마스크 외곽선으로 합성합니다.
    segment = cv2.bitwise_and(frame1, frame1, mask=contour)
    frame2 = cv2.bitwise_or(frame2, frame2, mask=ncontour)
    new = cv2.bitwise_or(segment, frame2)

    # 합성한 프레임을 재생합니다.
    cv2.imshow('Blended', new)

    # 합성한 프레임을 저장합니다.
    writer.write(new)

    # q 키를 누르면 종료합니다.
    if cv2.waitKey(1) == ord('q'):
        break

# 웹캠, 동영상, 비디오 라이터, 창을 닫습니다.
webcam.release()
video.release()
writer.release()
cv2.destroyAllWindows()