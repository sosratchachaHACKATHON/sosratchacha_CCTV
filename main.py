import cv2
import numpy as np
import time  # -- 프레임 계산을 위해 사용
import math


def detectAndDisplay(frame, SUM_human_dog_distance, human_dog_sameFrame_cnt):
    AVG_human_dog_distance = 0.0

    min_confidence = 0.5

    start_time = time.time()
    img = cv2.resize(frame, None, fx=0.8, fy=0.8)
    height, width, channels = img.shape
    # cv2.imshow("Original Image", img)

    # -- 창 크기 설정
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # -- 탐지한 객체의 클래스 예측
    class_ids = []
    confidences = []
    boxes = []

    # -- 인간 중심 좌표
    human_x = 0.0
    human_y = 0.0
    # -- 강아지 중심 좌표
    dog_x = 0.0
    dog_y = 0.0
    # -- 인간과 강아지 평균 중심 거리 차이

    human_dog_distance = 0.0

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > min_confidence:
                # 탐지한 객체 박싱
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                if class_id == 0:
                    human_x = center_x
                    human_y = center_y
                if class_id == 16:
                    dog_x = center_x
                    dog_y = center_y

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, 0.4)
    font = cv2.FONT_HERSHEY_DUPLEX

    dog_exist = False
    human_exist = False
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = "{}: {:.2f}".format(classes[class_ids[i]], confidences[i] * 100)

            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 5), font, 1, color, 1)

            if classes[class_ids[i]] == 'person':
                print(f"HUMAN Center: {human_x}, {human_y}")
                human_exist = True
            if classes[class_ids[i]] == 'dog':
                print(f"DOG Center: {dog_x}, {dog_y}")
                dog_exist = True

    if human_exist == True and dog_exist == True:
        human_dog_distance = math.sqrt((human_x - dog_x)**2 + (human_y - dog_y)**2)
        SUM_human_dog_distance += human_dog_distance
        human_dog_sameFrame_cnt += 1

        try:
            AVG_human_dog_distance = SUM_human_dog_distance / human_dog_sameFrame_cnt
        except ZeroDivisionError:
            print(ZeroDivisionError)
        print(f"Human AND Dog BOTH exitst !!!!! AVG DISTANCE: {AVG_human_dog_distance} NOW DISTANCE: {human_dog_distance} HUMANXDOG_cnt: {human_dog_sameFrame_cnt}")

        if abs(AVG_human_dog_distance - human_dog_distance) >= 100:
            print("|------------------------------|")
            print("|                              |")
            print("|      DOG LOST!!!!!!          |")
            print("|                              |")
            print("|------------------------------|")
            cv2.imwrite('screenshot.png', frame)

    end_time = time.time()
    process_time = end_time - start_time
    print("=== A frame took {:.3f} seconds".format(process_time))
    cv2.imshow("YOLO test", img)

    return (SUM_human_dog_distance, human_dog_sameFrame_cnt)


# -- yolo 포맷 및 클래스명 불러오기
model_file = './config/yolov3.weights'
config_file = './config/yolov3.cfg'
net = cv2.dnn.readNet(model_file, config_file)

# -- GPU 사용
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# -- 클래스(names파일) 오픈
classes = []
with open("./config/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
# print(net.getUnconnectedOutLayers()) #[200 227 254]
output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# -- 비디오 활성화
if __name__=="__main__":

    cap = cv2.VideoCapture(0)  # -- 웹캠 사용시 vedio_path를 0 으로 변경
    if not cap.isOpened:
        print('--(!)Error opening video capture')
        exit(0)
    SUM_human_dog_distance = 0.0
    human_dog_sameFrame_cnt = 0
    while True:
        ret, frame = cap.read()
        if frame is None:
            print('--(!) No captured frame -- Break!')
            break
        (SUM_human_dog_distance, human_dog_sameFrame_cnt) = detectAndDisplay(frame, SUM_human_dog_distance, human_dog_sameFrame_cnt)
        # -- q 입력시 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()