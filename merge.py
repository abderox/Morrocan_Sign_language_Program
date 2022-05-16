import argparse
import cv2

import pyttsx3
from predict import *


from yolo import YOLO

engine = pyttsx3.init()
engine.setProperty('rate', 105)
engine.setProperty('voice', 1)
sentence = ""
ap = argparse.ArgumentParser()

ap.add_argument('-n', '--network', default="normal", choices=["normal", "tiny", "prn", "v4-tiny"],
                help='Network Type')
ap.add_argument('-d', '--device', type=int, default=0, help='Device to use')
ap.add_argument('-s', '--size', default=416, help='Size for yolo')
ap.add_argument('-c', '--confidence', default=0.2, help='Confidence for yolo')
ap.add_argument('-nh', '--hands', default=-1, help='Total number of hands to be detected per frame (-1 for all)')
args = ap.parse_args()

if args.network == "normal":
    print("loading yolo...")
    yolo = YOLO("models/cross-hands.cfg", "models/cross-hands.weights", ["hand"])
elif args.network == "prn":
    print("loading yolo-tiny-prn...")
    yolo = YOLO("models/cross-hands-tiny-prn.cfg", "models/cross-hands-tiny-prn.weights", ["hand"])
elif args.network == "v4-tiny":
    print("loading yolov4-tiny-prn...")
    yolo = YOLO("models/cross-hands-yolov4-tiny.cfg", "models/cross-hands-yolov4-tiny.weights", ["hand"])
else:
    print("loading yolo-tiny...")
    yolo = YOLO("models/cross-hands-tiny.cfg", "models/cross-hands-tiny.weights", ["hand"])

yolo.size = int(args.size)
yolo.confidence = float(args.confidence)

print("starting webcam...")
cv2.namedWindow("preview")
vc = cv2.VideoCapture(args.device)

if vc.isOpened():  # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    width, height, inference_time, results = yolo.inference(frame)

    # display fps
    cv2.putText(frame, f'{round(1/inference_time,2)} FPS', (15,15), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,255,255), 2)

    # sort by confidence
    results.sort(key=lambda x: x[2])

    # how many hands should be shown
    hand_count = len(results)
    if args.hands != -1:
        hand_count = int(args.hands)

    # display hands
    for detection in results[:hand_count]:
        id, name, confidence, x, y, w, h = detection
        cx = x + (w / 2)
        cy = y + (h / 2)

        # draw a bounding box rectangle and label on the image
        if confidence > 0.45 :
            color = (0, 255, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "%s (%s)" % (name, round(confidence, 2))
            # cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
            #             0.5, color, 2)
            img1 = frame[y: y + h, x: x + w]
            img_ycrcb = cv2.cvtColor(img1, cv2.COLOR_BGR2YCR_CB)
            blur = cv2.GaussianBlur(img_ycrcb, (11, 11), 0)

            # lower  and upper skin color
            skin_ycrcb_min = np.array((0, 138, 67))
            skin_ycrcb_max = np.array((255, 173, 133))

            mask = cv2.inRange(blur, skin_ycrcb_min, skin_ycrcb_max)  # detecting the hand in the bounding box

            kernel = np.ones((2, 2), dtype=np.uint8)

            # Fixes holes in foreground
            mask = cv2.dilate(mask, kernel, iterations=1)

            naya = cv2.bitwise_and(img1, img1, mask=mask)
            
            hand_bg_rm = naya
            hand = img1

            # Control Key
            c = cv2.waitKey(1) & 0xff

            # Speak the sentence
            if len(sentence) > 0 and c == ord('s'):
                engine.say(sentence)
                engine.runAndWait()
            # Clear the sentence
            if c == ord('c') or c == ord('C'):
                sentence = ""
            # Delete the last character
            if c == ord('d') or c == ord('D'):
                sentence = sentence[:-1]

            # Put Space between words
            if c == ord('m') or c == ord('M'):
                sentence += " "

            # If  valid hand area is cropped
            if hand.shape[0] != 0 and hand.shape[1] != 0:
                conf, label = which(hand_bg_rm)
                if conf >= THRESHOLD:
                    cv2.putText(frame, label, (x, y - 5),cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)
                if c == ord('n') or c == ord('N'):
                    sentence += label
            
            cv2.putText(frame, sentence, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)
           

    cv2.imshow("preview", frame)

    rval, frame = vc.read()

    key = cv2.waitKey(20)
    if key == 27:  
        break

cv2.destroyWindow("preview")
vc.release()




