"""
08 02 2023 09:17h CET
Data collection for 5 actions of LSE using mediapipe library
LSE (Spanish Sign Language)
Source:
https://www.youtube.com/watch?v=9MTiQMxTXPE

Data collection of 30 videos per action
each one split in 30 arrays (1 array per frame) and saved in
the corresponding folder annotated by a number.
"""

import cv2
import time
import datetime
import numpy as np
import os

cap= cv2.VideoCapture(1, cv2.CAP_DSHOW)

width= int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# path = C:\Users\maalvear\PycharmProjects\Creacion_BaseDatos_LSE\PRUEBA
path = os.path.join('C:/Users/maalvear/PycharmProjects/vowels_lse_gesture_recognition/VIDEOS_LSE_PRUEBA')

# Actions that we try to detect
actions = np.array(['Hola', 'Gracias', 'Buenos_dias', 'Buenas_tardes', 'Buenas_noches'])

# Thirty videos worth of data
no_sequences = 30

# Videos are going to be 30 frames in length
sequence_length = 30


for action in actions:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(path, action, str(sequence)))
        except:
            pass

# writer= cv2.VideoWriter(path + 'basicvideo1.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (width,height))
writer= cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))


for action in actions:
    # Loop through sequences aka videos
    for sequence in range(no_sequences):
        # Loop through video length aka sequence length
        for frame_num in range(sequence_length):

            # Read feed
            ret, frame = cap.read()
            # frame = cv2.flip(frame, 0) to flip the frame
            date_string = datetime.datetime.now().strftime("%Y-%m-%d  %I.%M.%S%p   %A")
            cv2.imshow('frame', frame)

            # NEW Apply wait logic
            if frame_num == 0:
                cv2.putText(frame, 'STARTING COLLECTION', (120, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                cv2.putText(frame, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15, 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                # Show to screen
                cv2.imshow('OpenCV Feed', frame)
                cv2.waitKey(2000)
            else:
                cv2.putText(frame, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15, 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                # Show to screen
                cv2.imshow('OpenCV Feed', frame)

                # NEW Export keypoints
                #keypoints = extract_keypoints(results) # Saving videos as numpy arrays
                npy_path = os.path.join(path, action, str(sequence), str(frame_num))
                writer = cv2.VideoWriter(npy_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
                writer.write(frame)


                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()



cap.release()
cv2.destroyAllWindows()
