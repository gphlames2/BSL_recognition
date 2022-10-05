import keras.models
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3 as audio

mp_holistic = mp.solutions.holistic #Holistic model
mp_drawing = mp.solutions.drawing_utils
threshold = 0.8

selected_actions = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L','M','N','O','P','Q','R',
                             'S','T','U','V','W','X','Y','Z','dot','space'])
st.set_page_config(
    page_title='BSL Detection App',
    layout='wide'
)

# model_lstm = keras.models.load_model('data_gloves_LSTM_pose.h5')
# model_lstm_nopose = keras.models.load_model('data_gloves_LSTM_nopose.h5')
model_gru = keras.models.load_model('data_gloves_GRU_pose.h5')
model_gru_nopose = keras.models.load_model('data_gloves_GRU_nopose.h5')
# model_conv1d = keras.models.load_model('data_gloves_CONV1D_pose.h5')
# model_conv1d_nopose = keras.models.load_model('data_gloves_CONV1D_nopose.h5')

def mediapipe_detection(image, model):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
    img.flags.writeable = False  # Image is no longer writeable
    res = model.process(image)  # Make prediction
    img.flags.writeable = True  # Image is now writeable
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR CONVERSION RGB 2 BGR
    return img, res

def draw_styled_landmarks(image, results,pose_choice):
    if pose_choice == 'Pose':
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                                  )
    else:
        pass
    # Draw left-hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))
    # Draw right-hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

def extract_datapoints(frame_results, pose_choice):
    # pose = []
    # for rs in frame_results.pose_landmarks.landmark:
    #     test = np.array([rs.x, rs.y, rs.z, rs.visibility])
    #     pose.append(test)
    ps = np.array([[rs.x, rs.y, rs.z, rs.visibility] for rs in frame_results.pose_landmarks.landmark]).flatten() if frame_results.pose_landmarks else np.zeros(132)
    lh = np.array([[rs.x, rs.y, rs.z] for rs in
            frame_results.left_hand_landmarks.landmark]).flatten() if frame_results.left_hand_landmarks else np.zeros(
            21 * 3)
    rh = np.array([[rs.x, rs.y, rs.z] for rs in
            frame_results.right_hand_landmarks.landmark]).flatten() if frame_results.right_hand_landmarks else np.zeros(
            21 * 3)
    if pose_choice == 'Pose':
        return np.concatenate([ps, lh, rh])
    else:
        return np.concatenate([lh, rh])


def detect(sequence, sentence, choice, pose):

    if ['LSTM','GRU','CONV1D'].__contains__(choice):
        if choice == 'LSTM' and pose == 'Pose':
            prediction = model_lstm.predict(np.expand_dims(sequence[:30], axis=0))[0]
        elif choice == 'LSTM' and pose == 'No Pose':
            prediction = model_lstm_nopose.predict(np.expand_dims(sequence[:30], axis=0))[0]
        elif choice == 'GRU' and pose == 'Pose':
            prediction = model_gru.predict(np.expand_dims(sequence[:30], axis=0))[0]
        elif choice == 'GRU' and pose == 'No Pose':
            prediction = model_gru_nopose.predict(np.expand_dims(sequence[:30], axis=0))[0]
        elif choice == 'CONV1D' and pose == 'Pose':
            prediction = model_conv1d.predict(np.expand_dims(sequence[:30], axis=0))[0]
        elif choice == 'CONV1D' and pose == 'No Pose':
            prediction = model_conv1d_nopose.predict(np.expand_dims(sequence[:30], axis=0))[0]
        else:
            pass
        if prediction[np.argmax(prediction)] >= threshold:
            if selected_actions[np.argmax(prediction)] == 'dot':
                sentence.append('.')
                # st.write(''.join(sentence))
            elif selected_actions[np.argmax(prediction)] == 'space':
                sentence.append(' ')
            else:
                sentence.append(selected_actions[np.argmax(prediction)])
                # st.write(''.join(sentence))
        else:
            pass
    else:
        pass



def main():
    cam = cv2.VideoCapture(1)
    count = 0
    frame_sequence = []
    sentence = []
    st.title('BSL Detection App')
    st.header('Build with streamlit, OpenCV and mediapipe')
    options = ['GRU']
    pose_options = ['Pose','No Pose']
    choices = st.sidebar.selectbox('Select Model', options)
    pose_choice = st.sidebar.selectbox('Select pose options', pose_options)
    placeholder = st.container()
    with placeholder:
        frame_window = st.image([])
        text_area = st.empty()
        #_ = st.text_area('Predicted selected_actions', textarea)
    with mp_holistic.Holistic(min_detection_confidence=0.6, min_tracking_confidence=0.6) as holistic:
        while cam.isOpened():
            _, frame = cam.read()
            image, result = mediapipe_detection(frame, holistic)
            if result.right_hand_landmarks is None and result.left_hand_landmarks is None:
                frame_sequence.clear()
                count = 0
                draw_styled_landmarks(image, result, pose_choice)
                frame_window.image(image)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                continue
            else:
                draw_styled_landmarks(image, result, pose_choice)
                key_points = extract_datapoints(result, pose_choice)
                frame_sequence.append(key_points)
                frame_window.image(image)
                count += 1
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            if count == 30:
                detect(frame_sequence, sentence, choices, pose_choice)
                text_area.text('')
                text = ''.join(sentence)
                text_area.subheader(text)
                if len(sentence) == 0:
                    pass
                else:
                    if sentence[len(sentence) - 1] == '.':
                        sentence.pop()
                        audio.speak(''.join(sentence))
                        sentence.clear()
                        text_area.text('')
                    else:
                        pass

                count = 0


main()
