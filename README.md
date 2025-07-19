
This is a hand gesture recognition project using video data and machine learning.

To run the project:

1. You have to add your own gesture videos to the `train_videos` folder.
2. Run `generating_data.py` to convert videos into image data for training.
3. Run `training_model.py` to train the model using the generated data.
4. (Optional) Run `sample.py` to test preprocessing or hand tracking features.
5. Run `testing.py` to start real-time hand gesture prediction using your webcam.

Make sure all required libraries are installed before running the code.
>>conda create --name ai_env python=3.8.8 (creating environment)
>>conda activate ai_env (<----Be careful before installing requirements---->)
>>pip install tensorflow
>>pip install opencv-python
>>pip install mediapipe

[Run sample.py]
1. open hand_gestures folder path in vs code
2. conda activate ai_env
3. python sample.py
