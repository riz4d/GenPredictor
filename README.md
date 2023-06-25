# GenderPredictor
This is a Python script Using OpenCV and pre-trained models to detect faces in an image and predict the gender of each detected face.

### Requirements
» Python 3 <br>
» OpenCV (pip install opencv-python)<br>
» NumPy (pip install numpy)
 
### Usage

Open your favorite Terminal and run these commands.

##### 1. Clone This Repository in to Local Machine:

```sh
gitclone https://github.com/riz4d/GenderPredictor/
```

##### Install Requirements:

```sh
pip install -r requirements.txt
```

##### Run the script :

```sh
python3 main.py
```

### Note

» The script uses a confidence threshold (conf_threshold) to filter out weak face detections. You can adjust this value to control the sensitivity of the detection. <br>
» If no faces are detected in the image, the script will display a message indicating that no faces were Detected.

### Acknowledgments

the pre-trained models are taken from the [opencv](https://github.com/opencv) repositories further reference take look on their repositories for the models and their usage.
