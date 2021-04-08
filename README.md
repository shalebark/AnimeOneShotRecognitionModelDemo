# AnimeOneShotModel

## What is this?

This is a demo for one shot face encoding model for anime characters.

The model is meant to be used with dlib's python library. dlib[http://dlib.net/]

Using dlib's face encoding api, you can use this model to generate face encodings.

## Model Details

The model is trained using 2400 images pulled from various anime series. Only images that are near portrait were selected and used.

Testing data I used 1600 images pulled from different series. Only images that are near portrait were selected.

Results for comparing each class of images to every class of images (48 classes in total)

Accuracies: 0.945469

Precisions: 0.366326

Recalls: 0.969469

Specificities: 0.943694

F1 Score: 0.531731


Results for comparing each image to the best of each class (again 48 classses in total).

Accuracy: 94%


## Data Details

The training data comes from scanning entire episodes of anime series.

Faces from the series were extracted, aligned, so that the eyes are horizontal to each other, and zoomed in so the face is near centered.

See Aligner_Cleaned.py as an example.

Those faces were then selected with the criteria of being nearly portrait images (meaning the character was not facing heavly to the side),
and sorted into it's own class.


The test data follows the same methodology but on a different set of anime series.


There is another set of data that has each character sorted in it's own class, but does not adhere to portrait rules. That's labeled as "wild_data".


Link to Training Data: https://www.kaggle.com/andock/anime-face-from-video-frames-portrait-data
Link to Test Data: https://www.kaggle.com/andock/anime-face-from-video-frames-portrait-data
Link to Wild Data:

# Usage

Download the model. Link[https://github.com/shalebark/AnimeOneShotRecognitionModelDemo/raw/master/models/facial_portrait_only_3_29_21.dat]


```
encoder = dlib.face_recognition_model_v1('models/facial_portrait_only_3_29_21.dat')
threshold = 0.6 # a smaller number for more similar faces, a higher for more range of accepted faces

encode1 = np.array(encoder.compute_face_descriptor(extract_face(image1), 1))
encode2 = np.array(encoder.compute_face_descriptor(extract_face(image2), 1))
return np.linalg.norm([encode1] - encode2, axis=1)[0] <= threshold

```

# Installation

## Ubuntu

```
apt-get install -y build-essential cmake python3-opencv
pip install -r requirements.txt
```

## Demo

```
python demo.py {firstimage.jpg} {secondimage.jpg}
```

## Usage with Docker

If you have docker installed, you can run the image through docker.

### Build the docker image

```
docker build . -t aosdemo
```

### Docker example with your image

```
docker run -v {parent-directory-to-your-image}:/app aosdemo image1.jpg imag2.jpg
```

Running demo images

```
docker run aosdemo  /code/images/arima1.jpg /code/images/arima2.jpg
```

# Results

Image1             |  Image2             |      Result
:-------------------------:|:-------------------------:|:-------------------------:|
![](https://github.com/shalebark/AnimeOneShotRecognitionModelDemo/raw/master/images/arima1.jpg =300x)  |  ![](https://github.com/shalebark/AnimeOneShotRecognitionModelDemo/raw/master/images/arima2.jpg =300x) | TRUE

![](https://github.com/shalebark/AnimeOneShotRecognitionModelDemo/raw/master/images/tsubaki1.jpg =300)  |  ![](https://github.com/shalebark/AnimeOneShotRecognitionModelDemo/raw/master/images/tsubak2.jpg =300) | TRUE

![](https://github.com/shalebark/AnimeOneShotRecognitionModelDemo/raw/master/images/ryouta1.png =300)  |  ![](https://github.com/shalebark/AnimeOneShotRecognitionModelDemo/raw/master/images/kaori2.png =300) | FALSE

![](https://github.com/shalebark/AnimeOneShotRecognitionModelDemo/raw/master/images/tsubaki.jpg =300)  |  ![](https://github.com/shalebark/AnimeOneShotRecognitionModelDemo/raw/master/images/ryouta1.png =300) | TRUE

# NOTES

## Usage Tips

For this demo, I used the animeface python library, but that's mostly for demo purposes.

In actual data gathering, I use the following the following face detector: https://github.com/qhgz2013/anime-face-detector[https://github.com/qhgz2013/anime-face-detector].
I then use another repository for landmarks: https://github.com/kanosawa/anime_face_landmark_detection[https://github.com/kanosawa/anime_face_landmark_detection].

I found the animeface library to have difficulties detecting faces that are too big, too small, wears glasses, or tilted a certain way, and isn't quite suited for face detection from videos.

However, you may use any face detection or landmark detection tool you wish. Only the face detection tool is truly important, and as long as you can center the image such that the face is roughly centered, with the center of the face around the 50% mark, eyes around 45 and 55% marks respectively, and the output size being 150x150, the encoder should work.

## Tips for Multiple Images per Class

Sometimes you have multiple images of a character, and you wish to see if a new image matches that character.
I find the best strategy to handle that situation is to use either the average encoding of all images in that class or use the encoding closest to that average encoding.

The average encoding of all images in a class, is the set of encodings that best represents that image. I call this the "ideal encoding".
The best encoding is the encodes of an image that is closest to the ideal encoding, this can be done by finding the encoding with the shortest euclidean distance.

For my metrics, I used best encoding as it's possible to get an image that represents those encodings.
But if your concern is to just find a match, ideal encoding might be better.

# ISSUES & IMPROVEMENTS

## Size of Training Data

Training data is only 2300 images, this can be much bigger.

## Art Style and Demographics

Most of the anime series comes from 2010s to 2020s era. It can benefit from anime series other decades.

Some series possess art styles that are quite distinct from the rest. One Piece, Jojo, Crayon Shinchan are popular series but no data for them is collected.

Art styles unique to certain demographics like "shoujo" styled art has not been collected.

Characters that are not human or does not appear human are poorly represented in the dataset. Examples such as Nyanta from Log Horizon. Most face detectors I've found have difficulty
detecting characters that are not human, and therefore, their data is poorly represented.

