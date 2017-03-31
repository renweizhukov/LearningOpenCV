# LearningOpenCV

Below are two example OpenCV Eclipse projects given in the tutorial http://docs.opencv.org/3.2.0/d7/d16/tutorial_linux_eclipse.html. 

## 1. DisplayImage

This OpenCV project is created in Eclipse from scratch.

## 2. HelloWorld

This OpenCV project is firstly created with CMake and then imported into Eclipse.

## 3. DisplayPicture

Modified from Example 2-2 in the book "Learning OpenCV 3: Computer Vision in C++ with the OpenCV Library".

## 4. PlayVideo

Modified from Example 2-3 in the book "Learning OpenCV 3: Computer Vision in C++ with the OpenCV Library".

## 5. SimpleTransformation

Modified from Example 2-5 in the book "Learning OpenCV 3: Computer Vision in C++ with the OpenCV Library".

## 6. FlannMatching1to1

Modified from the based on the tutorial "Feature Matching with FLANN" at http://docs.opencv.org/3.2.0/d5/d6f/tutorial_feature_flann_matcher.html.

Given two images, the executable will detect the keypoints via the SURF detector and compute the descriptors. Then it will match the descriptor vectors using the match() method of the FLANN matcher. At the end, 
it will find the "good" matches and display them by connecting the "good"-matched keypoints between the best matched source image and the target image.

```bash
./FlannMatching1to1 [image1] [image2]
```

## 7. FlannKnnMatching1to1

TODO

## 8. FlannMatching1toN

Given a directory of source images and a single target image, the executable will detect the keypoints via the SURF detector and compute the descriptors. Then it will match the descriptor vectors between 
each source image and the target image using the knnMatch() method of the FLANN matcher. At the end, it will find the "good" matches and display them by connecting the "good"-matched keypoints between the best matched source image and the target image.

Note that the executable has two modes: slow and fast. While it will do the match between the target image and each source image one-by-one in the slow mode, it will do the match between the target image and all the source images as a whole once in the fast mode. It is expected to see a significant running time reduction in the fast mode. 

```bash
./FlannMatching1toN [slow/fast] [source-image-directory] [target-image]
```


