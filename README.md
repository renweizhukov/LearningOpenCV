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

One example:

```bash
./FlannMatching1to1 ../../Pictures/airplanes/image_0001.jpg ../../Pictures/image_0001-new.jpg
```

## 7. FlannKnnMatching1to1

This executable is similar to FlannMatching1to1 where the main difference is that it uses the knnMatch() method of the FLANN matcher.

```bash
./FlannKnnMatching1to1 [image1] [image2]
```

One example:

```bash
./FlannKnnMatching1to1 ../../Pictures/airplanes/image_0001.jpg ../../Pictures/image_0001-new.jpg
```

## 8. FlannMatching1toN

Given a directory of source images and a single target image, the executable will detect the keypoints via the SURF/ORB detector and compute the descriptors. Then it will match the descriptor vectors between 
each source image and the target image using the knnMatch() method of the FLANN matcher. At the end, it will find the "good" matches and display them by connecting the "good"-matched keypoints between the best matched source image and the target image.

Note that the executable has two modes: slow and fast. While it will do the match between the target image and each source image one-by-one in the slow mode, it will do the match between the target image and all the source images as a whole once in the fast mode. It is expected to see a significant running time reduction in the fast mode. 

```bash
./FlannMatching1toN [surf/orb] [slow/fast] [source-image-directory] [target-image]
```

Two examples:

```bash
./FlannMatching1toN surf slow ../../Pictures/airplanes ../../Pictures/image_0001-new.jpg
./FlannMatching1toN surf fast ../../Pictures/airplanes ../../Pictures/image_0001-new.jpg
./FlannMatching1toN orb slow ../../Pictures/airplanes ../../Pictures/image_0001-new.jpg
./FlannMatching1toN orb fast ../../Pictures/airplanes ../../Pictures/image_0001-new.jpg
```

## 9. FlannKnnMatching1toN

This executable is similar to FlannMatching1to1 where the main difference is that it uses the knnMatch() method of the FLANN matcher.

```bash
./FlannKnnMatching1toN [slow/fast] [source-image-directory] [target-image]
```

Two examples:

```bash
./FlannKnnMatching1toN surf slow ../../Pictures/airplanes ../../Pictures/image_0001-new.jpg
./FlannKnnMatching1toN surf fast ../../Pictures/airplanes ../../Pictures/image_0001-new.jpg
./FlannKnnMatching1toN orb slow ../../Pictures/airplanes ../../Pictures/image_0001-new.jpg
./FlannKnnMatching1toN orb fast ../../Pictures/airplanes ../../Pictures/image_0001-new.jpg
```

## 10. BowSvmClassifier

This executable implements a simple object classifier with Bag-of-Word (BOW) and 1-vs-all SVM using OpenCV as introduced in http://www.morethantechnical.com/2011/08/25/a-simple-object-classifier-with-bag-of-words-using-opencv-2-3-w-code/. Note that it depends on a newer version 3.2.0 of OpenCV, and it is compiled and tested on Ubuntu 16.04LTS. 

To display the information about the available commands and options, please run the help command.

```bash
$ ./BowSvmClassifier help
```

Besides the help command, this executable has three other commands: build, train, and test.

### 10.1 Build the vocabulary.

Below is an example build command.

```bash
./BowSvmClassifier build -d ./train-images -e ./descriptors.yml -v ./vocabulary.yml
```

The SURF descriptors of the images are written to descriptors.yml and the BOW vocabulary is written to vocabulary.yml. Note that the labelled images need to be stored in the following hierachical tree:

```
train-images
├── label1
│   ├── image11
│   ├── image12
│   └── image13
└── label2
    ├── image21
    └── image22 
``` 

### 10.2 Train the 1-vs-all SVM classifiers.

Below is an example train command.

```bash
./BowSvmClassifier train -p ./SvmClassifier -e ./descriptors.yml -v ./vocabulary.yml
```

The SURF descriptors of the train images are loaded from descriptor.yml, so they don't need to be computed again. The BOW vocabulary is loaded from vocabulary.yml. The 1-vs-all SVM classifiers are saved in a set of yml files with the common prefix "SvmClassifier".

### 10.3 Test the 1-vs-all SVM classifiers.

(1) To test one image,

```bash
./BowSvmClassifier test -p ./SvmClassifier -c o -i ./test.jpg -r ./result.yml -v ./vocabulary.yml
```

The option "-c" specifies the true class or the expected class of the test image. The evaluated class and the decision function values of all the 1-vs-all SVM classifiers are written to result.yml.

(2) To test a set of images,

```bash
./BowSvmClassifier test -p ./SvmClassifier -d ./test-images -r ./results.yml -v ./vocabulary.yml
```

Note that the test images need to be stored in a tree similar to the one for building the vocabulary above, where the true class or the expected class of each image is given by its directory name.

## 11. SimpleHsvHistComparison

This executable converts two BGR-colored images into HSV, computes their single-channel or multi-channel histograms, and then compares their histograms via various methods. Note that 

* While BGR is Blue, Green, and Red, HSV is Hue, Saturation, and Value. 
* Any combination of the three h, s, and v channels may be specified for computing and comparing the histograms. If not specified, the default value "hs" will be used. Also, if only one channel is specified, the two histograms will be drawn up and down, next to each other.
* One of the following comparison methods may be specified: 

    correl = Correlation (method = CV_COMP_CORREL), 
    chisqr = Chi-Square (method = CV_COMP_CHISQR), 
    chisqr_alt = Alternative Chi-Square (method = CV_COMP_CHISQR_ALT), 
    intersection = Intersection (method = CV_COMP_INTERSECT), 
    bhattacharyya or hellinger = Bhattacharyya distance (method = CV_COMP_BHATTACHARYYA or method = CV_COMP_HELLINGER), 
    kl_div = Kullback-Leibler divergence (method = CV_COMP_KL_DIV),
    
  If "all" is specified, the histograms will be compared via all the above methods. If not specified, the default value "correl" will be used.

To get the help info,

```bash
$./SimpleHsvHistComparison -h
```

Below are a couple of sample commands.

```bash
$./SimpleHsvHistComparison ./image1.jpg ./image2.jpg
$./SimpleHsvHistComparison ./image1.jpg ./image2.jpg -c h -m chisqr_alt
$./SimpleHsvHistComparison ./image1.jpg ./image2.jpg -c hs -m all
```

## 12. EmdHsvHistComparison

This executable converts two BGR-colored images into HSV, computes their single-channel or double-channel histograms, and then compares their histograms via the Earth Mover's Distance (EMD) method. Note that

* Only one or two of the HSV channels may be specified for computing and comparing the histograms. If not specified, the default value "hs" will be used. Also, if only one channel is specified, the two histograms will be drawn up and down, next to each other.
* The current EMD implementation in OpenCV supports the following three definitions of distance: 

    l1: Manhattan distance, 
    l2: Euclidean distance, 
    c: Checkboard distance, 
    
  If "all" is specified, the histograms will be compared with all the above distances. If not specified, the default value "l1" will be used.

To get the help info,

```bash
$./EmdHsvHistComparison -h
```

Below are a couple of sample commands.

```bash
$./EmdHsvHistComparison ./image1.jpg ./image2.jpg
$./EmdHsvHistComparison ./image1.jpg ./image2.jpg -c h -m l1
$./EmdHsvHistComparison ./image1.jpg ./image2.jpg -c hs -m all 
```

## 13. BgrHistComparison

This executable computes the single-channel (Blue, Green, or Red) histograms of two BGR-colored images and then compares their histograms via various methods. Note that

* Any combination of the three b, g, and r channels may be specified for computing and comparing the histograms. If not specified, default bgr. 
* One histogram is generated for each specified channel.
* One of the following comparison methods may be specified: 

    correl = Correlation (method = CV_COMP_CORREL), 
    chisqr = Chi-Square (method = CV_COMP_CHISQR), 
    chisqr_alt = Alternative Chi-Square (method = CV_COMP_CHISQR_ALT), 
    intersection = Intersection (method = CV_COMP_INTERSECT), 
    bhattacharyya or hellinger = Bhattacharyya distance (method = CV_COMP_BHATTACHARYYA or method = CV_COMP_HELLINGER), 
    kl_div = Kullback-Leibler divergence (method = CV_COMP_KL_DIV), 
    emd = Earth Mover's Distance 
    
  If "all" is specified, the histograms will be compared via all the above methods. If not specified, the default value "correl" will be used.

To get the help info,

```bash
$./BgrHistComparison -h
```

Below are a couple of sample commands.

```bash
$./BgrHistComparison ./image1.jpg ./image2.jpg
$./BgrHistComparison ./image1.jpg ./image2.jpg -c b -m emd -d l2
$./BgrHistComparison ./image1.jpg ./image2.jpg -c bgr -m all
```

## 14. SimpleTemplateMatching

This executable matches a template image patch against an input image by "sliding" the patch over the input image using one of cv::matchTemplate()'s available matching method. The best match for the template image will be marked by a black rectangle in the input image. Note that 

* Although both images are always loaded as BGR images, the template matching can be done with four different channel type: bgr, grayscale, hsv, and hs.
* The cv::matchTemplate()'s available matching methods are as below: 

    Square Difference (method = cv::TM_SQDIFF) 
    Normalized Square Difference (method = cv::TM_SQDIFF_NORMED) 
    Cross Correlation (method = cv::TM_CCORR) 
    Normalized Cross Correlation (method = cv::TM_CCORR_NORMED) 
    Correlation Coefficient (method = cv::TM_CCOEFF) 
    Normalized Correlation Coefficient (method = cv::TM_CCOEFF_NORMED) 
    
  The matching method can be changed by moving the track bar at the top of the input image window.

To get the help info,

```bash
$./SimpleTemplateMatching -h
```

Below are a couple of sample commands.

```bash
$./SimpleTemplateMatching ./inputImg.jpg ./templateImg.jpg
$./SimpleTemplateMatching ./inputImg.jpg ./templateImg.jpg -c grayscale
$./SimpleTemplateMatching ./inputImg.jpg ./templateImg.jpg -c hs
```