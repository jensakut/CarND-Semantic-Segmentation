# Semantic Segmentation
### Introduction
In this project, the VGG-16 encoder will be used to classify road and non-road in a pixel-by-pixel accuracy  [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php). 


### Project description
#### Dataset
The Kitti Road dataset includes road, non-drivable road and non-road. In respect to the gpu, a 970m, a binary classification is executed. The subset used contains 289 classified images for training and 290 unclassified for verification. 

#### Code structure
The [SemanticSegmentation Notebook](https://github.com/jensakut/CarND-Semantic-Segmentation) consists of several functions. After checking gpu-preconditions, the main net-parameters are defined. The function load-vgg loads up the model and gets tested by a helper function. The next block defines and tests the encoder implementation. 
Optimize and trainNN define the neural net and get called by run. run then uses the trained net to label test data. 

#### Tuning of hyperparameters
A parameter variation was performed to find a good parameter set for kernel regularizer and kernel initializer, learning rate, dropout coefficient, and the influence of augmentation. 

Choosing the right meta-parameters first enabled basic classification (kernel regularizer and initializer) and improved it (learning rate, dropout coefficient). Using flipped images to double the number of classified examples and training with random brightness and contrast improved the performance significantly, especially when shadows increase detection difficulty. 

Using the pipeline on a video enables some continuity assumptions. A simple approach would be thresholded averaging over several frames. The [optical flow libraries](https://docs.opencv.org/3.3.1/d7/d8b/tutorial_py_lucas_kanade.html) are very promising as well.  


#### The control group in a gif

![](result.gif)


#### Further steps

The loss doesn't monotonically decrease. Maybe this is due to the aggressive augmentation and a very small sample size but it may also indicate a bug. 

The dataset used contains just 300 classified pictures, which is a good start, but given the amount of features to be learnt, even with good augmentation the results seem limited. Sticking with this dataset, further augmentation seems promising. 

Therefore, training this net with the cityscapes dataset using the road class would be very interesting. Right now, my account and the license to use the dataset is pending. 

In this implementation, the loss and a manual visual inspection of the classified pictures serve as a measure of quality. Implementing IoU would improve and simplify hyperparameter optimization.

Freezing the graph, fusing nodes, inference optimization and reducing precision to 8 bit, improves the speed of the network, thus it could be used to classify videos significantly faster. 



### Setup
##### GPU
`main.py` will check to make sure you are using GPU - if you don't have a GPU on your system, you can use AWS or another cloud computing platform.
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Start
##### Implement
Implement the code in the `main.py` module indicated by the "TODO" comments.
The comments indicated with "OPTIONAL" tag are not required to complete.
##### Run
Run the following command to run the project:
```
python main.py
```
**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.

### Submission
1. Ensure you've passed all the unit tests.
2. Ensure you pass all points on [the rubric](https://review.udacity.com/#!/rubrics/989/view).
3. Submit the following in a zip file.
 - `helper.py`
 - `main.py`
 - `project_tests.py`
 - Newest inference images from `runs` folder  (**all images from the most recent run**)
 
 ### Tips
- The link for the frozen `VGG16` model is hardcoded into `helper.py`.  The model can be found [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip).
- The model is not vanilla `VGG16`, but a fully convolutional version, which already contains the 1x1 convolutions to replace the fully connected layers. Please see this [post](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/forum_archive/Semantic_Segmentation_advice.pdf) for more information.  A summary of additional points, follow. 
- The original FCN-8s was trained in stages. The authors later uploaded a version that was trained all at once to their GitHub repo.  The version in the GitHub repo has one important difference: The outputs of pooling layers 3 and 4 are scaled before they are fed into the 1x1 convolutions.  As a result, some students have found that the model learns much better with the scaling layers included. The model may not converge substantially faster, but may reach a higher IoU and accuracy. 
- When adding l2-regularization, setting a regularizer in the arguments of the `tf.layers` is not enough. Regularization loss terms must be manually added to your loss function. otherwise regularization is not implemented.
 
### Using GitHub and Creating Effective READMEs
If you are unfamiliar with GitHub , Udacity has a brief [GitHub tutorial](http://blog.udacity.com/2015/06/a-beginners-git-github-tutorial.html) to get you started. Udacity also provides a more detailed free [course on git and GitHub](https://www.udacity.com/course/how-to-use-git-and-github--ud775).

To learn about REAMDE files and Markdown, Udacity provides a free [course on READMEs](https://www.udacity.com/courses/ud777), as well. 

GitHub also provides a [tutorial](https://guides.github.com/features/mastering-markdown/) about creating Markdown files.
