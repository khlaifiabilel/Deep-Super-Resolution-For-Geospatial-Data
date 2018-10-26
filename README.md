# VDSR4Geo #
TensorFlow implementation of "Accurate Image Super-Resolution Using Very Deep Convolutional Networks" adapted for working with geospatial data


### For more information, see:


Initial blog on Super-Resolution: TBA November 2019

arXiv paper: TBA November 2019

VDSR original paper: [Kim et al. 2016](https://arxiv.org/pdf/1511.04587.pdf)

____
## Running VDSR4Geo

____

### 0. Create Docker file
All commands should be run in docker, create it via the following commands

	cd ./Docker
	nvidia-docker build -t VDSR4Geo ./
	NV_GPU=0 nvidia-docker run -it -v /home:/home/ --name VDSR_GPU0 VDSR4Geo


____

### 1. Train

Training will rely on one set of imagery, you should provide your high-resolution (HR) target imagery to the model. All imagery will be automatically degraded and downsampled depending on your scale to a lower quality, then a model will be created to attempt to relearn the HR target imagery.  A few specific items need to be adjusted to match your file structure and environment.  Presently, VDSR4Geo is built to handle only 8bit RGB imagery.

##### -Open data.py

Set your DATA_PATH to you working directory containing a training set and test/validation set.  The training and test set should be subdirectories within the DATA_PATH.

Set your desired scaling factors in the function headers for the TrainSet, TestSet, and SR_Run classes.  

This code will naturally blur your imagery with a simulated PSF.  You may want to turn this off, simply comment out a few lines all the classes to do this.

##### -Open params.json

Specify the folder names of your training and validation/test set ("train_set" and "validation_set").  Again, these should be subdirectories in the DATA_PATH folder you just specified.

The validation set can be a smaller subset of your total imagery dataset.  For training I would recommend a subset of imagery not much larger the 0.5GB.  With augmentation, conversion to floating point, and generation of tensors, a dataset this large can get quite memory intensive. On an NVIDIA Titan X with this much data, training can take about 55 hours.  With less data, training will naturally be shorter, but results may be worse.

##### Big images?
If you have massive images they may need to be tiled into smaller chunks to make your GPU happy.  This will remove the geospatial information, but fear not!  We have code to stitch these images back together after you run inference, and add accurate geospatial info back as well.  Check out our tiler, stitcher and georeferencing package here: TBA

##### Time to train

	python3 train.py

____

### 2. Test
We can now test our models on larger independet datasets.  

##### -Open test.py

Edit the set_name and scaling_factors.  The set_name should again be a subdirectory in the DATA_PATH set in data.py.  Feel free to readjust this.  Again you should be feeding the model the proper resolution data for this to work.  Like in the training process, testing will naturally degrade your imagery, then create an SR output and attempt to score it against the oringal HR image.   Testing will provide PSNR and SSIM scores per level of enhancement.  Note that if you tile your imagery it will affect your scoring!  Particularly SSIM, in smaller images finer details are considered more significant than if you are using a larger images.


### 3. Output super resolved images

Here we will use either Create_SR.py or Create_SR_NoGEO.py.  If you have georeferencing information saved in your imagery, I would recommend Create_SR to maintain it.  Our input here is different than our previous two tasks.  If you have a model built to super-resolve imagery to 30cm, you should input 60cm imagery for a 2x enhancement, 120cm for a 4x, and 240cm for an 8x.  Run a command similar to the one below:

	python3 Create_SR.py "/input/60cmdata/" "/output/30cmSRdata/" 2
	python3 Create_SR_NoGEO.py "/input/240cmdata/" "/output/60cmSRdata/" 4
    
### 4.  Optionally stitch and add georeferencing

Again, if you have tiled data, you can use our stitcher and georeferencing code found here: TBA


    
    
    
    
    



