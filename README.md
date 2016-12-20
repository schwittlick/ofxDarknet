# ofxDarknet

ofxDarknet is a openFrameworks wrapper for darknet.

Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation. http://pjreddie.com/darknet/

## Features

### YOLO: Real-Time Object Detection aka Dense Captioning (http://pjreddie.com/darknet/yolo/)

Darknet comes with two pre-trained models for this task. Additionally each has a smaller (and faster) but therefore less accurate version:

MS COCO dataset (80 different classes)
* yolo.cfg & [yolo.weights](http://pjreddie.com/media/files/yolo.weights) (256 MB COCO-model)
* tiny-yolo.cfg & [tiny-yolo.weights](http://pjreddie.com/media/files/tiny-yolo.weights) (60 MB COCO-model)

```
	std::string datacfg = "cfg/coco.data";
	std::string cfgfile = "cfg/tiny-yolo.cfg";
	std::string weightfile = "data/tiny-yolo.weights";
	std::string nameslist = "data/names.list";
	darknet.init( cfgfile, weightfile, datacfg, nameslist );
```

Pascal VOC dataset (20 different classes)
* yolo-voc.cfg & [yolo-voc.weights](http://pjreddie.com/media/files/yolo-voc.weights) (256 MB VOC-model)
* tiny-yolo-voc.cfg & [tiny-yolo-voc.weights](http://pjreddie.com/media/files/tiny-yolo-voc.weights) (60 MB VOC-model)

```
	std::string datacfg = "cfg/voc.data";
	std::string cfgfile = "cfg/tiny-yolo-voc.cfg";
	std::string weightfile = "data/tiny-yolo-voc.weights";
	std::string nameslist = "data/voc.names";
	darknet.init( cfgfile, weightfile, datacfg, nameslist );
```

```
	float thresh = 0.25;
	std::vector< detected_object > detections = darknet.yolo( image.getPixelsRef(), thresh );

	for( detected_object d : detections )
	{
		ofSetColor( d.color );
		glLineWidth( ofMap( d.probability, 0, 1, 0, 8 ) );
		ofNoFill();
		ofDrawRectangle( d.rect );
		ofDrawBitmapStringHighlight( d.label + ": " + ofToString(d.probability), d.rect.x, d.rect.y + 20 );
	}
```

![YOLO2](https://raw.githubusercontent.com/mrzl/ofxDarknet/master/images/yolo2.jpg)

### Imagenet Classification (http://pjreddie.com/darknet/imagenet/)

In order to classify an image with more classes, this is the spot. This classifies an image according to the 1000-class [ImageNet Challenge](http://image-net.org/challenges/LSVRC/2015/index).

* [AlexNet](http://pjreddie.com/darknet/imagenet/#alexnet)
cfg: [alexnet.cfg](https://github.com/mrzl/ofxDarknet/blob/master/example-imagenet/src/cfg/alexnet.cfg) weights: [alexnet.weights](http://pjreddie.com/media/files/alexnet.weights)
* [Darknet Reference](http://pjreddie.com/darknet/imagenet/#reference)
cfg: [darknet.cfg](https://github.com/mrzl/ofxDarknet/blob/master/example-imagenet/src/cfg/darknet.cfg) weights: [darknet.cfg](http://pjreddie.com/media/files/darknet.weights)
* [VGG-16](http://pjreddie.com/darknet/imagenet/#vgg)
cfg: [vgg-16.cfg](https://github.com/mrzl/ofxDarknet/blob/master/example-imagenet/src/cfg/vgg-16.cfg) weights: [vgg-16.weights](http://pjreddie.com/media/files/vgg-16.weights)
* [Extraction](http://pjreddie.com/darknet/imagenet/#extraction)
cfg: [extraction.cfg](https://github.com/mrzl/ofxDarknet/blob/master/example-imagenet/src/cfg/extraction.cfg) weights: [extraction.weights](http://pjreddie.com/media/files/extraction.weights)
* [Darknet19](http://pjreddie.com/darknet/imagenet/#darknet19)
cfg: [darknet16.cfg](https://github.com/mrzl/ofxDarknet/blob/master/example-imagenet/src/cfg/darknet19.cfg) weights: [darknet19.weights](http://pjreddie.com/media/files/darknet19.weights)
* [Darknet19 448x448](http://pjreddie.com/darknet/imagenet/#darknet19_448)
cfg: [darknet19_488.cfg](https://github.com/mrzl/ofxDarknet/blob/master/example-imagenet/src/cfg/darknet19_448.cfg) weights: [darknet19_488.weights](http://pjreddie.com/media/files/darknet19_448.weights)

```
	std::string datacfg = "cfg/imagenet1k.data";
	std::string cfgfile = "cfg/darknet.cfg";
	std::string weightfile = "darknet.weights";
	std::string nameslist = "data/imagenet.shortnames.list";
	darknet.init( cfgfile, weightfile, datacfg, nameslist );

	classifications = darknet.classify( image.getPixelsRef() );
	int offset = 20;
	for( classification c : classifications )
	{
		std::stringstream ss;
		ss << c.label << " : " << ofToString( c.probability );
		ofDrawBitmapStringHighlight( ss.str(), 20, offset );
		offset += 20;
	}
```

![Classification](https://raw.githubusercontent.com/mrzl/ofxDarknet/master/images/imagenet_classification.jpg)

### Deep Dream (http://pjreddie.com/darknet/nightmare/)

[vgg-conv.cfg](https://github.com/mrzl/ofxDarknet/blob/master/example-deepdream/src/cfg/vgg-conv.cfg) & [vgg-conv.weights](http://pjreddie.com/media/files/vgg-conv.weights)

```
	std::string cfgfile = "cfg/vgg-conv.cfg";
	std::string weightfile = "vgg-conv.weights";
	darknet.init( cfgfile, weightfile );
	
	int max_layer = 13;
	int range = 3;
	int norm = 1;
	int rounds = 4;
	int iters = 20;
	int octaves = 4;
	float rate = 0.01;
	float thresh = 1.0;
	nightmare = darknet.nightmate( image.getPixelsRef(), max_layer, range, norm, rounds, iters, octaves, rate, thresh );
```

![DeepDream](https://raw.githubusercontent.com/mrzl/ofxDarknet/master/images/deep_dream.jpg)

### Recurrent Neural Network (http://pjreddie.com/darknet/rnns-in-darknet/)

Darknet pre-trained weights files:
* [Shakespeare](http://pjreddie.com/media/files/grrm.weights)
* [George R.R. Martin](http://pjreddie.com/media/files/shakespeare.weights)
* [Leo Tolstoy](http://pjreddie.com/media/files/tolstoy.weights)
* [Immanuel Kant](http://pjreddie.com/media/files/kant.weights)

ofxDarknet custom pre-trained weight files (each trained for 20h on NVidia TitanX):
* [Anonymous - Hypersphere](http://mrzl.net/ofxdarknet/anonymous-hypersphere.weights)
Hypersphere, written by Anonymous with the help of the 4chan board /lit/ (of The Legacy of Totalitarianism in a Tundra fame) is an epic tale spanning over 700 pages.
A postmodern collaborative writing effort containing Slavoj Žižek erotica, top secret Donald Trump emails, poetry, repair instructions for future cars, a history of bottles in the Ottoman empire; actually, it contains everything since it takes place in the Hypersphere, and the Hypersphere is a big place; really big in fact.
* [Books on art history & aesthetics](http://mrzl.net/ofxdarknet/arts_arthistory_aesthetics.weights)
* [Books on digital culture](http://mrzl.net/ofxdarknet/digital_and_internet_theory.weights)


```
	std::string cfgfile = "cfg/rnn.cfg";
	std::string weightfile = "data/shakespeare.weights";
	darknet.init( cfgfile, weightfile );

	int character_count = 100;
	float temperature = 0.8;
	std::string seed_text = "openframeworks is ";
	std::string generated_text = darknet.rnn( character_count, seed_text, temperature );
```

![RNN](https://raw.githubusercontent.com/mrzl/ofxDarknet/master/images/rnn.jpg)

You can train your own RNN models with darknet

```
	// no need to init
	darknet.train_rnn( "D:\\path\\to\\text\\text.txt", "cfg/rnn.cfg" );
```

## Setup

### Windows

Install the dependencies for building darknet on Windows 10:
* [Visual Studio 2015 (Community)](https://www.microsoft.com/download/details.aspx?id=48146)
* [CUDA 8.0 64bit](https://developer.nvidia.com/cuda-downloads)
* [OpenCV 2.4.9](https://sourceforge.net/projects/opencvlibrary/files/opencv-win/2.4.9/opencv-2.4.9.exe/download)

### OSX

An OSX version is on the way and will be updated here..

## Training your own models

### YOLO

tcb

## Credits

* Original Code: https://github.com/pjreddie/darknet
* Help to compile on Windows: https://github.com/AlexeyAB/darknet/
* Help to call from C++: https://github.com/prabindh/darknet
