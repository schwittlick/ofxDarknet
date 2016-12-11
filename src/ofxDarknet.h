#pragma once

#include <time.h>
#include <stdlib.h>
#include <stdio.h>

#include "ofMain.h"

#include "option_list.h"
#include "image.h"
#include "parser.h"
#include "list.h"
#include "box.h"
#include "tree.h"
#include "layer.h"
#include "matrix.h"
#include "connected_layer.h"
#include "gru_layer.h"
#include "rnn_layer.h"
#include "crnn_layer.h"
#include "activation_layer.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "normalization_layer.h"
#include "reorg_layer.h"
#include "avgpool_layer.h"
#include "cost_layer.h"
#include "softmax_layer.h"
#include "route_layer.h"
#include "shortcut_layer.h"
#include "cuda.h"
#include "utils.h"

#include "ofxOpenCv.h"

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#endif

struct detected_object {
	ofRectangle rect;
	std::string label;
	float probability;
};

class ofxDarknet
{
public:
	ofxDarknet();
	~ofxDarknet();

	void init( char *datacfg = "cfg/coco.data", char *cfgfile = "cfg/yolo.cfg", char *weightfile = "yolo.weights", char *nameslist = "data/names.list" );
	std::vector< detected_object > detect( ofPixels & pix, float threshold = 0.24f );

private:
	list1 *options1;
	char *name_list;
	char **names;

	image **alphabet;
	network net;

	image convert( ofPixels & pix );
};