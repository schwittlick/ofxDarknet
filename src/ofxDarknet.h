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
#include "nightmare.h"
#include "rnn.h"

#include "ofxOpenCv.h"

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#endif

struct detected_object {
	ofRectangle rect;
	std::string label;
	float probability;
	ofColor color;
};

struct classification {
	std::string label;
	float probability;
};

class ofxDarknet
{
public:
	ofxDarknet();
	~ofxDarknet();

	void init( std::string cfgfile, std::string weightfile, std::string datacfg = "cfg/coco.data", std::string nameslist = "data/names.list" );
	std::vector< detected_object > yolo( ofPixels & pix, float threshold = 0.24f );
	ofImage nightmate( ofPixels & pix );
	std::vector< classification > classify( ofPixels & pix );
	std::string rnn( int num, std::string seed, float temp );

private:
	list1 *options1;
	char *name_list;
	char **names;

	image **alphabet;
	network net;

	image convert( ofPixels & pix );
	ofPixels convert( image & image );
	char * str2char( std::string string );
};