#pragma once

#include <time.h>
#include <stdlib.h>
#include <stdio.h>

#define GPU

#include "ofMain.h"

#include "activations.h"
#include "avgpool_layer.h"
#include "activation_layer.h"
#include "convolutional_layer.h"
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
#include "detection_layer.h"
#include "region_layer.h"
#include "normalization_layer.h"
#include "reorg_layer.h"
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
    vector<float> features;
	ofColor color;
};

struct classification {
	std::string label;
	float probability;
};

struct activations {
    std::vector<float> acts;
    int rows;
    int cols;
    float min;
    float max;
    void getImage(ofImage & img);
};


class ofxDarknet
{
public:
	ofxDarknet();
	~ofxDarknet();

	void init( std::string cfgfile, std::string weightfile, std::string nameslist = "");
    bool isLoaded() {return loaded;}
    
    std::vector< classification > classify( ofPixels & pix, int count = 5 );
    std::vector< detected_object > yolo( ofPixels & pix, float threshold = 0.24f, float maxOverlap = 0.5f );
    std::vector< activations > getFeatureMaps(int idxLayer);
    //float * get_network_output_layer_gpu(int i);

    ofImage nightmare( ofPixels & pix, int max_layer, int range, int norm, int rounds, int iters, int octaves, float rate, float thresh );
    std::string rnn( int num, std::string seed, float temp );
	void train_rnn( std::string textfile, std::string cfgfile );
    
    network & getNetwork() {return net;}
    vector<string> getLayerNames() {return layerNames;}

protected:
    image convert( ofPixels & pix );
    ofPixels convert( image & image );
    
	list1 *options1;
	char **names;
    vector<string> layerNames;
	network net;
    bool loaded;
    bool labelsAvailable;
};


#include "ofxDarknetGo.h"
