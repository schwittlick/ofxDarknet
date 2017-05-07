#pragma once

#include "ofMain.h"

#include "ofxDarknet.h"
#include "ofxGui.h"

class ofApp : public ofBaseApp
{
public:
	void setup();
	void update();
	void draw();
    void keyPressed(int key);
    
    void makeDream();
    void startDream(bool & b);
    
	ofxDarknet darknet;
    ofVideoGrabber cam;
	ofImage nightmare;
    
    ofParameter<bool> isDreaming;
    ofParameter<int> max_layer;
    ofParameter<int> iters;
    ofParameter<int> octaves;
    ofParameter<float> thresh;
    ofParameter<int> range;
    ofParameter<int> norm;
    ofParameter<float> rate;
    ofParameter<float> blendAmt;

    ofxPanel gui;
};

