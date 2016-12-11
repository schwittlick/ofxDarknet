#pragma once

#include "ofMain.h"

#include "ofxDarknet.h"

class ofApp : public ofBaseApp
{
public:
	void setup();
	void update();
	void draw();

	ofxDarknet darknet;
	int camWidth, camHeight;
	ofVideoGrabber video;
};

