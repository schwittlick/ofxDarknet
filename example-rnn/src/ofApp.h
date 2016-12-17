#pragma once

#include "ofMain.h"

#include "ofxDarknet.h"

class ofApp : public ofBaseApp
{
public:
	void setup();
	void update();
	void draw();
	void keyReleased( int key );

	ofxDarknet darknet;
	int camWidth, camHeight;
	ofVideoGrabber video;
	
	std::string seed_text, generated_text;
};

