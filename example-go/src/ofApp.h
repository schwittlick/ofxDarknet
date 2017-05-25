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
    
	ofxDarknetGo darknet;
    ofVideoGrabber cam;

    ofxPanel gui;
};

