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
	
	std::string seed_text;
	std::vector< std::string > generated_texts;
};

