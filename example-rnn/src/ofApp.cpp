#include "ofApp.h"

void ofApp::setup() 
{
	std::string cfgfile = "cfg/rnn.cfg";
	std::string weightfile = "data/parsed_v3_arts_arthistory_aesthetics_valid.weights";
	darknet.init( cfgfile, weightfile );

	camWidth = 640;  // try to grab at this size.
	camHeight = 480;

	video.setDeviceID( 0 );
	video.setDesiredFrameRate( 30 );
	video.initGrabber( camWidth, camHeight );
}

void ofApp::update()
{
	video.update();
}

void ofApp::draw()
{
	ofBackground( 0 );
	ofDrawBitmapStringHighlight( "Input: " + seed_text, 20, 20 );
	ofDrawBitmapStringHighlight( generated_text, 20, 60 );
}

void ofApp::keyReleased( int key )
{
	if( key == OF_KEY_BACKSPACE ) {
		seed_text = seed_text.substr( 0, seed_text.size() - 1 );
	}
	else {
		char k = key;
		seed_text += k;
	}
	
	generated_text = darknet.rnn( 50, seed_text, 0.8 );
}