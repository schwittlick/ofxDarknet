#include "ofApp.h"

void ofApp::setup() 
{
	dog.loadImage( "dog.jpg" );
	dog.resize( 640, 480 );
	
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
	nightmare = darknet.nightmate( dog.getPixelsRef(), max_layer, range, norm, rounds, iters, octaves, rate, thresh );
}

void ofApp::update()
{
}

void ofApp::draw()
{
	nightmare.draw( 0, 0, ofGetWidth(), ofGetHeight() );
}