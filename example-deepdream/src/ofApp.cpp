#include "ofApp.h"

void ofApp::setup() 
{
	dog.loadImage( "dog.jpg" );
	dog.resize( 640, 480 );
	
	std::string cfgfile = "cfg/vgg-conv.cfg";
	std::string weightfile = "vgg-conv.weights";
	darknet.init( cfgfile, weightfile );
	nightmare = darknet.nightmate( dog.getPixelsRef() );
}

void ofApp::update()
{
}

void ofApp::draw()
{
	nightmare.draw( 0, 0, ofGetWidth(), ofGetHeight() );
}