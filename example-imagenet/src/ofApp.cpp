#include "ofApp.h"

void ofApp::setup() 
{
	std::string datacfg = "cfg/imagenet1k.data";
	std::string cfgfile = "cfg/darknet.cfg";
	std::string weightfile = "data/darknet.weights";
	std::string nameslist = "data/imagenet.shortnames.list";
	darknet.init( cfgfile, weightfile, datacfg, nameslist );

	video.setDeviceID( 0 );
	video.setDesiredFrameRate( 30 );
	video.initGrabber( 640, 480 );
}

void ofApp::update()
{
	ofLog() << ofGetFrameRate();
	video.update();
}

void ofApp::draw()
{
	video.draw( 0, 0 );

	if( video.isFrameNew() ) {
		classifications = darknet.classify( video.getPixelsRef() );
	}
	
	int offset = 20;
	for( classification c : classifications )
	{
		std::stringstream ss;
		ss << c.label << " : " << ofToString( c.probability );
		ofDrawBitmapStringHighlight( ss.str(), 20, offset );
		offset += 20;
	}
}