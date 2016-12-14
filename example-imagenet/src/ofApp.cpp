#include "ofApp.h"

void ofApp::setup() 
{
	char *datacfg = "cfg/imagenet1k.data";
	char *cfgfile = "cfg/darknet.cfg";
	char *weightfile = "darknet.weights";
	char *nameslist = "data/labels.list";

	darknet.init( datacfg, cfgfile, weightfile, nameslist );

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
	float thresh = ofMap( ofGetMouseX(), 0, ofGetWidth(), 0, 1 );
	
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