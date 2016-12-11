#include "ofApp.h"

void ofApp::setup() 
{
	char *datacfg = "cfg/coco.data";
	char *cfgfile = "cfg/yolo.cfg";
	char *weightfile = "yolo.weights";
	char *nameslist = "data/names.list";

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

	if( video.isFrameNew() ) {
		std::vector< detected_object > detections = darknet.yolo( video.getPixelsRef(), thresh );

		ofSetColor( 255 );
		video.draw( 0, 0 );
		ofNoFill();
		for( detected_object d : detections )
		{
			ofDrawRectangle( d.rect );
		}
	}
}