#include "ofApp.h"

void ofApp::setup() 
{
	std::string datacfg = "cfg/coco.data";
	std::string cfgfile = "cfg/yolo.cfg";
	std::string weightfile = "data/yolo.weights";
	std::string nameslist = "data/coco.list";
	darknet.init( cfgfile, weightfile, datacfg, nameslist );

	camWidth = 640;
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
	
	ofSetColor( 255 );
	video.draw( 0, 0 );

	if( video.isFrameNew() ) {
		std::vector< detected_object > detections = darknet.yolo( video.getPixelsRef(), thresh );

		video.draw( 0, 0 );
		ofNoFill();	
		for( detected_object d : detections )
		{
			ofSetColor( d.color );
			glLineWidth( ofMap( d.probability, 0, 1, 0, 8 ) );
			ofNoFill();
			ofDrawRectangle( d.rect );
			ofDrawBitmapStringHighlight( d.label + ": " + ofToString(d.probability), d.rect.x, d.rect.y + 20 );
		}
	}
}