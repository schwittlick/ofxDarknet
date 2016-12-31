#include "ofApp.h"

void ofApp::setup() 
{
    video.initGrabber( 640, 480 );
    
    string datacfg = "/Users/gene/Learn/darknet_static/cfg/COCO_COPY_TEMP.data";
    string cfgfile = "/Users/gene/Learn/darknet/cfg/yolo.cfg";
    string weightfile = "/Users/gene/Learn/darknet_static/yolo.weights";
    string filename = "/Users/gene/Learn/darknet/data/dog.jpg";
    string nameslist = "/Users/gene/Learn/darknet/data/coco.list";
    
    darknet.init( cfgfile, weightfile, datacfg, nameslist );
}

void ofApp::update()
{
	ofLog() << ofGetFrameRate();
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