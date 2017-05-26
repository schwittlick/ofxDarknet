#include "ofApp.h"

void ofApp::setup()
{
    string cfgfile = ofToDataPath("cfg/go.test.cfg");
    string weightfile = "/Users/gene/Learn/darknet/go.weights";//ofToDataPath("go.weights");
    
    darknet.setup(cfgfile, weightfile);
    darknet.setMouseActive(true);
    darknet.setDrawPosition(50, 50, 420, 420);
}

void ofApp::update() {
}

void ofApp::draw(){
    darknet.draw();
}

void ofApp::keyPressed(int key) {
    if (key== ' ') {
        darknet.nextAuto();
    }
}
