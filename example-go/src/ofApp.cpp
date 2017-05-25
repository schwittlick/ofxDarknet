#include "ofApp.h"

void ofApp::setup()
{
    string cfgfile = ofToString("cfg/go.test.cfg");
    string weightfile = ofToString("go.weights");
    
    darknet.setup(cfgfile, weightfile);
    darknet.setDrawPosition(100, 50, 600, 600);
}

void ofApp::update()
{
}

void ofApp::draw()
{
    darknet.draw();
}

void ofApp::keyPressed(int key) {
    if (key== ' ') {
        //darknet.go_next();
    }
}
