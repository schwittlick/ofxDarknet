#pragma once

#include "ofMain.h"
#include "ofxDarknet.h"

class ofApp : public ofBaseApp
{
public:
    void setup();
    void update();
    void draw();
    void keyPressed(int key);
    void mouseMoved(int x, int y);
    void mouseDragged(int x, int y, int button);
    void mouseScrolled(ofMouseEventArgs &evt);
    void mousePressed(int x, int y, int button);
    
    void drawFeatureMaps();
    void drawClassifications();
    
    void setSourceWebcam();
    void setSourceImage(string path);
    
    ofImage pic;
    ofVideoGrabber grab;
    
    ofxDarknet darknet;
    vector<activations > maps;
    vector<classification > classifications;
    
    int layer;
    int inputMode;
    int highlighted;
    int scroll;
    int maxPageLength;
    int numClassifications;
};
