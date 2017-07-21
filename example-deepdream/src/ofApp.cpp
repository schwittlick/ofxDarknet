#include "ofApp.h"

void ofApp::setup()
{
    string cfgfile = ofToDataPath("cfg/vgg-conv.cfg");
    string weightfile = "vgg-conv.weights";
    darknet.init( cfgfile, weightfile );

    isDreaming.addListener(this, &ofApp::startDream);
    
    gui.setup();
    gui.setName("darkdream");
    gui.add(isDreaming.set("dream", false));
    gui.add(max_layer.set("max_layer", 13, 1, 13));
    gui.add(iters.set("iterations", 1, 1, 10));
    gui.add(octaves.set("octaves", 2, 1, 8));
    gui.add(thresh.set("thresh", 0.85, 0.0, 1.0));
    gui.add(range.set("range", 3, 1, 10));
    gui.add(norm.set("norm", 1, 1, 10));
    gui.add(rate.set("rate", 0.01, 0.0, 0.1));
    gui.add(blendAmt.set("blendAmt", 0.5, 0.0, 1.0));

    cam.initGrabber(320, 240);
    
}

void ofApp::makeDream()
{
    // blend last dream with cam
    ofPixels p1 = nightmare.getPixels();
    ofPixels p2 = cam.getPixels();
    ofPixels pix;
    pix.allocate(p1.getWidth(), p2.getHeight(), 3);
    for (int i=0; i<pix.size(); i++) {
        pix[i] = blendAmt * p1[i] + (1.0 - blendAmt) * p2[i];
    }
    
    // dream
    nightmare = darknet.nightmare( pix, max_layer, range, norm, 1, iters, octaves, rate, thresh );
}

void ofApp::update()
{
    cam.update();
    
    if (isDreaming) {
        makeDream();
    }
}

void ofApp::draw()
{
    cam.draw(300, 0);
    if (nightmare.isAllocated()){
        nightmare.draw(300, 260);
    }
    gui.draw();
}

void ofApp::startDream(bool & b) {
    if (b) {
        nightmare.setFromPixels(cam.getPixels());
        makeDream();
    }
}

void ofApp::keyPressed(int key) {
    
}
