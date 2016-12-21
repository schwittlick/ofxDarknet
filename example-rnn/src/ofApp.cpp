#include "ofApp.h"

void ofApp::setup() 
{
	std::string cfgfile = "cfg/rnn.cfg";
	std::string weightfile = "data/arts_arthistory_aesthetics.weights";
	darknet.init( cfgfile, weightfile );
}

void ofApp::update()
{

}

void ofApp::draw()
{
	ofDrawBitmapStringHighlight( "Input: " + seed_text, 20, 20 );

	int offset = 60;
	for( std::string s : generated_texts )
	{
		ofDrawBitmapStringHighlight( s, 20, offset );

		offset += 30;
	}
}

void ofApp::keyReleased( int key )
{
	if( key == OF_KEY_BACKSPACE ) {
		seed_text = seed_text.substr( 0, seed_text.size() - 1 );
	}
	else {
		char k = key;
		seed_text += k;
	}

	generated_texts.clear();
	for( int i = 0; i < 13; i++ )
	{
		std::string generated_text = darknet.rnn( 50, seed_text, 0.8 );
		generated_texts.push_back( generated_text );
	}
}