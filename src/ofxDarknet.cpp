#include "ofxDarknet.h"

ofxDarknet::ofxDarknet()
{
}

ofxDarknet::~ofxDarknet()
{
}

void ofxDarknet::init( char *datacfg /*= "cfg/coco.data"*/, char *cfgfile /*= "cfg/yolo.cfg"*/, char *weightfile /*= "yolo.weights"*/, char *nameslist /*= "data/names.list" */ )
{
	options1 = read_data_cfg( datacfg );
	name_list = option_find_str( options1, "names", nameslist );
	names = get_labels( name_list );

	alphabet = load_alphabet();
	net = parse_network_cfg( cfgfile );
	if( weightfile ) {
		load_weights( &net, weightfile );
	}
	set_batch_network( &net, 1 );
}

std::vector< detected_object > ofxDarknet::detect( ofPixels & pix, float threshold /*= 0.24f */ )
{
	image im = convert( pix );

	im = resize_image( im, net.w, net.h );
	layer l = net.layers[ net.n - 1 ];

	box *boxes = ( box* ) calloc( l.w*l.h*l.n, sizeof( box ) );
	float **probs = ( float** ) calloc( l.w*l.h*l.n, sizeof( float * ) );
	for( int j = 0; j < l.w*l.h*l.n; ++j ) probs[ j ] = ( float* ) calloc( l.classes, sizeof( float * ) );

	float *X = im.data1;
	network_predict( net, X );
	get_region_boxes( l, 1, 1, threshold, probs, boxes, 0, 0 );
	do_nms_sort( boxes, probs, l.w*l.h*l.n, l.classes, 0.4 );
	//draw_detections( im, l.w*l.h*l.n, thresh, boxes, probs, names, alphabet, l.classes );
	//show_image( im, "predictions" );

	std::vector< detected_object > detections;

	int num = l.w*l.h*l.n;

	for( int i = 0; i < num; ++i ) {
		int class1 = max_index( probs[ i ], l.classes );
		float prob = probs[ i ][ class1 ];
		if( prob > threshold ) {

			int width = im.h * .012;

			if( 0 ) {
				width = pow( prob, 1. / 2. ) * 10 + 1;
				alphabet = 0;
			}

			printf( "%s: %.0f%%\n", names[ class1 ], prob * 100 );
			int offset = class1 * 123457 % l.classes;
			float red = get_color( 2, offset, l.classes );
			float green = get_color( 1, offset, l.classes );
			float blue = get_color( 0, offset, l.classes );
			float rgb[ 3 ];

			//width = prob*20+2;
			box b = boxes[ i ];

			int left = ( b.x - b.w / 2. )*im.w;
			int right = ( b.x + b.w / 2. )*im.w;
			int top = ( b.y - b.h / 2. )*im.h;
			int bot = ( b.y + b.h / 2. )*im.h;

			if( left < 0 ) left = 0;
			if( right > im.w - 1 ) right = im.w - 1;
			if( top < 0 ) top = 0;
			if( bot > im.h - 1 ) bot = im.h - 1;

			left = ofMap(left, 0, net.w, 0, pix.getWidth());
			top = ofMap(top, 0, net.h, 0, pix.getHeight() );
			right = ofMap(right, 0, net.w, 0, pix.getWidth() );
			bot = ofMap( bot, 0, net.h, 0, pix.getHeight() );

			rgb[ 0 ] = red;
			rgb[ 1 ] = green;
			rgb[ 2 ] = blue;

			detected_object detection;
			detection.label = names[ class1 ];
			detection.probability = prob;
			detection.rect = ofRectangle( left, top, right - left, bot - top );

			detections.push_back( detection );
		}
	}
	return detections;
}

image ofxDarknet::convert( ofPixels & pix )
{
	unsigned char *data = ( unsigned char * ) pix.getData();
	int h = pix.getHeight();
	int w = pix.getWidth();
	int c = pix.getNumChannels();
	int step = w * c;
	image im = make_image( w, h, c );
	int i, j, k, count = 0;;

	for( k = 0; k < c; ++k ) {
		for( i = 0; i < h; ++i ) {
			for( j = 0; j < w; ++j ) {
				im.data1[ count++ ] = data[ i*step + j*c + k ] / 255.;
			}
		}
	}
	return im;
}