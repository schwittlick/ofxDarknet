#include "ofxDarknet.h"

ofxDarknet::ofxDarknet()
{
}

ofxDarknet::~ofxDarknet()
{
}

void ofxDarknet::init( std::string cfgfile, std::string weightfile, std::string datacfg, std::string nameslist )
{
	options1 = read_data_cfg( str2char( datacfg ) );
	name_list = option_find_str( options1, "names", str2char( nameslist ) );
	names = get_labels( name_list );

	alphabet = load_alphabet();
	net = parse_network_cfg( str2char( cfgfile ) );
	load_weights( &net, str2char( weightfile ) );
	set_batch_network( &net, 1 );
}

std::vector< detected_object > ofxDarknet::yolo( ofPixels & pix, float threshold /*= 0.24f */ )
{
	int originalWidth = pix.getWidth();
	int originalHeight = pix.getHeight();
	ofPixels  pix2( pix );
	pix2.resize( net.w, net.h );
	image im = convert( pix2 );
	image sized = resize_image( im, net.w, net.h );

	//im = resize_image( im, net.w, net.h );
	layer l = net.layers[ net.n - 1 ];

	box *boxes = ( box* ) calloc( l.w*l.h*l.n, sizeof( box ) );
	float **probs = ( float** ) calloc( l.w*l.h*l.n, sizeof( float * ) );
	for( int j = 0; j < l.w*l.h*l.n; ++j ) probs[ j ] = ( float* ) calloc( l.classes, sizeof( float * ) );

	float *X = im.data1;
	network_predict( net, X );
	get_region_boxes( l, 1, 1, threshold, probs, boxes, 0, 0 );
	do_nms_sort( boxes, probs, l.w*l.h*l.n, l.classes, 0.4 );
	free_image( sized );
	free_image( im );
	std::vector< detected_object > detections;

	int num = l.w*l.h*l.n;
	for( int i = 0; i < num; ++i ) {
		int class1 = max_index( probs[ i ], l.classes );
		float prob = probs[ i ][ class1 ];
		if( prob > threshold ) {

			int offset = class1 * 123457 % l.classes;
			float red = get_color( 2, offset, l.classes );
			float green = get_color( 1, offset, l.classes );
			float blue = get_color( 0, offset, l.classes );

			box b = boxes[ i ];

			int left = ( b.x - b.w / 2. )*im.w;
			int right = ( b.x + b.w / 2. )*im.w;
			int top = ( b.y - b.h / 2. )*im.h;
			int bot = ( b.y + b.h / 2. )*im.h;

			if( left < 0 ) left = 0;
			if( right > im.w - 1 ) right = im.w - 1;
			if( top < 0 ) top = 0;
			if( bot > im.h - 1 ) bot = im.h - 1;

			left = ofMap( left, 0, net.w, 0, originalWidth );
			top = ofMap( top, 0, net.h, 0, originalHeight );
			right = ofMap( right, 0, net.w, 0, originalWidth );
			bot = ofMap( bot, 0, net.h, 0, originalHeight );

			detected_object detection;
			detection.label = names[ class1 ];
			detection.probability = prob;
			detection.rect = ofRectangle( left, top, right - left, bot - top );
			detection.color = ofColor( red * 255, green * 255, blue * 255);

			detections.push_back( detection );
		}
	}
	return detections;
}

ofImage ofxDarknet::nightmate( ofPixels & pix )
{
	int max_layer = 13;
	int range = 3;
	int norm = 1;
	int rounds = 4;
	int iters = 20;
	int octaves = 4;
	float rate = 0.01;
	float thresh = 1.0;

	image im = convert( pix );

	for( int e = 0; e < rounds; ++e ) {
		fprintf( stderr, "Iteration: " );
		fflush( stderr );
		for( int n = 0; n < iters; ++n ) {
			fprintf( stderr, "%d, ", n );
			fflush( stderr );
			int layer = max_layer + rand() % range - range / 2;
			int octave = rand() % octaves;
			optimize_picture( &net, im, layer, 1 / pow( 1.33333333, octave ), rate, thresh, norm );
		}
	}
	return ofImage( convert( im ) );
}

std::vector< classification > ofxDarknet::classify( ofPixels & pix )
{
	int top = 5;
	if( top == 0 ) top = option_find_int( options1, "top", 1 );

	char **names = get_labels( name_list );
	int *indexes = ( int* ) calloc( top, sizeof( int ) );

	pix.resize( net.w, net.h );
	image im = convert( pix );
	//resize_network( &net, r.w, r.h );

	float *predictions = network_predict( net, im.data1 );
	//if( net.hierarchy ) hierarchy_predictions( predictions, net.outputs, net.hierarchy, 0 );
	top_k( predictions, net.outputs, top, indexes );
	std::vector< classification > classifications;
	for( int i = 0; i < top; ++i ) {
		int index = indexes[ i ];
		classification c;
		c.label = names[ index ];
		c.probability = predictions[ index ];
		classifications.push_back( c );
		//if( net.hierarchy ) printf( "%d, %s: %f, parent: %s \n", index, names[ index ], predictions[ index ], ( net.hierarchy->parent[ index ] >= 0 ) ? names[ net.hierarchy->parent[ index ] ] : "Root" );
	}
	free_image( im );
	return classifications;
}

std::string ofxDarknet::rnn(int num, std::string seed, float temp )
{
	int inputs = get_network_input_size( net );

	for( int i = 0; i < net.n; ++i )
	{
		net.layers[ i ].temperature = temp;
	}

	int c = 0;
	int len = seed.length();
	float *input = ( float* ) calloc( inputs, sizeof( float ) );

	std::string sampled_text;

	for( int i = 0; i < len - 1; ++i ) {
		c = seed[ i ];
		input[ c ] = 1;
		network_predict( net, input );
		input[ c ] = 0;
		
		char _c = c;
		sampled_text += _c;
		
	}
	if( len ) c = seed[ len - 1 ];
	
	char _c = c;
	sampled_text += _c;
	
	for( int i = 0; i < num; ++i ) {
		input[ c ] = 1;
		float *out = network_predict( net, input );
		input[ c ] = 0;
		for( int j = 0; j < inputs; ++j ) {
			if( out[ j ] < .0001 ) out[ j ] = 0;
		}
		c = sample_array( out, inputs );
		
		char _c = c;
		sampled_text += _c;
	}

	delete input;

	return sampled_text;
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

ofPixels ofxDarknet::convert( image & im )
{
	unsigned char *data = ( unsigned char* ) calloc( im.w*im.h*im.c, sizeof( char ) );
	int i, k;
	for( k = 0; k < im.c; ++k ) {
		for( i = 0; i < im.w*im.h; ++i ) {
			data[ i*im.c + k ] = ( unsigned char ) ( 255 * im.data1[ i + k*im.w*im.h ] );
		}
	}

	ofPixels pix;
	pix.setFromPixels( data, im.w, im.h, im.c );
	return pix;
}

char * ofxDarknet::str2char( std::string string )
{
	std::vector<char> v( string.length() + 1 );
	std::strcpy( &v[ 0 ], string.c_str() );
	char* ch = &v[ 0 ];
	return ch;
}
