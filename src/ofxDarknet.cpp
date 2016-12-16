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
	//draw_detections( im, l.w*l.h*l.n, thresh, boxes, probs, names, alphabet, l.classes );
	//show_image( im, "predictions" );
	free_image( sized );
	free_image( im );
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

			left = ofMap( left, 0, net.w, 0, originalWidth );
			top = ofMap( top, 0, net.h, 0, originalHeight );
			right = ofMap( right, 0, net.w, 0, originalWidth );
			bot = ofMap( bot, 0, net.h, 0, originalHeight );

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

ofImage ofxDarknet::nightmate( ofPixels pix )
{
	char *cfg = "cfg/vgg-conv.cfg";
	char *weights = "vgg-conv.weights";

	//char *cfg = "cfg/jnet-conv.cfg";
	//char *weights = "jnet-conv.weights";

	char *input = "data/1469625954789.jpg";
	int max_layer = 7;

	int range = 1;
	int norm = 1;
	int rounds = 1;
	int iters = 10;
	int octaves = 4;
	float zoom = 1.0;
	float rate = 0.04;
	float thresh = 1.0;
	float rotate = 0;
	float momentum = 0.9;
	float lambda = 0.01;
	char *prefix = 0;
	int reconstruct = 0; //1
	int smooth_size = 1;

	network net = parse_network_cfg( cfg );
	load_weights( &net, weights );
	char *cfgbase = basecfg( cfg );
	char *imbase = basecfg( input );

	set_batch_network( &net, 1 );
	image im = load_image_color( input, 0, 0 );
	if( 0 ) {
		float scale = 1;
		if( im.w > 512 || im.h > 512 ) {
			if( im.w > im.h ) scale = 512.0 / im.w;
			else scale = 512.0 / im.h;
		}
		image resized = resize_image( im, scale*im.w, scale*im.h );
		free_image( im );
		im = resized;
	}

	float *features = 0;
	image update;
	if( reconstruct ) {
		resize_network( &net, im.w, im.h );

		int zz = 0;
		network_predict( net, im.data1 );
		image out_im = get_network_image( net );
		image crop = crop_image( out_im, zz, zz, out_im.w - 2 * zz, out_im.h - 2 * zz );
		//flip_image(crop);
		image f_im = resize_image( crop, out_im.w, out_im.h );
		free_image( crop );
		printf( "%d features\n", out_im.w*out_im.h*out_im.c );


		im = resize_image( im, im.w, im.h );
		f_im = resize_image( f_im, f_im.w, f_im.h );
		features = f_im.data1;

		int i;
		for( i = 0; i < 14 * 14 * 512; ++i ) {
			features[ i ] += rand_uniform( -.19, .19 );
		}

		free_image( im );
		im = make_random_image( im.w, im.h, im.c );
		update = make_image( im.w, im.h, im.c );

	}

	int e;
	int n;
	for( e = 0; e < rounds; ++e ) {
		fprintf( stderr, "Iteration: " );
		fflush( stderr );
		for( n = 0; n < iters; ++n ) {
			fprintf( stderr, "%d, ", n );
			fflush( stderr );
			if( reconstruct ) {
				reconstruct_picture( net, features, im, update, rate, momentum, lambda, smooth_size, 1 );
				//if ((n+1)%30 == 0) rate *= .5;
				show_image( im, "reconstruction" );
#ifdef OPENCV
				cvWaitKey( 10 );
#endif
			}
			else {
				int layer = max_layer + rand() % range - range / 2;
				int octave = rand() % octaves;
				optimize_picture( &net, im, layer, 1 / pow( 1.33333333, octave ), rate, thresh, norm );
			}
		}
		fprintf( stderr, "done\n" );
		if( 0 ) {
			image g = grayscale_image( im );
			free_image( im );
			im = g;
		}
		char buff[ 256 ];
		if( prefix ) {
			sprintf( buff, "%s/%s_%s_%d_%06d", prefix, imbase, cfgbase, max_layer, e );
		}
		else {
			sprintf( buff, "%s_%s_%d_%06d", imbase, cfgbase, max_layer, e );
		}
		printf( "%d %s\n", e, buff );
		save_image( im, buff );

		show_image(im, buff);
		//cvWaitKey(0);

		if( rotate ) {
			image rot = rotate_image( im, rotate );
			free_image( im );
			im = rot;
		}
		image crop = crop_image( im, im.w * ( 1. - zoom ) / 2., im.h * ( 1. - zoom ) / 2., im.w*zoom, im.h*zoom );
		image resized = resize_image( crop, im.w, im.h );
		free_image( im );
		free_image( crop );
		im = resized;
	}

	return ofImage();
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

std::string ofxDarknet::rnn(int num, std::string seed, float temp, int rseed )
{
	int inputs = get_network_input_size( net );

	std::vector<char> v( seed.length() + 1 );
	std::strcpy( &v[ 0 ], seed.c_str() );
	char* s = &v[ 0 ];

	int i, j;
	for( i = 0; i < net.n; ++i ) net.layers[ i ].temperature = temp;
	int c = 0;
	int len = seed.length();
	float *input = ( float* ) calloc( inputs, sizeof( float ) );

	std::string sampled_text;

	for( i = 0; i < len - 1; ++i ) {
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
	
	for( i = 0; i < num; ++i ) {
		input[ c ] = 1;
		float *out = network_predict( net, input );
		input[ c ] = 0;
		for( j = 0; j < inputs; ++j ) {
			if( out[ j ] < .0001 ) out[ j ] = 0;
		}
		c = sample_array( out, inputs );
		
		char _c = c;
		sampled_text += _c;
	}
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