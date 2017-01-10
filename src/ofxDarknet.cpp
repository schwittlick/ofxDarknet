#include "ofxDarknet.h"

ofxDarknet::ofxDarknet()
{
}

ofxDarknet::~ofxDarknet()
{
}

void ofxDarknet::init( std::string cfgfile, std::string weightfile, std::string datacfg, std::string nameslist )
{
    cuda_set_device(0);
	net = parse_network_cfg( cfgfile.c_str() );
	load_weights( &net, weightfile.c_str() );
	set_batch_network( &net, 1 );

	if( !datacfg.empty() )
	{
		options1 = read_data_cfg((char *) datacfg.c_str() );
	}
	if( !nameslist.empty() )
	{
		names = get_labels( option_find_str( options1, "names", nameslist.c_str() ) );
	}
}

std::vector< detected_object > ofxDarknet::yolo( ofPixels & pix, float threshold /*= 0.24f */ )
{
	int originalWidth = pix.getWidth();
	int originalHeight = pix.getHeight();
	ofPixels  pix2( pix );
	pix2.resize( net.w, net.h );
	image im = convert( pix2 );
	layer l = net.layers[ net.n - 1 ];

	box *boxes = ( box* ) calloc( l.w*l.h*l.n, sizeof( box ) );
	float **probs = ( float** ) calloc( l.w*l.h*l.n, sizeof( float * ) );
	for( int j = 0; j < l.w*l.h*l.n; ++j ) probs[ j ] = ( float* ) calloc( l.classes, sizeof( float * ) );

	float *X = im.data1;
	network_predict( net, X );
	get_region_boxes( l, 1, 1, threshold, probs, boxes, 0, 0 );
	do_nms_sort( boxes, probs, l.w*l.h*l.n, l.classes, 0.4 );
	//free_image( sized );
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

ofImage ofxDarknet::nightmate( ofPixels & pix, int max_layer, int range, int norm, int rounds, int iters, int octaves, float rate, float thresh )
{
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

std::vector< classification > ofxDarknet::classify( ofPixels & pix, int count )
{
	int *indexes = ( int* ) calloc( count, sizeof( int ) );

	if( pix.getWidth() != net.w && pix.getHeight() != net.h ) {
		pix.resize( net.w, net.h );
	}

	image im = convert( pix );

	float *predictions = network_predict( net, im.data1 );

	top_k( predictions, net.outputs, count, indexes );
	std::vector< classification > classifications;
	for( int i = 0; i < count; ++i ) {
		int index = indexes[ i ];
		classification c;
		c.label = names[ index ];
		c.probability = predictions[ index ];
		classifications.push_back( c );
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

void ofxDarknet::train_rnn( std::string textfile, std::string cfgfile )
{
	srand( time( 0 ) );
	unsigned char *text = 0;
	int *tokens = 0;
	size_t size;
	FILE *fp = fopen( textfile.c_str(), "rb" );

	fseek( fp, 0, SEEK_END );
	size = ftell( fp );
	fseek( fp, 0, SEEK_SET );

	text = ( unsigned char* ) calloc( size + 1, sizeof( char ) );
	fread( text, 1, size, fp );
	fclose( fp );

	//char *backup_directory = "/home/pjreddie/backup/";
	char *base = basecfg( cfgfile.c_str() );
	fprintf( stderr, "%s\n", base );
	float avg_loss = -1;
	network net = parse_network_cfg( cfgfile.c_str() );

	int inputs = get_network_input_size( net );
	fprintf( stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay );
	int batch = net.batch;
	int steps = net.time_steps;
	int i = ( *net.seen ) / net.batch;

	int streams = batch / steps;
	size_t *offsets = ( size_t* ) calloc( streams, sizeof( size_t ) );
	int j;
	for( j = 0; j < streams; ++j ) {
		offsets[ j ] = rand_size_t() % size;
	}

	clock_t time;
	while( get_current_batch( net ) < net.max_batches ) {
		i += 1;
		time = clock();
		float_pair p;
		p = get_rnn_data( text, offsets, inputs, size, streams, steps );

		float loss = train_network_datum( net, p.x, p.y ) / ( batch );
		free( p.x );
		free( p.y );
		if( avg_loss < 0 ) avg_loss = loss;
		avg_loss = avg_loss*.9 + loss*.1;

		int chars = get_current_batch( net )*batch;
		fprintf( stderr, "%d: %f, %f avg, %f rate, %lf seconds, %f epochs\n", i, loss, avg_loss, get_current_rate( net ), sec( clock() - time ), ( float ) chars / size );

		for( j = 0; j < streams; ++j ) {
			//printf("%d\n", j);
			if( rand() % 10 == 0 ) {
				//fprintf(stderr, "Reset\n");
				offsets[ j ] = rand_size_t() % size;
				reset_rnn_state( net, j );
			}
		}

		if( i % 1000 == 0 ) {
			char buff[ 256 ];
			sprintf( buff, "%s_%d.weights", base, i );
			save_weights( net, buff );
		}
		if( i % 10 == 0 ) {
			char buff[ 256 ];
			sprintf( buff, "%s.backup", base );
			save_weights( net, buff );
		}
	}
	char buff[ 256 ];
	sprintf( buff, "%s_final.weights", base );
	save_weights( net, buff );
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
