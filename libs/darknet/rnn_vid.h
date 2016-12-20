#ifndef RNN_VID_H
#define RNN_VID_H

typedef struct {
	float *x;
	float *y;
} float_pair;

#ifdef __cplusplus 
extern "C" {
#endif

	void reconstruct_picture( network net, float *features, image recon, image update, float rate, float momentum, float lambda, int smooth_size, int iters );

#ifdef __cplusplus 
}
#endif

float_pair get_rnn_vid_data( network net, char **files, int n, int batch, int steps );
void train_vid_rnn( char *cfgfile, char *weightfile );
image save_reconstruction( network net, image *init, float *feat, char *name, int i );
void generate_vid_rnn( char *cfgfile, char *weightfile );
void run_vid_rnn( int argc, char **argv );

#endif