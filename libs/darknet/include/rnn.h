#ifndef RNN_H
#define RNN_H

typedef struct {
	float *x;
	float *y;
} float_pair;

#ifdef __cplusplus 
extern "C" {
#endif
	char **read_tokens( char *filename, size_t *read );
	void print_symbol( int n, char **tokens );
	void reset_rnn_state( network net, int b );
	float_pair get_rnn_data( unsigned char *text, size_t *offsets, int characters, size_t len, int batch, int steps );
#ifdef __cplusplus 
}
#endif

int *read_tokenized_data( char *filename, size_t *read );

float_pair get_rnn_token_data( int *tokens, size_t *offsets, int characters, size_t len, int batch, int steps );


void train_char_rnn( char *cfgfile, char *weightfile, char *filename, int clear, int tokenized );

void test_char_rnn( char *cfgfile, char *weightfile, int num, char *seed, float temp, int rseed, char *token_file );
void test_tactic_rnn( char *cfgfile, char *weightfile, int num, float temp, int rseed, char *token_file );
void valid_tactic_rnn( char *cfgfile, char *weightfile, char *seed );
void valid_char_rnn( char *cfgfile, char *weightfile, char *seed );
void vec_char_rnn( char *cfgfile, char *weightfile, char *seed );
void run_char_rnn( int argc, char **argv );

#endif