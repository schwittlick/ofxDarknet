#ifndef PARSER_H
#define PARSER_H
#include "network.h"
#ifdef __cplusplus 
extern "C" {
#endif
	network parse_network_cfg( const char *filename );
	void load_weights( network *net, const char *filename );
	void save_weights( network net, char *filename );
#ifdef __cplusplus 
}
#endif
void save_network(network net, char *filename);

void save_weights_upto(network net, char *filename, int cutoff);
void save_weights_double(network net, char *filename);


void load_weights_upto(network *net, char *filename, int cutoff );

#endif
