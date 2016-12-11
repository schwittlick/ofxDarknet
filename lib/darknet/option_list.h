#ifndef OPTION_LIST_H
#define OPTION_LIST_H
#include "list.h"

typedef struct{
    char *key;
    char *val;
    int used;
} kvp;

#ifdef __cplusplus 
extern "C" {
#endif
	list1 *read_data_cfg( char *filename );
	char *option_find_str( list1 *l, char *key, char *def );
#ifdef __cplusplus 
}
#endif

int read_option(char *s, list1 *options);
void option_insert( list1 *l, char *key, char *val);
char *option_find( list1 *l, char *key);

int option_find_int( list1 *l, char *key, int def);
int option_find_int_quiet( list1 *l, char *key, int def);
float option_find_float( list1 *l, char *key, float def);
float option_find_float_quiet( list1 *l, char *key, float def);
void option_unused( list1 *l);

#endif
