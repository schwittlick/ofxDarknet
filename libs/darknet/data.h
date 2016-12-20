#ifndef DATA_H
#define DATA_H
#include <pthread.h>

#if defined(_MSC_VER) && _MSC_VER < 1900
	#define inline __inline
#endif

#include "matrix.h"
#include "list.h"
#include "image.h"
#include "tree.h"

static inline float distance_from_edge(int x, int max)
{
    int dx = (max/2) - x;
    if (dx < 0) dx = -dx;
    dx = (max/2) + 1 - dx;
    dx *= 2;
    float dist = (float)dx/max;
    if (dist > 1) dist = 1;
    return dist;
}

typedef struct data1 {
    int w, h;
    matrix X;
    matrix y;
    int shallow;
    int *num_boxes;
    box **boxes;
} data1;

typedef enum {
    CLASSIFICATION_DATA, DETECTION_DATA, CAPTCHA_DATA, REGION_DATA, IMAGE_DATA, COMPARE_DATA, WRITING_DATA, SWAG_DATA, TAG_DATA, OLD_CLASSIFICATION_DATA, STUDY_DATA, DET_DATA, SUPER_DATA
} data_type;

typedef struct load_args{
    int threads;
    char **paths;
    char *path;
    int n;
    int m;
    char **labels;
    int h;
    int w;
    int out_w;
    int out_h;
    int nh;
    int nw;
    int num_boxes;
    int min, max, size;
    int classes;
    int background;
    int scale;
    float jitter;
    float angle;
    float aspect;
    float saturation;
    float exposure;
    float hue;
	data1 *d;
    image *im;
    image *resized;
    data_type type;
    tree *hierarchy;
} load_args;

typedef struct{
    int id;
    float x,y,w,h;
    float left, right, top, bottom;
} box_label;

#ifdef __cplusplus 
extern "C" {
#endif

	void free_data( data1 d );

	pthread_t load_data( load_args args );

	pthread_t load_data_in_thread( load_args args );

	void print_letters( float *pred, int n );
	data1 load_data_captcha( char **paths, int n, int m, int k, int w, int h );
	data1 load_data_captcha_encode( char **paths, int n, int m, int w, int h );
	data1 load_data_old( char **paths, int n, int m, char **labels, int k, int w, int h );
	data1 load_data_detection( int n, char **paths, int m, int w, int h, int boxes, int classes, float jitter, float hue, float saturation, float exposure );
	data1 load_data_tag( char **paths, int n, int m, int k, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure );
	matrix load_image_augment_paths( char **paths, int n, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure );
	data1 load_data_super( char **paths, int n, int m, int w, int h, int scale );
	data1 load_data_augment( char **paths, int n, int m, char **labels, int k, tree *hierarchy, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure );
	data1 load_go( char *filename );

	box_label *read_boxes( char *filename, int *n );
	data1 load_cifar10_data( char *filename );
	data1 load_all_cifar10();

	data1 load_data_writing( char **paths, int n, int m, int w, int h, int out_w, int out_h );

	list1 *get_paths( char *filename );
	char **get_labels( char *filename );
	void get_random_batch( data1 d, int n, float *X, float *y );
	data1 get_data_part( data1 d, int part, int total );
	data1 get_random_data( data1 d, int num );
	void get_next_batch( data1 d, int n, int offset, float *X, float *y );
	data1 load_categorical_data_csv( char *filename, int target, int k );
	void normalize_data_rows( data1 d );
	void scale_data_rows( data1 d, float s );
	void translate_data_rows( data1 d, float s );
	void randomize_data( data1 d );
	data1 *split_data( data1 d, int part, int total );
	data1 concat_data( data1 d1, data1 d2 );
	data1 concat_datas( data1 *d, int n );
	void fill_truth( char *path, char **labels, int k, float *truth );

#ifdef __cplusplus 
}
#endif

#endif
