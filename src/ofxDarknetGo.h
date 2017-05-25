#pragma once

#include "ofMain.h"
#include "ofxDarknet.h"

#include "network.h"
#include "utils.h"
#include "parser.h"
#include "option_list.h"
#include "blas.h"


class ofxDarknetGo : public ofxDarknet {
public:
    typedef struct {
        char **data;
        int n;
    } moves;
    
    void setup(string cfgfile, string weightfile);
    void setDrawPosition(int x, int y, int width, int height);
    
    void getRecommendations();
    void draw();
    
    void makeMove(int row, int col);

    void flip_board(float *board);

protected:
    image make_empty_image(int w, int h, int c);
    void rotate_image_cw(image im, int times);
    image float_to_image(int w, int h, int c, float *data);
    void flip_image(image a);
    
    void string_to_board(char *s, float *board);
    void board_to_string(char *s, float *board);
    void print_board(float *board, int swap, int *indexes);
    
    int inverted = 1;
    int noi = 1;
    static const int nind = 5;
    float *board;
    float *move;
    int color;
    vector<int> recommendations;
    ofRectangle box;
};
