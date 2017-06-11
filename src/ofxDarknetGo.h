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
    
    ofxDarknetGo();
    
    void setup(string cfgfile, string weightfile);
    void setDrawPosition(int x, int y, int width, int height);
    void setMouseActive(bool active);
    void setNumRecommendations(int n);
    
    float * getBoard() {return board;}
    
    void getRecommendations();
    void drawBoard();
    void drawRecommendations(int x, int y);
    
    void makeMove(int row, int col);
    void flip_board(float *board);

    void mouseMoved(ofMouseEventArgs &evt);
    void mousePressed(ofMouseEventArgs &evt);
    void nextAuto();
    
    const char * alphanum = {"ABCDEFGHJKLMNOPQRST"};
protected:
    
    image make_empty_image(int w, int h, int c);
    void rotate_image_cw(image im, int times);
    image float_to_image(int w, int h, int c, float *data);
    void flip_image(image a);
    
    void string_to_board(char *s, float *board);
    void board_to_string(char *s, float *board);
    void print_board(float *board, int swap, int *indexes);
    
    int inverted;
    int noi;
    int nind;
    float *board;
    float *move;
    int color;
    vector<int> recommendations;
    vector<float> probabilities;
    ofRectangle box;
    int active;
};
