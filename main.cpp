//
//  main.cpp
//  DAI-questao-02-deck
//
//  Created by Cassiano Rabelo on oct/16.
//  Copyright © 2016 Cassiano Rabelo. All rights reserved.
//

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include "deck.hpp"

using namespace std;
using namespace cv;

static void help(char exe[])
{
    cout
    << "------------------------------------------------------------------------------" << endl
    << "DAI . DETECAO E ANALISE DE IMAGENS"                                             << endl
    << "2016/2o. SEMESTRE"                                                              << endl
    << "ALUNO: CASSIANO RABELO E SILVA"                                                 << endl
    << "QUESTAO #2 . DETECCAO E RASTREAMENTO DE ALVO"                                   << endl << endl
    << "Utilizacao:"                                                                    << endl
    << exe << " [--camera <camera_id> | --video <video>] --output <video>"              << endl
    << "------------------------------------------------------------------------------" << endl
    << "Utilizando OpenCV " << CV_VERSION << endl << endl;
}

/////////////////////////////////////
int main(int argc, char *argv[]) {
    
    if (argc == 1) {
        help(argv[0]);
        return -1;
    }
    
    string output;                      // recorded video
    string input;                       // input video
    int camera = 0;                     // system camera to use
    bool isInputVideo = false;          // read from camera or from disk?
    bool isInputCamera = false;         // read from camera or from disk?
    bool writeOutput = false;           // write output?
    VideoWriter outputVideo;
    
    for (int i = 1; i < argc; ++i)
    {
        if (string(argv[i]) == "--camera") {
            camera = atoi(argv[++i]);
            isInputCamera = true;
        } else if (string(argv[i]) == "--video") {
            input = argv[++i];
            isInputVideo = true;
        } else if (string(argv[i]) == "--output") {
            output = argv[++i];
            writeOutput = true;
        } else if (string(argv[i]) == "--help") {
            help(argv[0]);
            return -1;
        } else {
            cout << "Parametro desconhecido: " << argv[i] << endl;
            return -1;
        }
    }
    
    VideoCapture capture;
    Mat frame, frameGray;
    
    if (isInputVideo) {
        capture.open(input);
        CV_Assert(capture.isOpened());
    }
    else {
        capture.open(camera);
        CV_Assert(capture.isOpened());
    }
    
    // output resolution based on input
    Size S = Size((int) capture.get(CAP_PROP_FRAME_WIDTH),
                  (int) capture.get(CAP_PROP_FRAME_HEIGHT));
    
    if (writeOutput) {
        int FPS = 30;
        outputVideo = VideoWriter(output, CV_FOURCC('M','J','P','G'), FPS, S, true);
        
        if (!outputVideo.isOpened())
            cerr << "Nao foi possivel abrir o arquivo de video para escrita" << endl;
        
        cout
        << "Salvando frames no arquivo: " << output << " com as seguintes caracteristicas:" << endl
        << "Largura=" << S.width << endl
        << "Altura=" << S.height << endl
        << "FPS=" << FPS << endl
        << "CODEC: MJPG - Motion JPEG" << endl
        << "------------------------------------------------------------------------------" << endl
        << "PARA SAIR APERTE A TECLA 'ESC'" << endl;
    }
    
    gDeckPosition = vector<Point2f>(NUM_POSITIONS, Point2f(S.width/2, S.height/2)); // initialize position storage
    
    while(capture.grab()){
        capture.retrieve(frame);
        
        if (frame.empty())
            break;
        
        vector< vector< Point2f > > corners;    // store corner points
        vector< vector< Point2f > > rejected;   // store detected but rejected decks
        cvtColor(frame, frameGray, COLOR_BGR2GRAY);
        
        detectDecks(frame, frameGray, corners, rejected);
        drawDetectedDecks(frame, corners);
        
        if (gDebug)
        {
            display(frame, Point(80, S.height-30), Scalar(0), "Press \"d\" to turn OFF debug mode");
        }
        else
        {
            display(frame, Point(80, S.height-30), Scalar(0), "Press \"d\" to turn ON debug mode");
        }
        
        if (writeOutput)
            outputVideo.write(frame);
        
        imshow( "DAI . Questao #2 . Cassiano Rabelo", frame);

        char key = (char)waitKey(1); // 10ms/frame
        if(key == 27) break;
        
        switch (key)
        {
            case 'd':
                gDebug = !gDebug;
                cout << "debug=" << (gDebug?"ON":"OFF") << endl;
                break;
        }
    }
    
    outputVideo.release();
    return 0;
}
