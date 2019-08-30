//
//  take_my_photo.cpp
//  project
//
//  Created by BO ZHANG on 10/29/18.
//  Copyright © 2018 BO ZHANG. All rights reserved.
//

#include<iostream>
#include<opencv2/core.hpp>
#include<opencv2/opencv.hpp>
#include<opencv2/highgui.hpp>
using namespace cv;
using namespace std;




int main()
{
    Mat img;
    int k;
    int i=0; //begin with number1
    string ImgName;
    string fileName;
    VideoCapture cap(0);
    if (!cap.isOpened())
        return 1;
    while (1) {
        cap >> img;
        GaussianBlur(img, img, Size(3, 3), 0);
        imshow("1", img);
        k = waitKey(100);
        if (k == 'p')//按s保存图片
            {
                i++;
                ImgName = format("my_photo/%d.jpg",i);
                fileName = format("my_photo/%d.pgm",i);
          //      cout << ImgName << endl;
                imshow("image", img);
                cout << ImgName << endl;
                imwrite(ImgName, img);
                ImgName.at(0)++;
                img.release();
                destroyWindow("image");
                
                }
        else if (k == 27)//Esc键
            break;
        }
    return 0;
}
