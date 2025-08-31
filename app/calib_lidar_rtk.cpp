/*
 * @Author: ylh 
 * @Date: 2024-05-22 21:00:24 
 * @Last Modified by: ylh 2252512364@qq.com
 * @Last Modified time: 2025-08-31 18:16:54
 */

#include "hand_eye_calibrator.h"
#include <iostream>

int main(int argc, char** argv) {
    if(argc < 2) {
        std::cout << "Usage: calib_lidar_rtk <config_file>\n";
        std::cout << "poses (timestamp(s) x y z qx qy qz qw)\n";
        return -1;
    }
    std::string cfgPath = argv[1];
    HandEyeCalibrator::Ptr calib = std::make_shared<HandEyeCalibrator>();
    if(calib->initParams(cfgPath) != 0) return -1;
    calib->calibProcess();

    return 0;
}

