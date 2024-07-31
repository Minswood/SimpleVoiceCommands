#include <iostream>
#include <vector>
#include <random>
#include "../Headers/spectrogram.h"

double RandomNumber(double Min, double Max)
{
    return ((double(rand()) / double(RAND_MAX)) * (Max - Min)) + Min;
}

int main()
{
    Spectrogram s;
    std::vector<double> testAudio;
    for (int i = 0; i < 16000; i++)
    {
        testAudio.push_back(RandomNumber(-1, 1));
    }

    std::vector<std::vector<double>> spectrogram;
    s.GetSpectrogram(testAudio, spectrogram);

    std::cout << "\nspectrogram shape: " << (int)spectrogram.size() << ", " << (int)spectrogram[0].size() << "\n"
              << std::endl;

    //     for(int i = 0; i < (int)spectrogram.size(); i++){
    //         for(int j = 0; j< (int)spectrogram[0].size(); j++)
    //         {
    //             std::cout << spectrogram[i][j] << std::endl;
    //         }
    //     }

    return 0;
}
