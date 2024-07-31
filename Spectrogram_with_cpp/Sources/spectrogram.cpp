#ifndef M_PI
#define M_PI (3.14159265358979323846264338327950288)
#endif
#include <iostream>
#include <complex>
#include <math.h>
#include <vector>
#include "../Headers/spectrogram.h"

Spectrogram::Spectrogram()
{
}

std::vector<std::complex<double>> Spectrogram::WindowHanning(int N)
{
    std::vector<std::complex<double>> output;
    double ND = static_cast<double>(N);
    for (int n = 0; n < N; n++)
    {
        double nD = static_cast<double>(n);
        double realPart = 0.5 - 0.5 * (cos((2 * M_PI * nD) / ND));
        output.push_back(std::complex<double>(realPart, 0.0));
    }
    return output;
}

std::vector<double> Spectrogram::dft(std::vector<double> X)
{
    int N = X.size();
    int L = N / 2 + 1;
    std::vector<std::complex<double>> windowed = WindowHanning(N);

    std::complex<double> Xk;

    std::vector<double> output;
    output.reserve(L);

    for (int k = 0; k < L; k++)
    {
        Xk = std::complex<double>(0.0, 0.0);
        for (int n = 0; n < N; n++)
        {
            double realPart = cos(((2 * M_PI) / N) * k * n);
            double imagPart = sin(((2 * M_PI) / N) * k * n);
            std::complex<double> w(realPart, -imagPart);
            Xk += X[n] * w * windowed[n];
        }
        output.push_back(fabs(Xk));
    }
    return output;
}

void Spectrogram::GetSpectrogram(std::vector<double> waveform, std::vector<std::vector<double>> &spectrogram)
{
    int windowLength = 256, windowStep = 128;
    spectrogram.clear();
    std::vector<double> dftStep;
    int wfSize = (int)waveform.size();

    for (int i = 0; i < wfSize; i += windowStep)
    {
        if ((i + windowLength) <= wfSize)
        {
            std::vector<double> step;
            for (int x = i; x < (i + windowLength); x++)
            {
                step.push_back(waveform[x]);
            }

            dftStep = dft(step);
            spectrogram.insert(spectrogram.end(), dftStep);
            step.clear();
            dftStep.clear();
        }
    }
}
