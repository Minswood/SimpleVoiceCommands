#ifndef WAVFILEREADER_H
#define WAVFILEREADER_H

#include <vector>
#include <complex>

class Spectrogram
{
public:
    Spectrogram();

    void GetSpectrogram(std::vector<double> waveform, std::vector<std::vector<double>>& spectrogram);
    std::vector<std::complex<double>> WindowHanning(int N);
    std::vector<double> dft(std::vector<double> X);

};

#endif // WAVFILEREADER_H
