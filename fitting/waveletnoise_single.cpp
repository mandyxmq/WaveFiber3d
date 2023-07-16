#include "wavelet.h"
#include <sys/stat.h>
#include <string.h>

int main(int argc, char **argv)
{

    std::string noisedir = "noise";
    char noisedir_array[noisedir.length()];
    strcpy(noisedir_array, noisedir.c_str());
    mkdir(noisedir_array, 0777);

    std::string fiberdir = "noise/fiber1";
    char fiberdir_array[fiberdir.length()];
    strcpy(fiberdir_array, fiberdir.c_str());
    mkdir(fiberdir_array, 0777);

    int bstart = 0;
    int bend = 4;

    for (int i = bstart; i < bend; ++i)
    {
        std::string b = std::to_string(i);
        std::string banddir = "noise/fiber1/band" + b;
        char banddir_array[banddir.length()];
        strcpy(banddir_array, banddir.c_str());
        mkdir(banddir_array, 0777);
    }

    int thetanum = 128;
    int phinum = 128;
    int olap = 0;
    int instancenum = 100;
    int lambdanum = 50;
    float w[4] = {1, 1, 1, 1};

    float q[3];
    int id = 0; 
    float u = 0;
    int imageHeight = 400; // theta
    int imageWidth = 400;  // phi

    float *noiseMap0 = new float[imageHeight * imageWidth];
    for (int i = 0; i < imageHeight * imageWidth; ++i)
    {
        noiseMap0[i] = 0;
    }

    float *noiseMap1 = new float[imageHeight * imageWidth];
    for (int i = 0; i < imageHeight * imageWidth; ++i)
    {
        noiseMap1[i] = 0;
    }

    float *noiseMap2 = new float[imageHeight * imageWidth];
    for (int i = 0; i < imageHeight * imageWidth; ++i)
    {
        noiseMap2[i] = 0;
    }

    float *noiseMap3 = new float[imageHeight * imageWidth];
    for (int i = 0; i < imageHeight * imageWidth; ++i)
    {
        noiseMap3[i] = 0;
    }

    float thetastart = -90;
    float phistart = 0;

    float thetastep = 6;
    float phistep = 8;

    float thetarange = 5;
    float phirange = 5;

    float fixlambda = 0.5;

    for (int instance = 0; instance < instancenum; ++instance)
    {
        std::cout << "instance " << instance << std::endl;
        GenerateNoiseTile_r(lambdanum, thetanum, phinum, olap);
        GenerateNoiseTile_i(lambdanum, thetanum, phinum, olap);

            for (unsigned j = 0; j < imageHeight; ++j)
            {
                for (unsigned i = 0; i < imageWidth; ++i)
                {
                    float theta = -thetarange / 2.0 + (float)thetarange / imageHeight * j;
                    float phi = (float)phirange / imageWidth * i;

                    // compute half vector angle index
                    float thetaindex = (theta - thetastart) / thetastep;
                    float phiindex = (phi - phistart) / phistep;

                    float intensity = 0;
                    for (int b = bstart; b < bend; b++)
                    {
                        // color result
                        // x: phi, y: theta, z:lambda
                        float p[3] = {phiindex, thetaindex, fixlambda};
                        for (int i = 0; i <= 2; i++)
                        {
                            q[i] = 2 * p[i] * std::pow(2, b);
                        }
                        float curintensity_r = 0;
                        float curintensity_i = 0;
                        WNoise(lambdanum, thetanum, phinum, q, u, i, curintensity_r, curintensity_i,
                               id);

                        float curintensity = (curintensity_r * curintensity_r + curintensity_i * curintensity_i);
                        if (b == 0)
                        {
                            noiseMap0[j * imageWidth + i] = curintensity;
                        }
                        else if (b == 1)
                        {
                            noiseMap1[j * imageWidth + i] = curintensity;
                        }else if (b == 2)
                        {
                            noiseMap2[j * imageWidth + i] = curintensity;
                        }else
                        {
                            noiseMap3[j * imageWidth + i] = curintensity;
                        }
                    }
                }
            }

        std::string filename = "noise/fiber1/band0/wavelet" + std::to_string(instance) + ".binary";
        std::ofstream noise0(filename, std::ios::out | std::ios::binary);
        noise0.write((char *)&noiseMap0[0], sizeof(float) * imageHeight * imageWidth);

        filename = "noise/fiber1/band1/wavelet" + std::to_string(instance) + ".binary";
        std::ofstream noise1(filename, std::ios::out | std::ios::binary);
        noise1.write((char *)&noiseMap1[0], sizeof(float) * imageHeight * imageWidth);

        filename = "noise/fiber1/band2/wavelet" + std::to_string(instance) + ".binary";
        std::ofstream noise2(filename, std::ios::out | std::ios::binary);
        noise2.write((char *)&noiseMap2[0], sizeof(float) * imageHeight * imageWidth);

        filename = "noise/fiber1/band3/wavelet" + std::to_string(instance) + ".binary";
        std::ofstream noise3(filename, std::ios::out | std::ios::binary);
        noise3.write((char *)&noiseMap3[0], sizeof(float) * imageHeight * imageWidth);
    }
    std::cout << "finished all instances" << std::endl;

    delete[] noiseMap0;
    delete[] noiseMap1;
    delete[] noiseMap2;
    delete[] noiseMap3;
    return 0;
}
