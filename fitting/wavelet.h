/*
    This code is adapted from the code snippet provided by 
    the paper "Wavelet Noise" by Robert L. Cook Tony DeRose.
*/


#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdio>
#include <random>
#include <functional>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <chrono>

int Num1 = 128;
int Num2 = 128;
float *noiseTileData_r = new float[Num1 * Num2 * 50];
float *noiseTileData_i = new float[Num1 * Num2 * 50];
unsigned *permutationTable = new unsigned[Num1];

std::default_random_engine generator;

float gaussianNoise(float sigma)
{
    std::normal_distribution<float> distribution(0.f, sigma);
    return distribution(generator);
}

int Mod(int x, int n)
{
    int m = x % n;
    return (m < 0) ? m + n : m;
}
#define ARAD 16
void Downsample(float *from, float *to, int n, int stride)
{
    float *a, aCoeffs[2 * ARAD] = {
                  0.000334, -0.001528, 0.000410, 0.003545, -0.000938, -0.008233, 0.002172, 0.019120,
                  -0.005040, -0.044412, 0.011655, 0.103311, -0.025936, -0.243780, 0.033979, 0.655340,
                  0.655340, 0.033979, -0.243780, -0.025936, 0.103311, 0.011655, -0.044412, -0.005040,
                  0.019120, 0.002172, -0.008233, -0.000938, 0.003546, 0.000410, -0.001528, 0.000334};
    a = &aCoeffs[ARAD];
    for (int i = 0; i < n / 2; i++)
    {
        to[i * stride] = 0;
        for (int k = 2 * i - ARAD; k < 2 * i + ARAD; k++)
        {
            float tmp1 = a[k - 2 * i];
            float tmp2 = from[Mod(k, n) * stride];
            to[i * stride] += a[k - 2 * i] * from[Mod(k, n) * stride];
        }
    }
}
void Upsample(float *from, float *to, int n, int stride)
{
    float *p, pCoeffs[4] = {0.25, 0.75, 0.75, 0.25};
    p = &pCoeffs[2];
    for (int i = 0; i < n; i++)
    {
        to[i * stride] = 0;
        for (int k = i / 2; k <= i / 2 + 1; k++)
            to[i * stride] += p[i - 2 * k] * from[Mod(k, n / 2) * stride];
    }
}
void GenerateNoiseTile_r(int lambdanum, int thetanum, int phinum, int olap)
{
    int ix, iy, iz, i, sz = thetanum * phinum * lambdanum * sizeof(float);
    float *temp1 = (float *)malloc(sz), *temp2 = (float *)malloc(sz), *noise = (float *)malloc(sz);

    /* Step 1. Fill the tile with random numbers in the range -1 to 1. */
    float lambdastart = 400;
    float lambdaend = 700;
    for (i = 0; i < lambdanum; i++)
    {
        float lambda = lambdastart + (lambdaend - lambdastart) * i / lambdanum;
        float sigma = 1;
        for (int j = 0; j < thetanum; j++)
        {
            for (int k = 0; k < phinum; k++)
            {
                // sample from 2D Gaussian, and assign to r and i parts
                float value = gaussianNoise(sigma);
                noise[i * thetanum * phinum + j * phinum + k] = value;
            }
        }
    }

    for (i = 0; i < lambdanum * thetanum * phinum; i++)
    {
        temp1[i] = 0;
        temp2[i] = 0;
    }

    /* Steps 2 and 3. Downsample and upsample the tile */
    for (iy = 0; iy < thetanum; iy++)
        for (iz = 0; iz < lambdanum; iz++)
        { /* each x row */
            i = iy * phinum + iz * thetanum * phinum;
            Downsample(&noise[i], &temp1[i], phinum, 1);
            Upsample(&temp1[i], &temp2[i], phinum, 1);
        }

    for (ix = 0; ix < phinum; ix++)
        for (iz = 0; iz < lambdanum; iz++)
        { /* each y row */
            i = ix + iz * thetanum * phinum;
            Downsample(&temp2[i], &temp1[i], thetanum, phinum);
            Upsample(&temp1[i], &temp2[i], thetanum, phinum);
        }

    for (ix = 0; ix < phinum; ix++)
        for (iy = 0; iy < thetanum; iy++)
        { /* each z row */
            i = ix + iy * phinum;
            Downsample(&temp2[i], &temp1[i], lambdanum, thetanum * phinum);
            Upsample(&temp1[i], &temp2[i], lambdanum, thetanum * phinum);
        }

    /* Step 4. Subtract out the coarse-scale contribution */
    for (i = 0; i < lambdanum * thetanum * phinum; i++)
    {
        noise[i] -= temp2[i];
    }

    /* Avoid even/odd variance difference by adding odd-offset version of noise to itself.*/
    int offset1 = phinum / 2;
    if (offset1 % 2 == 0)
        offset1++;
    int offset2 = thetanum / 2;
    if (offset2 % 2 == 0)
        offset2++;
    int offset3 = lambdanum / 2;
    if (offset3 % 2 == 0)
        offset3++;

    for (i = 0, ix = 0; ix < phinum; ix++)
        for (iy = 0; iy < thetanum; iy++)
            for (iz = 0; iz < lambdanum; iz++)
                temp1[i++] = noise[Mod(ix + offset1, phinum) + Mod(iy + offset2, thetanum) * phinum + Mod(iz + offset3, lambdanum) * thetanum * phinum]; // variance?
    for (i = 0; i < lambdanum * thetanum * phinum; i++)
    {
        noise[i] += temp1[i];
    }

    for (i = 0; i < lambdanum * thetanum * phinum; i++)
    {
        noiseTileData_r[i] = noise[i];
    }

    free(temp1);
    free(temp2);
}

void GenerateNoiseTile_i(int lambdanum, int thetanum, int phinum, int olap)
{
    int ix, iy, iz, i, sz = thetanum * phinum * lambdanum * sizeof(float);
    float *temp1 = (float *)malloc(sz), *temp2 = (float *)malloc(sz), *noise = (float *)malloc(sz);

    /* Step 1. Fill the tile with random numbers in the range -1 to 1. */
    float lambdastart = 400;
    float lambdaend = 700;
    for (i = 0; i < lambdanum; i++)
    {
        float lambda = lambdastart + (lambdaend - lambdastart) * i / lambdanum;
        // float sigma = 1.0 * (1 + (lambda - lambdastart) / (lambdaend - lambdastart));
        float sigma = 1;
        // std::cout<<"sigma "<<sigma<<std::endl;
        for (int j = 0; j < thetanum; j++)
        {
            for (int k = 0; k < phinum; k++)
            {
                // sample from 2D Gaussian, and assign to r and i parts
                float value = gaussianNoise(sigma);
                noise[i * thetanum * phinum + j * phinum + k] = value;
            }
        }
    }

    for (i = 0; i < lambdanum * thetanum * phinum; i++)
    {
        temp1[i] = 0;
        temp2[i] = 0;
    }

    /* Steps 2 and 3. Downsample and upsample the tile */
    for (iy = 0; iy < thetanum; iy++)
        for (iz = 0; iz < lambdanum; iz++)
        { /* each x row */
            i = iy * phinum + iz * thetanum * phinum;
            Downsample(&noise[i], &temp1[i], phinum, 1);
            Upsample(&temp1[i], &temp2[i], phinum, 1);
        }

    for (ix = 0; ix < phinum; ix++)
        for (iz = 0; iz < lambdanum; iz++)
        { /* each y row */
            i = ix + iz * thetanum * phinum;
            Downsample(&temp2[i], &temp1[i], thetanum, phinum);
            Upsample(&temp1[i], &temp2[i], thetanum, phinum);
        }

    for (ix = 0; ix < phinum; ix++)
        for (iy = 0; iy < thetanum; iy++)
        { /* each z row */
            i = ix + iy * phinum;
            Downsample(&temp2[i], &temp1[i], lambdanum, thetanum * phinum);
            Upsample(&temp1[i], &temp2[i], lambdanum, thetanum * phinum);
        }

    /* Step 4. Subtract out the coarse-scale contribution */
    for (i = 0; i < lambdanum * thetanum * phinum; i++)
    {
        noise[i] -= temp2[i];
    }

    /* Avoid even/odd variance difference by adding odd-offset version of noise to itself.*/
    int offset1 = phinum / 2;
    if (offset1 % 2 == 0)
        offset1++;
    int offset2 = thetanum / 2;
    if (offset2 % 2 == 0)
        offset2++;
    int offset3 = lambdanum / 2;
    if (offset3 % 2 == 0)
        offset3++;

    for (i = 0, ix = 0; ix < phinum; ix++)
        for (iy = 0; iy < thetanum; iy++)
            for (iz = 0; iz < lambdanum; iz++)
                temp1[i++] = noise[Mod(ix + offset1, phinum) + Mod(iy + offset2, thetanum) * phinum + Mod(iz + offset3, lambdanum) * thetanum * phinum]; // variance?
    for (i = 0; i < lambdanum * thetanum * phinum; i++)
    {
        noise[i] += temp1[i];
    }

    for (i = 0; i < lambdanum * thetanum * phinum; i++)
    {
        noiseTileData_i[i] = noise[i];
    }

    free(temp1);
    free(temp2);
}

void WNoise(int lambdanum, int thetanum, int phinum, float p[3], float u,
            int wavelength, float &intensity_r, float &intensity_i, unsigned id)
{
    int i, f[3], c[3], mid[3]; /* f, c = filter, noise coeff indices */
    float w[3][3], t;
    unsigned totalnum = thetanum * phinum * lambdanum;
    /* Evaluate quadratic B-spline basis functions */
    for (i = 0; i < 3; i++)
    {
        mid[i] = std::ceil(p[i] - 0.5); // p range?
        t = mid[i] - (p[i] - 0.5);      // t?
        w[i][0] = t * t / 2;            // w formula
        w[i][2] = (1 - t) * (1 - t) / 2;
        w[i][1] = 1 - w[i][0] - w[i][2];
    }

    /* Evaluate noise by weighting noise coefficients by basis function values */
    unsigned idoffset = id % phinum;
    for (f[2] = -1; f[2] <= 1; f[2]++)
    {
        for (f[1] = -1; f[1] <= 1; f[1]++)
        {
            for (f[0] = -1; f[0] <= 1; f[0]++)
            {
                float weight = 1;
                for (int i = 0; i < 3; i++)
                {
                    if (i == 0)
                        c[i] = Mod(mid[i] + f[i], phinum);
                    if (i == 1)
                        c[i] = Mod(mid[i] + f[i], thetanum);
                    if (i == 2)
                        c[i] = Mod(mid[i] + f[i], lambdanum);
                    weight *= w[i][f[i] + 1];
                }
                // c[0] is phi index, c[1] is thetaindex, c[2] is lambda index

                unsigned index =
                    c[2] * thetanum * phinum + c[1] * phinum + c[0];

                intensity_r += weight * noiseTileData_r[index];
                intensity_i += weight * noiseTileData_i[index];
            }
        }
    }
}