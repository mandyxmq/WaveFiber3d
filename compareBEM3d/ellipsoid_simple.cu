#include <thrust/inner_product.h>
#include <thrust/execution_policy.h>
#include "thrust/host_vector.h"
#include "thrust/device_vector.h"
#include "thrust/functional.h"
#include "thrust/complex.h"
#include "thrust/inner_product.h"
#include "thrust/random.h"

#include <stdio.h>
#include <iostream>
#include <math.h>
#include <fstream>

#include <string.h>
#include <iomanip>

#include <cstdio>
#include <ctime>
#include <cstdlib>

#include <sys/time.h>

#include <sys/stat.h>
#include <sys/types.h>
#include <bits/stdc++.h>

typedef thrust::complex<double> comThr;
const comThr cunit(0, 1);
double c0 = 299792458;
double mu0 = 4e-7 * M_PI;
double eps0 = 8.8541878e-12;
double Z0 = sqrt(mu0 / eps0);

// We'll use a 3-tuple to store our 3d vector type
typedef thrust::tuple<double, double, double> Double3;
typedef thrust::tuple<comThr, comThr, comThr> comDouble3;
typedef thrust::tuple<comThr, comThr, comThr, comThr, comThr, comThr> comDouble6;

typedef thrust::host_vector<double> hostDvec;
typedef thrust::device_vector<double> deviceDvec;

typedef thrust::host_vector<comThr> hostCvec;
typedef thrust::device_vector<comThr> deviceCvec;

typedef thrust::host_vector<Double3> hostD3vec;
typedef thrust::device_vector<Double3> deviceD3vec;

typedef thrust::host_vector<comDouble3> hostC3vec;
typedef thrust::device_vector<comDouble3> deviceC3vec;

typedef thrust::host_vector<comDouble6> hostC6vec;
typedef thrust::device_vector<comDouble6> deviceC6vec;

/////////GPU Code///////

template <typename T>
__host__ __device__
    T
    minus(T a)
{
  T result;
  thrust::get<0>(result) = -thrust::get<0>(a);
  thrust::get<1>(result) = -thrust::get<1>(a);
  thrust::get<2>(result) = -thrust::get<2>(a);
  return result;
}

template <typename T>
__host__ __device__
    T
    add(T a, T b)
{
  T result;
  thrust::get<0>(result) = thrust::get<0>(a) + thrust::get<0>(b);
  thrust::get<1>(result) = thrust::get<1>(a) + thrust::get<1>(b);
  thrust::get<2>(result) = thrust::get<2>(a) + thrust::get<2>(b);
  return result;
}

template <typename T>
__host__ __device__
    T
    subtract(T a, T b)
{
  T result;
  thrust::get<0>(result) = thrust::get<0>(a) - thrust::get<0>(b);
  thrust::get<1>(result) = thrust::get<1>(a) - thrust::get<1>(b);
  thrust::get<2>(result) = thrust::get<2>(a) - thrust::get<2>(b);
  return result;
}

template <typename T, typename U>
__host__ __device__
    comDouble3
    multiply(T a, U factor)
{
  comDouble3 result;
  thrust::get<0>(result) = thrust::get<0>(a) * factor;
  thrust::get<1>(result) = thrust::get<1>(a) * factor;
  thrust::get<2>(result) = thrust::get<2>(a) * factor;
  return result;
}

__host__ __device__
    Double3
    multiply(Double3 a, double factor)
{
  Double3 result;
  thrust::get<0>(result) = thrust::get<0>(a) * factor;
  thrust::get<1>(result) = thrust::get<1>(a) * factor;
  thrust::get<2>(result) = thrust::get<2>(a) * factor;
  return result;
}

__host__ __device__ double vecnorm(comDouble3 a)
{
  return sqrt(thrust::norm(thrust::get<0>(a)) + thrust::norm(thrust::get<1>(a)) + thrust::norm(thrust::get<2>(a)));
}

__host__ __device__ double vecnorm(Double3 a)
{
  return sqrt(thrust::get<0>(a) * thrust::get<0>(a) + thrust::get<1>(a) * thrust::get<1>(a) + thrust::get<2>(a) * thrust::get<2>(a));
}

__host__ __device__
    Double3
    normalize(Double3 a)
{
  double norm = sqrt(thrust::get<0>(a) * thrust::get<0>(a) + thrust::get<1>(a) * thrust::get<1>(a) + thrust::get<2>(a) * thrust::get<2>(a));
  Double3 result;
  thrust::get<0>(result) = thrust::get<0>(a) / norm;
  thrust::get<1>(result) = thrust::get<1>(a) / norm;
  thrust::get<2>(result) = thrust::get<2>(a) / norm;
  return result;
}

template <typename T>
__host__ __device__
    T
    dotproduct(thrust::tuple<T, T, T> a, Double3 b)
{
  T result;
  result = thrust::get<0>(a) * thrust::get<0>(b) + thrust::get<1>(a) * thrust::get<1>(b) + thrust::get<2>(a) * thrust::get<2>(b);
  return result;
}

__host__ __device__
    Double3
    crossproduct(Double3 a, Double3 b)
{
  Double3 result;
  thrust::get<0>(result) = thrust::get<1>(a) * thrust::get<2>(b) - thrust::get<1>(b) * thrust::get<2>(a);
  thrust::get<1>(result) = thrust::get<2>(a) * thrust::get<0>(b) - thrust::get<2>(b) * thrust::get<0>(a);
  thrust::get<2>(result) = thrust::get<1>(b) * thrust::get<0>(a) - thrust::get<1>(a) * thrust::get<0>(b);
  return result;
}

__host__ __device__
    comDouble3
    crossproduct(Double3 a, comDouble3 b)
{
  comDouble3 result;
  thrust::get<0>(result) = thrust::get<1>(a) * thrust::get<2>(b) - thrust::get<1>(b) * thrust::get<2>(a);
  thrust::get<1>(result) = thrust::get<2>(a) * thrust::get<0>(b) - thrust::get<2>(b) * thrust::get<0>(a);
  thrust::get<2>(result) = thrust::get<1>(b) * thrust::get<0>(a) - thrust::get<1>(a) * thrust::get<0>(b);
  return result;
}

__host__ __device__
    comDouble3
    crossproduct(comDouble3 a, Double3 b)
{
  comDouble3 result;
  thrust::get<0>(result) = thrust::get<1>(a) * thrust::get<2>(b) - thrust::get<1>(b) * thrust::get<2>(a);
  thrust::get<1>(result) = thrust::get<2>(a) * thrust::get<0>(b) - thrust::get<2>(b) * thrust::get<0>(a);
  thrust::get<2>(result) = thrust::get<1>(b) * thrust::get<0>(a) - thrust::get<1>(a) * thrust::get<0>(b);
  return result;
}

std::ostream &operator<<(std::ostream &os, const comDouble6 a)
{
  os << thrust::get<0>(a) << " " << thrust::get<1>(a) << " " << thrust::get<2>(a) << " " << thrust::get<3>(a) << " " << thrust::get<4>(a) << " " << thrust::get<5>(a);
  return os;
}

__device__
    comThr
    cos_internal_angle(comThr n, double theta)
{
  return sqrt(1.0 - sin(theta) / n * sin(theta) / n);
}

__device__
    comThr
    rs(double theta, comThr n0, comThr nt)
{
  comThr p0 = n0 * cos_internal_angle(n0, theta);
  comThr pt = nt * cos_internal_angle(nt / n0, theta);
  return (p0 - pt) / (p0 + pt);
}

__device__
    comThr
    rp(double theta, comThr n0, comThr nt)
{
  comThr q0 = cos_internal_angle(n0, theta) / n0;
  comThr qt = cos_internal_angle(nt, theta) / nt;
  return (q0 - qt) / (q0 + qt);
}

struct add6 : public thrust::binary_function<comDouble6, comDouble6, comDouble6>
{
  __host__ __device__
      comDouble6
      operator()(comDouble6 a, comDouble6 b)
  {
    comDouble6 result;
    thrust::get<0>(result) = thrust::get<0>(a) + thrust::get<0>(b);
    thrust::get<1>(result) = thrust::get<1>(a) + thrust::get<1>(b);
    thrust::get<2>(result) = thrust::get<2>(a) + thrust::get<2>(b);
    thrust::get<3>(result) = thrust::get<3>(a) + thrust::get<3>(b);
    thrust::get<4>(result) = thrust::get<4>(a) + thrust::get<4>(b);
    thrust::get<5>(result) = thrust::get<5>(a) + thrust::get<5>(b);
    return result;
  }
};

struct current_green
{
  Double3 ob;
  const double k;
  current_green(Double3 _ob, double _k) : ob(_ob), k(_k) {}

  __device__
      comDouble6
      operator()(const Double3 &x, const comDouble6 &y) const
  {
    // x is source point position, y is current
    comDouble6 field;
    comThr cunit(0.f, 1.f);
    double xdiff = thrust::get<0>(x) - thrust::get<0>(ob);
    double ydiff = thrust::get<1>(x) - thrust::get<1>(ob);
    double zdiff = thrust::get<2>(x) - thrust::get<2>(ob);
    comThr dis = sqrt(xdiff * xdiff + ydiff * ydiff + zdiff * zdiff);
    comThr green = exp(-cunit * k * dis) / dis;
    thrust::get<0>(field) = thrust::get<0>(y) * green;
    thrust::get<1>(field) = thrust::get<1>(y) * green;
    thrust::get<2>(field) = thrust::get<2>(y) * green;
    thrust::get<3>(field) = thrust::get<3>(y) * green;
    thrust::get<4>(field) = thrust::get<4>(y) * green;
    thrust::get<5>(field) = thrust::get<5>(y) * green;

    return field;
  }
};

void current_times_green(Double3 ob, double k, deviceD3vec &sourcevec, deviceC6vec &currentvec, deviceC6vec &fieldvec)
{
  thrust::transform(sourcevec.begin(), sourcevec.end(), currentvec.begin(), fieldvec.begin(), current_green(ob, k));
}

struct field_area
{
  field_area() {}

  __device__
      comDouble6
      operator()(const comDouble6 &x, const double &y) const
  {
    comDouble6 field;
    thrust::get<0>(field) = thrust::get<0>(x) * y;
    thrust::get<1>(field) = thrust::get<1>(x) * y;
    thrust::get<2>(field) = thrust::get<2>(x) * y;
    thrust::get<3>(field) = thrust::get<3>(x) * y;
    thrust::get<4>(field) = thrust::get<4>(x) * y;
    thrust::get<5>(field) = thrust::get<5>(x) * y;
    return field;
  }
};

void field_times_area(deviceC6vec &X, deviceDvec &Y)
{
  // X are fields, Y is unitare
  thrust::transform(X.begin(), X.end(), Y.begin(), X.begin(), field_area());
}

// implement local tangent plane calculation that works on GPU
struct LocalTangentPlane
{
  const Double3 wavedir;
  const Double3 Edir;
  const Double3 Hdir;
  const comThr eta_in;
  const comThr eta_out;
  const double Z0;
  const double costhetai;
  const double sigma;
  const double k0;
  const char mode;
  LocalTangentPlane(Double3 _wavedir, Double3 _Edir, Double3 _Hdir, comThr _eta_in, comThr _eta_out, double _Z0, double _costhetai, double _sigma, double _k0, char mode_) : wavedir(_wavedir), Edir(_Edir), Hdir(_Hdir), eta_in(_eta_in), eta_out(_eta_out), Z0(_Z0), costhetai(_costhetai), sigma(_sigma), k0(_k0), mode(mode_) {}

  __device__
      comDouble6
      operator()(const Double3 &source, const Double3 &normal) const
  {
    comDouble6 current;
    comThr cunit(0.0, 1.0);

    double localphi, beamfactor, dot;
    Double3 reflect_plane_normal, Rdir;
    comThr factor, Rs, Rp, Ei0, Hi0;
    comDouble3 Ei, Hi, Ei_perp, Ei_par, Es_perp, Hi_perp, Hi_par, Hs_perp, Hs_par, Es_par, Es, Hs, Et, Ht, J, M;

    // compute dot product (used to decide illuminated or shadow)
    double dotvalue = dotproduct(wavedir, normal);
    if (dotvalue <= 0)
    {
      dot = dotproduct(wavedir, source);
      double tmp = thrust::get<2>(source) * costhetai;
      beamfactor = 1.0;
      Ei0 = beamfactor * exp(-cunit * k0 * dot);
      Hi0 = Ei0 / Z0;
      Ei = multiply(Edir, Ei0);
      Hi = multiply(Hdir, Hi0);

      float tol = 1e-15;
      if (std::abs(std::abs(dotvalue) - 1) < tol)
      {
        if (mode == 'M')
        {
          Rs = (eta_in - eta_out) / (eta_in + eta_out);
          Es = multiply(Ei, Rs);
          Hs = multiply(Hi, Rs);
        }
        else
        {
          Rp = (eta_out - eta_in) / (eta_in + eta_out);
          Es = multiply(Ei, Rp);
          Hs = multiply(Hi, Rp);
        }
      }
      else
      {
        reflect_plane_normal = normalize(crossproduct(wavedir, normal)); // perp
        factor = dotproduct(Ei, reflect_plane_normal);
        Ei_perp = multiply(reflect_plane_normal, factor);
        Ei_par = subtract(Ei, Ei_perp);
        Hi_par = multiply(crossproduct(wavedir, Ei_par), 1.0 / Z0);
        Hi_perp = subtract(Hi, Hi_par);

        localphi = acos(abs(dotvalue));
        Rs = rs(localphi, eta_in, eta_out);
        Rp = rp(localphi, eta_in, eta_out);
        Es = subtract(multiply(Ei_perp, Rs), multiply(Ei_par, Rp));
        Hs = subtract(multiply(Hi_par, Rp), multiply(Hi_perp, Rs));
      }
    }
    else
    {
      Es = minus(Ei);
      Hs = minus(Hi);
    }
    Et = add(Ei, Es);
    Ht = add(Hi, Hs);
    M = crossproduct(Et, normal);
    J = crossproduct(normal, Ht);
    thrust::get<0>(current) = thrust::get<0>(J);
    thrust::get<1>(current) = thrust::get<1>(J);
    thrust::get<2>(current) = thrust::get<2>(J);
    thrust::get<3>(current) = thrust::get<0>(M);
    thrust::get<4>(current) = thrust::get<1>(M);
    thrust::get<5>(current) = thrust::get<2>(M);

    return current;
  }
};

void computeLocalTangentPlane(deviceD3vec &source, deviceD3vec &normal, deviceC6vec &current, Double3 wavedir, Double3 Edir, Double3 Hdir, comThr eta_in, comThr eta_out, double Z0, double costhetai, double sigma, double k0, char mode)
{
  thrust::transform(source.begin(), source.end(), normal.begin(), current.begin(), LocalTangentPlane(wavedir, Edir, Hdir, eta_in, eta_out, Z0, costhetai, sigma, k0, mode));
}

int main(int argc, const char *argv[])
{
  float delta;
  struct timeval start, end;
  gettimeofday(&start, NULL);

  std::string outputdir = "output";
  char dir_array[outputdir.length()];
  strcpy(dir_array, outputdir.c_str());
  mkdir(dir_array, 0777);


  // set geometry to larger size than the ellipsoid
  float gaussiansigma = 10000 * 1e-6;

  float Dis = 1;
  int thetaonum = 1;
  int phionum = 3600;
  int obnum = thetaonum * phionum;

  int thetainum = 1;
  int phiinum = 1;

  std::cout << "thetainum " << thetainum << " phiinum " << phiinum << " thetaonum " << thetaonum << " phionum " << phionum << std::endl;

  // polarization
  char mode = 'M';

  int numquad = 30322;
  // read in points, areavec, normalvec
  std::cout << "read in surface height" << std::endl;
  std::string filename = "mesh/PO/ellipsoid_3_3_2_float.binary";
  std::ifstream myfile1(filename, std::ios::in | std::ios::binary);
  float *quadpoint = new float[numquad * 3];
  myfile1.read((char *)&quadpoint[0], 3 * numquad * sizeof(float));

  std::cout << "finish reading point" << std::endl;

  filename = "mesh/PO/ellipsoid_3_3_2_area_float.binary";
  std::ifstream myfile2(filename, std::ios::in | std::ios::binary);
  float *quadarea = new float[numquad];
  myfile2.read((char *)&quadarea[0], numquad * sizeof(float));

  std::cout << "finish reading area" << std::endl;

  filename = "mesh/PO/ellipsoid_3_3_2_normal_float.binary";
  std::ifstream myfile3(filename, std::ios::in | std::ios::binary);
  float *quadnormal = new float[numquad * 3];
  myfile3.read((char *)&quadnormal[0], 3 * numquad * sizeof(float));

  std::cout << "finish reading normal" << std::endl;

  // Initialize all the points
  hostD3vec sourcevec(numquad);
  hostDvec unitareavec(numquad);
  hostD3vec normalvec(numquad);




  for (int i = 0; i < numquad; ++i)
  {
    thrust::get<0>(sourcevec[i]) = quadpoint[i];
    thrust::get<1>(sourcevec[i]) = quadpoint[numquad + i];
    thrust::get<2>(sourcevec[i]) = quadpoint[2 * numquad + i];
    unitareavec[i] = quadarea[i];

    thrust::get<0>(normalvec[i]) = quadnormal[i];
    thrust::get<1>(normalvec[i]) = quadnormal[numquad + i];
    thrust::get<2>(normalvec[i]) = quadnormal[2 * numquad + i];
  }

    std::cout<<"sourcevec[0] "<<thrust::get<0>(sourcevec[0])<<" "<<thrust::get<1>(sourcevec[0])<<" "<<thrust::get<2>(sourcevec[0])<<std::endl;
  std::cout<<"unitareavec[0] "<<unitareavec[0]<<std::endl;
  std::cout<<"normalvec[0] "<<thrust::get<0>(normalvec[0])<<" "<<thrust::get<1>(normalvec[0])<<" "<<thrust::get<2>(normalvec[0])<<std::endl;

  deviceD3vec normalvec_dev = normalvec;
  deviceD3vec sourcevec_dev = sourcevec;
  deviceDvec unitareavec_dev = unitareavec;
  hostC6vec currentvec(numquad);
  deviceC6vec currentvec_dev(numquad);

  double *scattering = new double[thetaonum * phionum];
  double wavelength = 500;
  wavelength *= 1e-9;
  double k = 2 * M_PI / wavelength;
  double omega = k * c0;
  comThr Efactor = -cunit * omega * mu0 / (4 * M_PI);
  comThr Hfactor = -cunit * omega * eps0 / (4 * M_PI);

  double thetai = 0;
  double phii = 0;

  double kappa = 0.3;
  comThr eta = 1.55 + kappa * cunit;
  double krho = k * cos(thetai);
  double kz = k * sin(thetai);

  // E H direction
  Double3 wavedir(3), Edir(3), Hdir(3);
  thrust::get<0>(wavedir) = cos(thetai) * cos(phii);
  thrust::get<1>(wavedir) = cos(thetai) * sin(phii);
  thrust::get<2>(wavedir) = sin(thetai);
  std::cout << "wavedir " << thrust::get<0>(wavedir) << " " << thrust::get<1>(wavedir) << " " << thrust::get<2>(wavedir) << std::endl;
  if (mode == 'M')
  {
    thrust::get<0>(Edir) = -sin(thetai) * cos(phii);
    thrust::get<1>(Edir) = -sin(thetai) * sin(phii);
    thrust::get<2>(Edir) = cos(thetai);

    thrust::get<0>(Hdir) = sin(phii);
    thrust::get<1>(Hdir) = -cos(phii);
    thrust::get<2>(Hdir) = 0;
  }
  else
  {
    thrust::get<0>(Edir) = -sin(phii);
    thrust::get<1>(Edir) = cos(phii);
    thrust::get<2>(Edir) = 0;

    thrust::get<0>(Hdir) = -sin(thetai) * cos(phii);
    thrust::get<1>(Hdir) = -sin(thetai) * sin(phii);
    thrust::get<2>(Hdir) = cos(thetai);
  }

  computeLocalTangentPlane(sourcevec_dev, normalvec_dev, currentvec_dev, wavedir, Edir, Hdir, comThr(1.0, 0.0), eta, Z0, cos(thetai), gaussiansigma, k, mode);

  // check
  thrust::copy(currentvec_dev.begin(), currentvec_dev.end(), currentvec.begin());

  field_times_area(currentvec_dev, unitareavec_dev);

  cudaDeviceSynchronize();

  gettimeofday(&end, NULL);
  delta = ((end.tv_sec - start.tv_sec) * 1000000u +
           end.tv_usec - start.tv_usec) /
          1.e6;
  std::cout << "after fresnel, time used: " << delta << std::endl;

  deviceC6vec fieldvec_dev(numquad);
  hostC6vec fieldvec(numquad);

  double thetao = 0;
  Double3 ob, dir;
  double dirz = sin(thetao);
  thrust::get<2>(dir) = dirz;
  thrust::get<2>(ob) = Dis * dirz;

  for (int phioindex = 0; phioindex < phionum; ++phioindex)
  {
    // for (int phioindex = 0; phioindex < 5; ++phioindex){
    double phio = 2 * M_PI / phionum * phioindex;
    // double phio = M_PI/3;
    double dirx = cos(thetao) * cos(phio);
    double diry = cos(thetao) * sin(phio);
    thrust::get<0>(dir) = dirx;
    thrust::get<1>(dir) = diry;
    thrust::get<0>(ob) = Dis * dirx;
    thrust::get<1>(ob) = Dis * diry;

    current_times_green(ob, k, sourcevec_dev, currentvec_dev, fieldvec_dev);

    comDouble6 fieldsum = thrust::reduce(thrust::device, fieldvec_dev.begin(), fieldvec_dev.end(), comDouble6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0), add6());
    comThr Ex = Efactor * thrust::get<0>(fieldsum);
    comThr Ey = Efactor * thrust::get<1>(fieldsum);
    comThr Ez = Efactor * thrust::get<2>(fieldsum);
    comThr Hx = Hfactor * thrust::get<3>(fieldsum);
    comThr Hy = Hfactor * thrust::get<4>(fieldsum);
    comThr Hz = Hfactor * thrust::get<5>(fieldsum);

    // project E and H
    comThr dotvalue = Ex*dirx + Ey*diry + Ez*dirz;
    Ex = Ex - dotvalue*dirx;
    Ey = Ey - dotvalue*diry;
    Ez = Ez - dotvalue*dirz;
    dotvalue = Hx*dirx + Hy*diry + Hz*dirz;
    Hx = Hx - dotvalue*dirx;
    Hy = Hy - dotvalue*diry;
    Hz = Hz - dotvalue*dirz;

    Ex = Ex - (diry*Hz-dirz*Hy) * Z0;
    Ey = Ey - (-(dirx*Hz-dirz*Hx)) * Z0;
    Ez = Ez - (dirx*Hy-diry*Hx) * Z0;

    scattering[phioindex] = thrust::norm(Ex) + thrust::norm(Ey) + thrust::norm(Ez);
  }

  // save scattering distiriubtion to disk
  filename = outputdir + "/ellipsoid_TM_kappa0.3.binary";
  std::cout<<"filename "<<filename<<std::endl;
  std::ofstream out(filename, std::ios::out | std::ios::binary);
  out.write((char *)&scattering[0], sizeof(double) * obnum);

  gettimeofday(&end, NULL);
  delta = ((end.tv_sec - start.tv_sec) * 1000000u +
           end.tv_usec - start.tv_usec) /
          1.e6;
  std::cout << "time used: " << delta << std::endl;

  delete[] quadpoint;
  delete[] quadarea;
  delete[] quadnormal;
  delete[] scattering;
  return 0;
}
