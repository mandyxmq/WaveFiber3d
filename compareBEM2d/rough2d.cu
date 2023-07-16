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
const comThr cunit(0,1);
double c0 = 299792458;
double mu0 = 4e-7 * M_PI;
double eps0 = 8.8541878e-12;
double Z0 = sqrt(mu0/eps0);

typedef std::vector<std::complex<double>> comvec;
typedef thrust::tuple<double,double> Double2;

struct DotDir
{
  const double dirx, diry;

  DotDir(double _dirx, double _diry) : dirx(_dirx), diry(_diry){}

  __host__ __device__
  double operator()(const Double2& x) const { 
    return thrust::get<0>(x) * dirx +    // x components
      thrust::get<1>(x) * diry;
  }
};

void dotdir_fast(double dirx, double diry, thrust::device_vector<double>& px, thrust::device_vector<double>& py, thrust::device_vector<double>& sourceobdot)
{
  // Y <- A * X + Y
  thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(px.begin(), py.begin())),
                    thrust::make_zip_iterator(thrust::make_tuple(px.end(), py.end())),
                    sourceobdot.begin(), DotDir(dirx, diry));
}



struct green
{
  const double krho;
  const double Dis;

  green(double _krho, double _Dis) : krho(_krho), Dis(_Dis){}

  __host__ __device__
  comThr operator()(const double& x) const {
    // x is dis
    comThr cunit(0.f, 1.f);
    return sqrt(2.0*cunit/(M_PI*krho*Dis))*exp(-cunit*krho*Dis)*exp(cunit*krho*x);
  }
};

void greenfunction(double krho, double Dis, thrust::device_vector<double>& X, thrust::device_vector<comThr>& Y)
{
  // x is dotproduct, y is zdiff
  // current <- hankel * current
  thrust::transform(X.begin(), X.end(), Y.begin(), green(krho, Dis));
}


struct current_green
{
  const double omega;
  const double scale;   // mu0 or eps0

  current_green(double _omega, double _scale) : omega(_omega), scale(_scale) {}

  __host__ __device__
  comThr operator()(const comThr& x, const comThr& y) const {
    // x is Green's function, y is current
    comThr cunit(0.f,1.f);
    return y * (-omega*scale/4.0*x); // missing integration unit 
  }
};

void current_times_green(double omega, double scale, thrust::device_vector<comThr>& X, thrust::device_vector<comThr>& Y, thrust::device_vector<comThr>& Z)
{
  // x is hankel, y is current
  // current <- hankel * current
  thrust::transform(X.begin(), X.end(), Y.begin(), Z.begin(), current_green(omega, scale));
}


struct field_area
{
  field_area(){}

  __host__ __device__
  comThr operator()(const comThr& x, const double& y) const {
    // x is hankel, y is current
    return x*y;
  }
};

void field_times_length(thrust::device_vector<comThr>& X, thrust::device_vector<double>& Y)
{//void field_times_area(thrust::device_vector<comThr>& X, thrust::device_vector<double>& Y, thrust::device_vector<comThr>& Z)
  // x is hankel, y is current
  // current <- hankel * current
  thrust::transform(X.begin(), X.end(), Y.begin(), X.begin(), field_area());
}

// complex vector manipulations
void output(comvec a){
  std::cout<<a[0]<<" "<<a[1]<<" "<<a[2]<<std::endl;
}

comvec multiply(comvec a, double b){
  comvec result(3);
  result[0] = a[0] * b;
  result[1] = a[1] * b;
  result[2] = a[2] * b;
  return result;
}

comvec multiply(comvec a, std::complex<double> b){
  comvec result(3);
  result[0] = a[0] * b;
  result[1] = a[1] * b;
  result[2] = a[2] * b;
  return result;
}

comvec add(comvec a, comvec b){
  comvec result(3);
  result[0] = a[0] + b[0];
  result[1] = a[1] + b[1];
  result[2] = a[2] + b[2];
  return result;
}

comvec minus(comvec a){
  comvec result(3);
  result[0] = -a[0];
  result[1] = -a[1];
  result[2] = -a[2];
  return result;
}

comvec subtract(comvec a, comvec b){
  comvec result(3);
  result[0] = a[0] - b[0];
  result[1] = a[1] - b[1];
  result[2] = a[2] - b[2];
  return result;
}

double norm(comvec a){
  return std::sqrt(std::norm(a[0]) + std::norm(a[1]) + std::norm(a[2]));
}


std::complex<double> dotproduct(comvec a, comvec b){
  std::complex<double> result = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  return result;
}

void crossproduct(comvec a, comvec b, comvec& result){
  result[0] = a[1] * b[2] - a[2] * b[1];
  result[1] = - (a[0] * b[2] - a[2] * b[0]);
  result[2] = a[0] * b[1] - a[1] * b[0];
}

void crossproduct(double* a, double* b, double* result){
  result[0] = a[1] * b[2] - a[2] * b[1];
  result[1] = - (a[0] * b[2] - a[2] * b[0]);
  result[2] = a[0] * b[1] - a[1] * b[0];
}

void crossproduct(comvec a, double* b, double* result){
  result[0] = a[1].real() * b[2] - a[2].real() * b[1];
  result[1] = - (a[0].real() * b[2] - a[2].real() * b[0]);
  result[2] = a[0].real() * b[1] - a[1].real() * b[0];
}

void crossproduct(double *a, comvec b, double* result){
  result[0] = a[1] * b[2].real() - a[2] * b[1].real();
  result[1] = - (a[0] * b[2].real() - a[2] * b[0].real());
  result[2] = a[0] * b[1].real() - a[1] * b[0].real();
}


comvec conjugate(comvec a){
  comvec result(3);
  result[0] = std::conj(a[0]);
  result[1] = std::conj(a[1]);
  result[2] = std::conj(a[2]);
  return result;
}

comvec realvector(comvec a){
  comvec result(3);
  result[0] = std::real(a[0]);
  result[1] = std::real(a[1]);
  result[2] = std::real(a[2]);
  return result;
}

double vectornorm(comvec a){
  double norm = std::sqrt(std::norm(a[0]) + std::norm(a[1]) + std::norm(a[2]));
  return norm;
}

std::complex<double> cos_internal_angle(std::complex<double>n, double theta){
  return sqrt(1.0 - sin(theta)/n*sin(theta)/n);
}

std::complex<double> rs(double theta, std::complex<double>n0, std::complex<double>nt){
  std::complex<double> p0 = n0*cos_internal_angle(n0,theta);
  std::complex<double> pt = nt*cos_internal_angle(nt/n0, theta);
  return (p0-pt)/(p0+pt);
}

std::complex<double> rp(double theta, std::complex<double> n0, std::complex<double> nt){
  std::complex<double> q0 = cos_internal_angle(n0,theta)/n0;
  std::complex<double> qt = cos_internal_angle(nt,theta)/nt;
  return (q0-qt)/(q0+qt);
}

void current_fresnel_oblique(
                             char mode, double eta_in,
                             std::complex<double> eta_out,
                             thrust::host_vector<comThr>& Jx,
                             thrust::host_vector<comThr>& Jy,
                             thrust::host_vector<comThr>& Jz,
                             thrust::host_vector<comThr>& Mx,
                             thrust::host_vector<comThr>& My,
                             thrust::host_vector<comThr>& Mz,
                             const thrust::host_vector<double>& sourcex,
                             const thrust::host_vector<double>& sourcey,
                             int numel, double theta,
                             double k, double phii)
{
  double dotvalue, localphi;
  double phiunit = 2 * M_PI / numel;
  comvec Ei(3), Hi(3),
    Ei_perp(3), Ei_par(3), Es_perp(3), Es_par(3),
    Hi_perp(3), Hi_par(3), Hs_perp(3), Hs_par(3),
    Es(3), Hs(3), Et(3), Ht(3),
    J(3), M(3);
  comvec incidentdir(3), normal(3), ellipsenormal(3), reflection_plane_normal(3), Rdir(3), p1(3), p2(3), diff(3);
  std::complex<double> R, Ei0, Hi0;
  //std::complex<double> Edir[3], Hdir[3];
  double Edir[3], Hdir[3];

  double x, y, nextx, nexty;
  comvec cylinderaxis={0, 0, 1};
  incidentdir[0] = cos(theta)*cos(phii);
  incidentdir[1] = cos(theta)*sin(phii);
  incidentdir[2] = sin(theta);

  if (mode == 'M'){
    Edir[0] = -sin(theta)*cos(phii);
    Edir[1] = -sin(theta)*sin(phii);
    Edir[2] = cos(theta);
  }
  else{
    Hdir[0] = -sin(theta)*cos(phii);
    Hdir[1] = -sin(theta)*sin(phii);
    Hdir[2] = cos(theta);
  }
  for (int i = 0; i < numel; ++i){
    x = sourcex[i];
    y = sourcey[i];
    if (i<numel-1){
      nextx = sourcex[i+1];
      nexty = sourcey[i+1];
    }else{
      nextx = sourcex[0];
      nexty = sourcey[0];
    }
    std::complex<double> tmp = k*(incidentdir[0]*x+incidentdir[1]*y); // z=0
    double phase = tmp.real();
    Ei0 = exp(-cunit*phase);
    Hi0 = exp(-cunit*phase) / Z0;

    if (mode == 'M'){
      Ei = {Edir[0] * Ei0, Edir[1] * Ei0, Edir[2] * Ei0};
      crossproduct(incidentdir, Edir, Hdir);
      Hi = {Hdir[0] * Hi0, Hdir[1] * Hi0, Hdir[2] * Hi0};
    }else{
      Hi = {Hdir[0] * Hi0, Hdir[1] * Hi0, Hdir[2] * Hi0};
      crossproduct(Hdir, incidentdir, Edir);
      Ei = {Edir[0] * Ei0, Edir[1] * Ei0, Edir[2] * Ei0};
    }

    diff[0] = nextx-x; diff[1] = nexty-y;
    normal[0] = diff[1]; normal[1] = -diff[0]; normal[2] = 0;

    normal = multiply(normal, 1.0/norm(normal));
    dotvalue = std::real(dotproduct(incidentdir, normal));

    if (dotvalue<=0){
      double tol = 1e-15;
      if (std::abs(std::abs(dotvalue)-1) < tol){
        R        = rs(0, eta_in, eta_out);
        Es       = multiply(Ei, R);
        Hs       = multiply(Hi, R);
      }else{
        crossproduct(incidentdir, normal, reflection_plane_normal);
        double norm = vectornorm(reflection_plane_normal);
        reflection_plane_normal = multiply(reflection_plane_normal, 1.0/norm);

        // project Ei onto reflection_plane_normal
        std::complex<double> factor = dotproduct(Ei, reflection_plane_normal);
        Ei_perp = multiply(reflection_plane_normal, factor);
        Ei_par  = subtract(Ei, Ei_perp);

        localphi = std::acos(std::abs(dotproduct(incidentdir,normal)));
        R        = rs(localphi, eta_in, eta_out);
        factor   = 2.0*dotproduct(incidentdir, normal);
        Rdir     = subtract(incidentdir, multiply(normal, factor));
        Es_perp  = multiply(Ei_perp, R);
        crossproduct(Rdir, Es_perp, Hs_perp);
        Hs_perp  = multiply(Hs_perp, 1.f/Z0);

        R        = rp(localphi, eta_in, eta_out);
        crossproduct(incidentdir, Ei_par, Hi_par);
        Hi_par   = multiply(Hi_par,  1.f/Z0);
        Hs_par   = multiply(Hi_par, R);

        crossproduct(Hs_par, Rdir, Es_par);
        Es_par   = multiply(Es_par, Z0);

        Es       = add(Es_perp, Es_par);
        Hs       = add(Hs_perp, Hs_par);

      }
    }else{
      Es       = minus(Ei);
      Hs       = minus(Hi);
    }

    Et = add(Es, Ei);
    Ht = add(Hs, Hi);
    crossproduct(normal, Et, M);
    M = minus(M);
    crossproduct(normal, Ht, J);

    Jx[i] = J[0];
    Jy[i] = J[1];
    Jz[i] = J[2];
    Mx[i] = M[0];
    My[i] = M[1];
    Mz[i] = M[2];
  }
}

void readgeometry(std::string xfile, std::string yfile, double *xvec, double *yvec){
  std::string line;
  std::ifstream myfile (xfile);
  int pos = 0;
  if (myfile.is_open())
    {
      while ( getline (myfile,line) )
        {
          xvec[pos++] = std::stod(line);
        }
      myfile.close();
    }
  else std::cout << "Unable to open x file";

  pos = 0;
  std::ifstream myfile2 (yfile);
  if (myfile2.is_open())
    {
      while ( getline (myfile2,line) )
        {
          yvec[pos++] = std::stod(line);
        }
      myfile2.close();
    }
  else std::cout << "Unable to open y file";
}

int main(int argc, const char * argv[]){

  struct timeval start, end;
  gettimeofday(&start, NULL);

  // input cases 0: circle, 1: ellipse
  int testcase = atoi(argv[1]);
  char mode = *(argv[2]);

  int numel;
  std::string xfile, yfile;
  std::string directory = "crosssection/";
  std::string outputdir = "output";
  double radius = 10 * 1e-6;
  std::string geo;
  if (testcase == 0){
    std::cout<<"test rough circle"<<std::endl;
    numel = 1000;
    xfile = directory + "circle_x.txt";
    yfile = directory + "circle_y.txt";
    geo = "circle";
  }else{
    std::cout<<"test rough ellipse"<<std::endl;
    numel = 2000;
    xfile = directory + "ellipse_new_x.txt";
    yfile = directory + "ellipse_new_y.txt";
    geo = "ellipse";
  }
  std::cout<<"polarization mode T"<<mode<<std::endl;

  double *xvec = new double[numel];
  double *yvec = new double[numel];

  std::cout<<"read in rough cross-section"<<std::endl;  
  std::cout<<"xfile "<<xfile<<" yfile "<<yfile<<std::endl;
  readgeometry(xfile, yfile, xvec, yvec);
  std::cout<<"finish reading file"<<std::endl; 

  char dir_array[outputdir.length()];
  strcpy(dir_array, outputdir.c_str());
  mkdir(dir_array, 0777); 

  int lambdaindexstart = atoi(argv[3]);
  int lambdaindexend = atoi(argv[4]);
  std::cout<<"labdastart "<<lambdaindexstart<<std::endl;
  std::cout<<"lambdaend "<<lambdaindexend<<std::endl;


  // wave parameters
  int lambdanum = 25;
  double wavelength_min = 400;
  double wavelength_max = 700;
  double delta;
  double Dis = 1;

  double theta = 0;
  double kappa = 0.1;
  comThr eta = 1.55 + kappa * cunit;
  int phionum = 3600;
  int phiinum = 1;
  for (int lambdaindex = lambdaindexstart; lambdaindex < lambdaindexend; ++lambdaindex){
    std::cout<<"lambdaindex "<<lambdaindex<<std::endl;
    double wavelength = wavelength_min + (wavelength_max-wavelength_min)/(double)lambdanum*lambdaindex;
    wavelength *= 1e-9;
    double k = 2 * M_PI / wavelength;
    double krho = k * cos(theta);
    double kz   = k * sin(theta);
    double omega = k * c0;
    double scattering[phionum*phiinum];
    for (int phiiindex = 0; phiiindex < phiinum; ++phiiindex){
      double phii = (double)phiiindex/(double)phiinum*2*M_PI + M_PI;
      // compute for all the points
      thrust::host_vector<double> sourcex( numel );
      thrust::host_vector<double> sourcey( numel );
      thrust::host_vector<double> unitlengthvec( numel );
      double nextx, nexty, curlength;
      
      for (int i = 0; i < numel; ++i){
        sourcex[i] = xvec[i];
        sourcey[i] = yvec[i];
        if (i<numel-1){
          nextx = xvec[i+1];
          nexty = yvec[i+1];
        }else{
          nextx = xvec[0];
          nexty = yvec[0];
        }
        curlength = std::sqrt((xvec[i] - nextx)*(xvec[i] - nextx)+(yvec[i] - nexty)*(yvec[i] - nexty));
        unitlengthvec[i] = curlength;
      }

      thrust::host_vector<comThr> Jx( numel );
      thrust::host_vector<comThr> Jy( numel );
      thrust::host_vector<comThr> Jz( numel );
      thrust::host_vector<comThr> Mx( numel );
      thrust::host_vector<comThr> My( numel );
      thrust::host_vector<comThr> Mz( numel );

      current_fresnel_oblique(mode, 1.0, eta, Jx, Jy, Jz, Mx, My, Mz, sourcex, sourcey, numel, theta, k, phii);

      // Copy host_vectors to device_vectors
      thrust::device_vector<double> sourcex_dev = sourcex;
      thrust::device_vector<double> sourcey_dev = sourcey;
      thrust::device_vector<comThr> Jx_dev = Jx;
      thrust::device_vector<comThr> Jy_dev = Jy;
      thrust::device_vector<comThr> Jz_dev = Jz;
      thrust::device_vector<comThr> Mx_dev = Mx;
      thrust::device_vector<comThr> My_dev = My;
      thrust::device_vector<comThr> Mz_dev = Mz;
      thrust::device_vector<double> vecdir(numel); // Storage for result of each dot product
      thrust::device_vector<comThr> greenvec(numel);
      thrust::device_vector<comThr> Ex_dev(numel),
        Ey_dev(numel),
        Ez_dev(numel),
        Hx_dev(numel),
        Hy_dev(numel),
        Hz_dev(numel);

      thrust::device_vector<double> unitlengthvec_dev = unitlengthvec;

      //loop through directions
      double dirx, diry, dirz;
      for (int phioindex = 0; phioindex < phionum; ++phioindex){
        double phi = (double)phioindex/(double)phionum*2*M_PI;
        dirx = cos(theta)*cos(phi);
        diry = cos(theta)*sin(phi);
        dirz = sin(theta);

        dotdir_fast(dirx, diry, sourcex_dev, sourcey_dev, vecdir); // dot source point and horrizontal dir
        double scale = mu0;
        greenfunction(krho, Dis, vecdir, greenvec);  // hankel variable in MATLAB

        current_times_green(omega, scale, greenvec, Jx_dev, Ex_dev);
        current_times_green(omega, scale, greenvec, Jy_dev, Ey_dev);
        current_times_green(omega, scale, greenvec, Jz_dev, Ez_dev);

        scale = eps0;
        current_times_green(omega, scale, greenvec, Mx_dev, Hx_dev);
        current_times_green(omega, scale, greenvec, My_dev, Hy_dev);
        current_times_green(omega, scale, greenvec, Mz_dev, Hz_dev);

        // multiply unitarea
        field_times_length(Ex_dev, unitlengthvec_dev);
        field_times_length(Ey_dev, unitlengthvec_dev);
        field_times_length(Ez_dev, unitlengthvec_dev);
        field_times_length(Hx_dev, unitlengthvec_dev);
        field_times_length(Hy_dev, unitlengthvec_dev);
        field_times_length(Hz_dev, unitlengthvec_dev);
  
        comThr Ex = thrust::reduce(thrust::device, Ex_dev.begin(), Ex_dev.end(), comThr(0.0f,0.0f), thrust::plus<comThr>());
        comThr Ey = thrust::reduce(thrust::device, Ey_dev.begin(), Ey_dev.end(), comThr(0.0f,0.0f), thrust::plus<comThr>());
        comThr Ez = thrust::reduce(thrust::device, Ez_dev.begin(), Ez_dev.end(), comThr(0.0f,0.0f), thrust::plus<comThr>());
        comThr Hx = thrust::reduce(thrust::device, Hx_dev.begin(), Hx_dev.end(), comThr(0.0f,0.0f), thrust::plus<comThr>());
        comThr Hy = thrust::reduce(thrust::device, Hy_dev.begin(), Hy_dev.end(), comThr(0.0f,0.0f), thrust::plus<comThr>());
        comThr Hz = thrust::reduce(thrust::device, Hz_dev.begin(), Hz_dev.end(), comThr(0.0f,0.0f), thrust::plus<comThr>());

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
        Hx = (diry*Ez-dirz*Ey) / Z0;
        Hy = (-(dirx*Ez-dirz*Ex)) / Z0;
        Hz = (dirx*Ey-diry*Ex) / Z0;

        double magnitudesqr = thrust::norm(Ex)+thrust::norm(Ey)+thrust::norm(Ez);
        scattering[phiiindex*phionum+phioindex] = magnitudesqr * Dis / (2*radius);
      }
    }
    // save scattering distiriubtion to disk   
    std::string filename = outputdir+"/"+geo+"_T"+mode+"_"+std::to_string(lambdaindex)+".binary";
    std::cout<<"filename "<<filename<<std::endl;
    std::ofstream out(filename, std::ios::out|std::ios::binary);
    out.write((char *) &scattering[0], sizeof(double)*phiinum*phionum);

    gettimeofday(&end, NULL);
    delta = ((end.tv_sec  - start.tv_sec) * 1000000u +
             end.tv_usec - start.tv_usec) / 1.e6;
    std::cout<<"wavelength index "<<lambdaindex<<" time used: "<<delta<<std::endl;
  }

  delete[] xvec;
  delete[] yvec;
  return 0;
}
