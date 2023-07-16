#include <thrust/random.h>
#include <cooperative_groups.h>
#include <thrust/inner_product.h>
#include <thrust/execution_policy.h>
#include "thrust/host_vector.h"
#include "thrust/device_vector.h"
#include "thrust/functional.h"
#include "thrust/complex.h"
#include "thrust/inner_product.h"
#include "helper_cuda.h"


typedef thrust::complex<float> comThr;
typedef thrust::complex<double> comThrD;
const comThr cunit(0.0,1.0);
float c0 = 299792458;
float mu0 = 4e-7 * M_PI;
float eps0 = 8.8541878e-12;
float Z0 = sqrt(mu0/eps0);

// We'll use a 3-tuple to store our 3d vector type
typedef thrust::tuple<float,float,float> Float3;
typedef thrust::tuple<comThr, comThr, comThr> comFloat3;
typedef thrust::tuple<float,float,float,float,float,float> Float6;
typedef thrust::tuple<comThr, comThr, comThr, comThr, comThr, comThr> comFloat6;

typedef thrust::host_vector<float> hostFvec;
typedef thrust::device_vector<float> deviceFvec;

typedef thrust::host_vector<comThr> hostCvec;
typedef thrust::device_vector<comThr> deviceCvec;

typedef thrust::device_vector<Float3> deviceF3vec;

typedef thrust::host_vector<comFloat3> hostC3vec;
typedef thrust::device_vector<comFloat3> deviceC3vec;

typedef thrust::host_vector<comFloat6> hostC6vec;
typedef thrust::device_vector<comFloat6> deviceC6vec;

typedef thrust::host_vector<float2> hostf2vec;
typedef thrust::device_vector<float2> devicef2vec;
typedef thrust::tuple<float2, float2, float2, float2, float2, float2> float6;
typedef thrust::host_vector<float6> hostf6vec;

float2 zerofloat2=make_cuFloatComplex(0.0, 0.0);


/////////FOR FFT///////
class Scale_by_constant
{
private:
  float c_;

public:
  Scale_by_constant(float c) { c_ = c; };

  __host__ __device__ float2 operator()(float2 &a) const
  {
    float2 output;

    output.x = a.x / c_;
    output.y = a.y / c_;

    return output;
  }

};

/////////BASIC OPERATIONS///////
__host__ __device__
float2 operator*(const float2 &a, const float &b){
  float2 result;
  result.x = a.x*b;
  result.y = a.y*b;
  return result;
}

__host__ __device__
float2 operator*(const float2 &a, const float2 &b){
  float2 result;
  result.x = a.x*b.x-a.y*b.y;
  result.y = a.x*b.y+a.y*b.x;
  return result;
}


__host__ __device__
float2 operator/(const float2 &a, const float &b){
  float2 result;
  result.x = a.x/b;
  result.y = a.y/b;
  return result;
}


__host__ __device__
float2 operator*(const float2 &a, const comThr &b){
  float2 result;
  result.x = a.x*b.real()-a.y*b.imag();
  result.y = a.x*b.imag()+a.y*b.real();
  return result;
}

__host__ __device__
float2 operator-(const float2 &a, const float2 &b){
  float2 result;
  result.x = a.x - b.x;
  result.y = a.y - b.y;
  return result;
}

__host__ __device__
float2 operator-(const float2 &a){
  float2 result;
  result.x = -a.x;
  result.y = -a.y;
  return result;
}


__host__ __device__
float2 operator+(const float2 &a, const float2 &b){
  float2 result;
  result.x = a.x + b.x;
  result.y = a.y + b.y;
  return result;
}


std::ostream& operator<<(std::ostream& os, const float2 a){
  os <<"("<<a.x<<", "<<a.y<<") ";
  return os;
}

__host__ __device__
float norm(float2 a){
  float result;
  result = a.x*a.x+a.y*a.y;
  return result;
}

__host__ __device__
float6 operator+(const float6 &a, const float6 &b){
  float6 result;
  thrust::get<0>(result) = thrust::get<0>(a) + thrust::get<0>(b);
  thrust::get<1>(result) = thrust::get<1>(a) + thrust::get<1>(b);
  thrust::get<2>(result) = thrust::get<2>(a) + thrust::get<2>(b);
  thrust::get<3>(result) = thrust::get<3>(a) + thrust::get<3>(b);
  thrust::get<4>(result) = thrust::get<4>(a) + thrust::get<4>(b);
  thrust::get<5>(result) = thrust::get<5>(a) + thrust::get<5>(b);
  return result;
}

__host__ __device__
float6 operator*(const float6 &a, const comThr &b){
  float6 result;
  thrust::get<0>(result) = thrust::get<0>(a) * b;
  thrust::get<1>(result) = thrust::get<1>(a) * b;
  thrust::get<2>(result) = thrust::get<2>(a) * b;
  thrust::get<3>(result) = thrust::get<3>(a) * b;
  thrust::get<4>(result) = thrust::get<4>(a) * b;
  thrust::get<5>(result) = thrust::get<5>(a) * b;
  return result;
}

__host__ __device__
float6 operator*(const float6 &a, const float2 &b){
  float6 result;
  thrust::get<0>(result) = thrust::get<0>(a) * b;
  thrust::get<1>(result) = thrust::get<1>(a) * b;
  thrust::get<2>(result) = thrust::get<2>(a) * b;
  thrust::get<3>(result) = thrust::get<3>(a) * b;
  thrust::get<4>(result) = thrust::get<4>(a) * b;
  thrust::get<5>(result) = thrust::get<5>(a) * b;
  return result;
}


std::ostream& operator<<(std::ostream& os, const float6 a){
  os <<thrust::get<0>(a)<<" "<<thrust::get<1>(a)<<" "<<thrust::get<2>(a)<<" "
      <<thrust::get<3>(a)<<" "<<thrust::get<4>(a)<<" "<<thrust::get<5>(a);
  return os;
}



std::ostream& operator<<(std::ostream& os, const Float3 a){
  os <<thrust::get<0>(a)<<" "<<thrust::get<1>(a)<<" "<<thrust::get<2>(a);
  return os;
}



__host__ __device__
float3 operator-(const float3 &a, const float3 &b){
  float3 result;
  result.x = a.x - b.x;
  result.y = a.y - b.y;
  result.z = a.z - b.z;
  return result;
}

std::ostream& operator<<(std::ostream& os, const float3 a){
  os <<a.x<<" "<<a.y<<" "<<a.z;
  return os;
}


__host__ __device__
comFloat6 operator+(comFloat6 a, comFloat6 b){
  comFloat6 result;
  thrust::get<0>(result) = thrust::get<0>(a) + thrust::get<0>(b);
  thrust::get<1>(result) = thrust::get<1>(a) + thrust::get<1>(b);
  thrust::get<2>(result) = thrust::get<2>(a) + thrust::get<2>(b);
  thrust::get<3>(result) = thrust::get<3>(a) + thrust::get<3>(b);
  thrust::get<4>(result) = thrust::get<4>(a) + thrust::get<4>(b);
  thrust::get<5>(result) = thrust::get<5>(a) + thrust::get<5>(b);
  return result;
}

__host__ __device__
comFloat6 operator*(comFloat6 a, comThr b){
  comFloat6 result;
  thrust::get<0>(result) = thrust::get<0>(a) * b;
  thrust::get<1>(result) = thrust::get<1>(a) * b;
  thrust::get<2>(result) = thrust::get<2>(a) * b;
  thrust::get<3>(result) = thrust::get<3>(a) * b;
  thrust::get<4>(result) = thrust::get<4>(a) * b;
  thrust::get<5>(result) = thrust::get<5>(a) * b;
  return result;
}

std::ostream& operator<<(std::ostream& os, const comFloat6 a){
  os <<thrust::get<0>(a)<<" "<<thrust::get<1>(a)<<" "<<thrust::get<2>(a)<<" "
      <<thrust::get<3>(a)<<" "<<thrust::get<4>(a)<<" "<<thrust::get<5>(a);
  return os;
}

void multiply_factor(float6 &fieldsum, const comThr &Efactor, const comThr &Hfactor){
  thrust::get<0>(fieldsum) = thrust::get<0>(fieldsum)*Efactor;
  thrust::get<1>(fieldsum) = thrust::get<1>(fieldsum)*Efactor;
  thrust::get<2>(fieldsum) = thrust::get<2>(fieldsum)*Efactor;
  thrust::get<3>(fieldsum) = thrust::get<3>(fieldsum)*Hfactor;
  thrust::get<4>(fieldsum) = thrust::get<4>(fieldsum)*Hfactor;
  thrust::get<5>(fieldsum) = thrust::get<5>(fieldsum)*Hfactor;
}

template <typename T>
__host__ __device__
T minus(T a){
  T result;
  thrust::get<0>(result) = -thrust::get<0>(a);
  thrust::get<1>(result) = -thrust::get<1>(a);
  thrust::get<2>(result) = -thrust::get<2>(a);
  return result;
}


template <typename T>
__host__ __device__
T add(T a, T b){
  T result;
  thrust::get<0>(result) = thrust::get<0>(a) + thrust::get<0>(b);
  thrust::get<1>(result) = thrust::get<1>(a) + thrust::get<1>(b);
  thrust::get<2>(result) = thrust::get<2>(a) + thrust::get<2>(b);
  return result;
}

template <typename T>
__host__ __device__
T subtract(T a, T b){
  T result;
  thrust::get<0>(result) = thrust::get<0>(a) - thrust::get<0>(b);
  thrust::get<1>(result) = thrust::get<1>(a) - thrust::get<1>(b);
  thrust::get<2>(result) = thrust::get<2>(a) - thrust::get<2>(b);
  return result;
}

template <typename T, typename U>
__host__ __device__
comFloat3 multiply(T a, U factor){
  comFloat3 result;
  thrust::get<0>(result) = thrust::get<0>(a) * factor;
  thrust::get<1>(result) = thrust::get<1>(a) * factor;
  thrust::get<2>(result) = thrust::get<2>(a) * factor;
  return result;
}

__host__ __device__
Float3 multiply(Float3 a, float factor){
  Float3 result;
  thrust::get<0>(result) = thrust::get<0>(a) * factor;
  thrust::get<1>(result) = thrust::get<1>(a) * factor;
  thrust::get<2>(result) = thrust::get<2>(a) * factor;
  return result;
}

__host__ __device__
float vecnorm(comFloat3 a){
  return sqrt(thrust::norm(thrust::get<0>(a))+thrust::norm(thrust::get<1>(a))
              +thrust::norm(thrust::get<2>(a)));
}

__host__ __device__
float vecnorm(Float3 a){
  return sqrt(thrust::get<0>(a)*thrust::get<0>(a)+thrust::get<1>(a)*thrust::get<1>(a)
              +thrust::get<2>(a)*thrust::get<2>(a));
}

__host__ __device__
Float3 normalize(Float3 a){
  float norm = sqrt(thrust::get<0>(a)*thrust::get<0>(a)+thrust::get<1>(a)*thrust::get<1>(a)
              +thrust::get<2>(a)*thrust::get<2>(a));
  Float3 result;
  thrust::get<0>(result) = thrust::get<0>(a) / norm;
  thrust::get<1>(result) = thrust::get<1>(a) / norm;
  thrust::get<2>(result) = thrust::get<2>(a) / norm;
  return result;
}


template <typename T> 
__host__ __device__
T dotproduct(thrust::tuple<T,T,T> a, Float3 b){
  T result;
  result = thrust::get<0>(a)*thrust::get<0>(b)+thrust::get<1>(a)*thrust::get<1>(b)
              +thrust::get<2>(a)*thrust::get<2>(b);
  return result;
}

float dotproduct(float3 a, Float3 b){
  float result;
  result = a.x*thrust::get<0>(b)+a.y*thrust::get<1>(b)+a.z*thrust::get<2>(b);
  return result;
}

__host__ __device__
Float3 crossproduct(Float3 a, Float3 b){
  Float3 result;
  thrust::get<0>(result) = thrust::get<1>(a)*thrust::get<2>(b)-thrust::get<1>(b)*thrust::get<2>(a);
  thrust::get<1>(result) = thrust::get<2>(a)*thrust::get<0>(b)-thrust::get<2>(b)*thrust::get<0>(a);
  thrust::get<2>(result) = thrust::get<1>(b)*thrust::get<0>(a)-thrust::get<1>(a)*thrust::get<0>(b);
  return result;
}

__host__ __device__
comFloat3 crossproduct(Float3 a, comFloat3 b){
  comFloat3 result;
  thrust::get<0>(result) = thrust::get<1>(a)*thrust::get<2>(b)-thrust::get<1>(b)*thrust::get<2>(a);
  thrust::get<1>(result) = thrust::get<2>(a)*thrust::get<0>(b)-thrust::get<2>(b)*thrust::get<0>(a);
  thrust::get<2>(result) = thrust::get<1>(b)*thrust::get<0>(a)-thrust::get<1>(a)*thrust::get<0>(b);
  return result;
}

__host__ __device__
comFloat3 crossproduct(comFloat3 a, Float3 b){
  comFloat3 result;
  thrust::get<0>(result) = thrust::get<1>(a)*thrust::get<2>(b)-thrust::get<1>(b)*thrust::get<2>(a);
  thrust::get<1>(result) = thrust::get<2>(a)*thrust::get<0>(b)-thrust::get<2>(b)*thrust::get<0>(a);
  thrust::get<2>(result) = thrust::get<1>(b)*thrust::get<0>(a)-thrust::get<1>(a)*thrust::get<0>(b);
  return result;
}

__device__
comThr cos_internal_angle(comThr n, float theta){
  return sqrt(1.0 - sin(theta)/n*sin(theta)/n);
}

__device__
comThr rs(float theta, comThr n0, comThr nt){
  comThr p0 = n0*cos_internal_angle(n0, theta);
  comThr pt = nt*cos_internal_angle(nt/n0, theta);
  return (p0-pt)/(p0+pt);
}

__device__
comThr rp(float theta, comThr n0, comThr nt){
  comThr q0 = cos_internal_angle(n0, theta)/n0;
  comThr qt = cos_internal_angle(nt, theta)/nt;
  return (q0-qt)/(q0+qt);
}

struct add6 : public thrust::binary_function<float6, float6, float6>
{
  __host__ __device__
  float6 operator()(float6 a, float6 b) {
    float6 result;
    thrust::get<0>(result) = thrust::get<0>(a) + thrust::get<0>(b);
    thrust::get<1>(result) = thrust::get<1>(a) + thrust::get<1>(b);
    thrust::get<2>(result) = thrust::get<2>(a) + thrust::get<2>(b);
    thrust::get<3>(result) = thrust::get<3>(a) + thrust::get<3>(b);
    thrust::get<4>(result) = thrust::get<4>(a) + thrust::get<4>(b);
    thrust::get<5>(result) = thrust::get<5>(a) + thrust::get<5>(b);
    return result;
  }
};