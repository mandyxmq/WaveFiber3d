// 
struct current_green_center
{
  Float3 dir;
  const float k;
  current_green_center(Float3 _dir, float _k) : dir(_dir), k(_k) {}

  __device__
  float6 operator()(const Float6& x, const comFloat6& y) const {
    // x is source point and center, y is current
    float6 field;
    comThr cunit(0.0,1.0);
    float xdiff = thrust::get<0>(x)-thrust::get<3>(x);
    float ydiff = thrust::get<1>(x)-thrust::get<4>(x);
    float zdiff = thrust::get<2>(x)-thrust::get<5>(x);
    float dot   = xdiff*thrust::get<0>(dir)+ydiff*thrust::get<1>(dir)+zdiff*thrust::get<2>(dir);
    comThr green = exp(cunit*k*dot);

    thrust::get<0>(field).x = (thrust::get<0>(y)*green).real();
    thrust::get<0>(field).y = (thrust::get<0>(y)*green).imag();

    thrust::get<1>(field).x = (thrust::get<1>(y)*green).real();
    thrust::get<1>(field).y = (thrust::get<1>(y)*green).imag();

    thrust::get<2>(field).x = (thrust::get<2>(y)*green).real();
    thrust::get<2>(field).y = (thrust::get<2>(y)*green).imag();

    thrust::get<3>(field).x = (thrust::get<3>(y)*green).real();
    thrust::get<3>(field).y = (thrust::get<3>(y)*green).imag();

    thrust::get<4>(field).x = (thrust::get<4>(y)*green).real();
    thrust::get<4>(field).y = (thrust::get<4>(y)*green).imag();

    thrust::get<5>(field).x = (thrust::get<5>(y)*green).real();
    thrust::get<5>(field).y = (thrust::get<5>(y)*green).imag();

    return field;
  }
};

void current_times_green_center(Float3 dir, float k, const deviceFvec& x, const deviceFvec& y, const deviceFvec& z, 
    const deviceFvec& centerx, const deviceFvec& centery, const deviceFvec& centerz, const deviceC6vec& currentvec, 
    devicef2vec& Ex, devicef2vec& Ey, devicef2vec& Ez, devicef2vec& Hx, devicef2vec& Hy, devicef2vec& Hz, int max_nodes)
{
  thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(x.begin(), y.begin(), z.begin(), 
    centerx.begin(), centery.begin(), centerz.begin())), 
    thrust::make_zip_iterator(thrust::make_tuple(x.end(), y.end(), z.end(), centerx.end(), centery.end(), centerz.end())), 
    currentvec.begin(), thrust::make_zip_iterator(thrust::make_tuple(Ex.begin(), Ey.begin(), Ez.begin(), 
    Hx.begin(), Hy.begin(), Hz.begin())), current_green_center(dir, k));
}


struct center_to_center
{
  Float3 dir;
  const float k;
  center_to_center(Float3 _dir, float _k) : dir(_dir), k(_k) {}

  __device__
  float6 operator()(const Float6& x, const float6& y) const {
    // x is source point and center, y is current
    float6 field;
    comThr cunit(0.0,1.0);
    float xdiff = thrust::get<0>(x)-thrust::get<3>(x);
    float ydiff = thrust::get<1>(x)-thrust::get<4>(x);
    float zdiff = thrust::get<2>(x)-thrust::get<5>(x);
    float dot   = xdiff*thrust::get<0>(dir)+ydiff*thrust::get<1>(dir)+zdiff*thrust::get<2>(dir);
    comThr green = exp(cunit*k*dot);

    thrust::get<0>(field) = thrust::get<0>(y)*green;
    thrust::get<1>(field) = thrust::get<1>(y)*green;
    thrust::get<2>(field) = thrust::get<2>(y)*green;
    thrust::get<3>(field) = thrust::get<3>(y)*green;
    thrust::get<4>(field) = thrust::get<4>(y)*green;
    thrust::get<5>(field) = thrust::get<5>(y)*green;

    return field;
  }
};


struct translation
{
  translation(){}

  __device__
  float6 operator()(const float6& x, const float2& factor) const {
    return x*factor;
  }
};

void translation_compute(devicef2vec& Ex, devicef2vec& Ey, devicef2vec& Ez, 
    devicef2vec& Hx, devicef2vec& Hy, devicef2vec& Hz, const devicef2vec &factor, 
    int startindex, int endindex, int leveloffset)
{
  // x, y, z is source point or child center coordinate
  thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(Ex.begin()+startindex, Ey.begin()+startindex, Ez.begin()+startindex, 
                    Hx.begin()+startindex, Hy.begin()+startindex, Hz.begin()+startindex)),
                    thrust::make_zip_iterator(thrust::make_tuple(Ex.begin()+endindex, Ey.begin()+endindex, Ez.begin()+endindex, 
                    Hx.begin()+endindex, Hy.begin()+endindex, Hz.begin()+endindex)), factor.begin()+leveloffset,
                    thrust::make_zip_iterator(thrust::make_tuple(Ex.begin()+startindex, Ey.begin()+startindex, Ez.begin()+startindex, 
                    Hx.begin()+startindex, Hy.begin()+startindex, Hz.begin()+startindex)), translation());
}


struct field_area
{
  field_area(){}

  __device__
  comFloat6 operator()(const comFloat6& x, const float& area) const {
    comFloat6 field;
    thrust::get<0>(field) = thrust::get<0>(x) * area;
    thrust::get<1>(field) = thrust::get<1>(x) * area;
    thrust::get<2>(field) = thrust::get<2>(x) * area;
    thrust::get<3>(field) = thrust::get<3>(x) * area;
    thrust::get<4>(field) = thrust::get<4>(x) * area;
    thrust::get<5>(field) = thrust::get<5>(x) * area;
    return field;
  }
};

void field_times_area(deviceC6vec& fields, deviceFvec& area)
{
  thrust::transform(fields.begin(), fields.end(), area.begin(), fields.begin(), field_area());
}

struct multiplyfactor
{
  comThr Efactor;
  comThr Hfactor;
  multiplyfactor(const comThr &Efactor_, const comThr &Hfactor_):Efactor(Efactor_),Hfactor(Hfactor_){}

  __device__
  comFloat6 operator()(const comFloat6& x) const {
    comFloat6 field;
    thrust::get<0>(field) = thrust::get<0>(x) * Efactor;
    thrust::get<1>(field) = thrust::get<1>(x) * Efactor;
    thrust::get<2>(field) = thrust::get<2>(x) * Efactor;
    thrust::get<3>(field) = thrust::get<3>(x) * Hfactor;
    thrust::get<4>(field) = thrust::get<4>(x) * Hfactor;
    thrust::get<5>(field) = thrust::get<5>(x) * Hfactor;
    return field;
  }
};

void multiply_current_factor(deviceC6vec& X, const comThr &Efactor, const comThr &Hfactor)
{
  thrust::transform(X.begin(), X.end(), X.begin(), multiplyfactor(Efactor, Hfactor));
}

/////////low level call///////
__global__
void sumchildren(const float2 *Ex, const float2 *Ey, const float2 *Ez, const float2 *Hx, const float2 *Hy, const float2 *Hz, 
float2 *Ex_parent, float2 *Ey_parent, float2 *Ez_parent, float2 *Hx_parent, float2 *Hy_parent, float2 *Hz_parent, int numdir)
{
  // for all direction, sum 8 children
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  //printf("within sumchildren");
  //printf("%d, \n", i);
  if (i < numdir){
    Ex_parent[i] = Ex_parent[i] + Ex[i];
    Ey_parent[i] = Ey_parent[i] + Ey[i];
    Ez_parent[i] = Ez_parent[i] + Ez[i];
    Hx_parent[i] = Hx_parent[i] + Hx[i];
    Hy_parent[i] = Hy_parent[i] + Hy[i];
    Hz_parent[i] = Hz_parent[i] + Hz[i];
  }
}


__global__
void computeintensity(float *intensity, int obnum, float2 *Exvec, float2 *Eyvec, float2 *Ezvec, 
float2 *Hxvec, float2 *Hyvec, float2 *Hzvec, float *dirvec, float Z0){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < obnum){
    float dirx = dirvec[i*3];
    float diry = dirvec[i*3+1];
    float dirz = dirvec[i*3+2];

    float2 Ex = Exvec[i];
    float2 Ey = Eyvec[i];
    float2 Ez = Ezvec[i];

    float2 Hx = Hxvec[i];
    float2 Hy = Hyvec[i];
    float2 Hz = Hzvec[i];

    float2 dotvalue = Ex*dirx + Ey*diry + Ez*dirz;
    Ex = Ex - dotvalue*dirx;
    Ey = Ey - dotvalue*diry;
    Ez = Ez - dotvalue*dirz;
    dotvalue = Hx*dirx + Hy*diry + Hz*dirz;
    Hx = Hx - dotvalue*dirx;
    Hy = Hy - dotvalue*diry;
    Hz = Hz - dotvalue*dirz;

    Ex = Ex - (Hz*diry-Hy*dirz) * Z0;
    Ey = Ey - (-(Hz*dirx-Hx*dirz)) * Z0;
    Ez = Ez - (Hy*dirx-Hx*diry) * Z0;

    intensity[i] = norm(Ex) + norm(Ey) + norm(Ez);
  }
}

void average_cpu(const hostFvec &scattering_h, hostFvec &scattering_final_h, const int thetaofinal, const int thetaonum, 
    const int phiofinal, const int phionum){

  std::cout<<"average phi"<<std::endl;
  int ratiophi = std::ceil(phionum/phiofinal);
  int ratiotheta = std::ceil(thetaonum/thetaofinal);

  std::cout<<"ratiophi "<<ratiophi<<" ratiotheta "<<ratiotheta<<" thetaofinal "<<thetaofinal<<
    " thetaonum "<<thetaonum<<" phiofinal "<<phiofinal<<" phionum "<<phionum<<std::endl;

  float scattering[thetaonum*phiofinal] = {0.f};
  for (int i=0; i<thetaonum; ++i){
    for (int j=0; j<phiofinal; ++j){
      for (int k = 0; k < ratiophi; ++k){
        scattering[i*phiofinal+j] += scattering_h[i*phionum+j*ratiophi+k] / (float)ratiophi;
      }
    }
  }

  std::cout<<"average theta"<<std::endl;
  for (int i = 0; i < thetaofinal; ++i){
    for (int j = 0; j < phiofinal; ++j){
      for (int k = 0; k < ratiotheta; ++k){
        scattering_final_h[i*phiofinal+j] += scattering[(i*ratiotheta+k)*phiofinal+j] / (float)ratiotheta;
      }
    }
  }

}
