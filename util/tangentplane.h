// local tangent plane calculation on GPU
struct LocalTangentPlane
{
  const Float3 wavedir;
  const Float3 Edir;
  const Float3 Hdir;
  const comThr eta_in;
  const comThr eta_out;
  const float  Z0;
  const float  costhetai;
  const float  sigma;
  const float  k0;
  const char   mode;
  LocalTangentPlane(Float3 _wavedir, Float3 _Edir, Float3 _Hdir, comThr _eta_in, comThr _eta_out, float _Z0, 
  float _costhetai, float _sigma, float _k0, char mode_) : wavedir(_wavedir), Edir(_Edir), Hdir(_Hdir), 
  eta_in(_eta_in), eta_out(_eta_out), Z0(_Z0), costhetai(_costhetai), sigma(_sigma), k0(_k0), mode(mode_){}

  __device__
  comFloat6 operator()(const Float3& source, const Float3& normal) const{
    comFloat6 current;
    comThr cunit(0.0, 1.0);

    float localphi, beamfactor, dot, tmp;
    Float3 reflect_plane_normal, Rdir, decaydir;
    comThr factor, Rs, Rp, Ei0, Hi0;
    comFloat3 Ei, Hi, Ei_perp, Ei_par, Es_perp, Hi_perp, Hi_par, Hs_perp, Hs_par, Es_par, Es, Hs, Et, Ht, J, M;

    // compute dot product (used to decide illuminated or shadow)
    float dotvalue = dotproduct(wavedir, normal);
    
    if (dotvalue <=0 ){
      dot                  = dotproduct(wavedir, source);
      tmp                  = thrust::get<2>(source);
      beamfactor           = exp(-0.5*tmp*tmp/(sigma*sigma));
      Ei0                  = beamfactor * exp(-cunit*k0*dot);
      Hi0                  = Ei0 / Z0;
      Ei                   = multiply(Edir, Ei0);
      Hi                   = multiply(Hdir, Hi0);
      float tol = 1e-10;
      if(std::abs(std::abs(dotvalue)-1)<tol){
        if (mode=='M'){
          Rs                 = (eta_in-eta_out)/(eta_in+eta_out);
          Es                 = multiply(Ei, Rs);
          Hs                 = multiply(Hi, Rs);
        }else{
          Rp                 = (eta_out-eta_in)/(eta_in+eta_out);
          Es                 = multiply(Ei, Rp);
          Hs                 = multiply(Hi, Rp);
        }
      }else{
        reflect_plane_normal = normalize(crossproduct(wavedir, normal));
        factor               = dotproduct(Ei, reflect_plane_normal);
        Ei_perp              = multiply(reflect_plane_normal, factor);
        Ei_par               = subtract(Ei, Ei_perp);
        Hi_par               = multiply(crossproduct(wavedir, Ei_par), 1.0/Z0);
        Hi_perp              = subtract(Hi, Hi_par);

        localphi             = acos(abs(dotvalue));
        Rs                   = rs(localphi, eta_in, eta_out);
        Rp                   = rp(localphi, eta_in, eta_out);
        Es                   = subtract(multiply(Ei_perp, Rs), multiply(Ei_par, Rp));
        Hs                   = subtract(multiply(Hi_par, Rp), multiply(Hi_perp, Rs));
      }
    }else{
      Es = minus(Ei);
      Hs = minus(Hi);
    }
    Et = add(Ei, Es);
    Ht = add(Hi, Hs);
    M  = crossproduct(Et, normal);
    J  = crossproduct(normal, Ht);
    thrust::get<0>(current) = thrust::get<0>(J);
    thrust::get<1>(current) = thrust::get<1>(J);
    thrust::get<2>(current) = thrust::get<2>(J);
    thrust::get<3>(current) = thrust::get<0>(M);
    thrust::get<4>(current) = thrust::get<1>(M);
    thrust::get<5>(current) = thrust::get<2>(M);

    return current;
  }

};

void computeLocalTangentPlane(deviceFvec& x, deviceFvec& y, deviceFvec& z, deviceFvec& norx, deviceFvec& nory, deviceFvec& norz, 
        deviceC6vec& current, Float3 wavedir, Float3 Edir, Float3 Hdir, comThr eta_in, comThr eta_out, float Z0, 
        float costhetai, float sigma, float k0, char mode)
{
  thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(x.begin(), y.begin(), z.begin())),
                    thrust::make_zip_iterator(thrust::make_tuple(x.end(), y.end(), z.end())), 
                    thrust::make_zip_iterator(thrust::make_tuple(norx.begin(), nory.begin(), norz.begin())), current.begin(), 
                    LocalTangentPlane(wavedir, Edir, Hdir, eta_in, eta_out, Z0, costhetai, sigma, k0, mode));
}
