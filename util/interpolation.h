// 
__global__
void setvalue(int batchnum, int gap, int nyqst, int n, int nfinal, float2 *vec){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < batchnum){
    vec[gap*i+nyqst-1] = vec[gap*i+nyqst-1]/2.0;
    vec[gap*i+nyqst+nfinal-n-1] = vec[gap*i+nyqst-1];
  }
}

void interpFFT(cufftHandle plan_phi, cufftHandle plan_inverse_phi, cufftHandle plan_theta, cufftHandle plan_inverse_theta, 
    devicef2vec &d_vec, devicef2vec &d_vec_result, int initialphinum, int finalphinum, int initialthetanum, int finalthetanum, 
    int phibatchnum, int thetabatchnum){

  const int NUM_THREADS_PER_BLOCK = 256;

  // forward phi
  checkCudaErrors(cufftExecC2C(plan_phi, thrust::raw_pointer_cast(d_vec.data()),thrust::raw_pointer_cast(d_vec_result.data()), CUFFT_FORWARD));

  // need to clear out d_vec
  thrust::fill(d_vec.begin(), d_vec.end(), zerofloat2);

  // pad + copy (from d_vec to d_vec_result)
  int nyqst = std::ceil((initialphinum+1)/2.0);
  int initialphidist = initialphinum;
  int finalphidist = finalphinum;
  for (int i = 0; i < phibatchnum; ++i){
    thrust::copy(d_vec_result.begin()+initialphidist*i, d_vec_result.begin()+initialphidist*i+nyqst, d_vec.begin()+finalphidist*i);
    thrust::copy(d_vec_result.begin()+initialphidist*i+nyqst, d_vec_result.begin()+initialphidist*i+initialphinum, 
                    d_vec.begin()+finalphidist*i+nyqst+finalphinum-initialphinum);
  }

  // set value
  if (initialphinum%2==0){
    int numblock = std::ceil(phibatchnum/NUM_THREADS_PER_BLOCK)+1;
    setvalue<<<numblock, 256>>>(phibatchnum, finalphidist, nyqst, initialphinum, finalphinum, thrust::raw_pointer_cast(d_vec.data()));
  }

  // inverse phi
  checkCudaErrors(cufftExecC2C(plan_inverse_phi, thrust::raw_pointer_cast(d_vec.data()), 
            thrust::raw_pointer_cast(d_vec_result.data()), CUFFT_INVERSE));

  // need to scale down by N (can we apply scale first in forward?)
  int middlenum = initialthetanum*finalphinum;
  thrust::transform(d_vec_result.begin(), d_vec_result.begin()+middlenum, d_vec_result.begin(), Scale_by_constant((float)(initialphinum)));

  // interpolate theta
  checkCudaErrors(cufftExecC2C(plan_theta, thrust::raw_pointer_cast(d_vec_result.data()),
            thrust::raw_pointer_cast(d_vec.data()), CUFFT_FORWARD));

  // need to clear out d_vec_result
  thrust::fill(d_vec_result.begin(),d_vec_result.end(), zerofloat2);

  // pad + copy (from d_vec_result to d_vec)
  nyqst = std::ceil((initialthetanum+1)/2.0);
  int initialthetadist = initialthetanum;
  int finalthetadist = finalthetanum;
  for (int i = 0; i < thetabatchnum; ++i){ // thetabatchnum should match finalphinum
    thrust::copy(d_vec.begin()+initialthetadist*i, d_vec.begin()+initialthetadist*i+nyqst, d_vec_result.begin()+finalthetadist*i);
    thrust::copy(d_vec.begin()+initialthetadist*i+nyqst, d_vec.begin()+initialthetadist*i+initialthetanum, 
    d_vec_result.begin()+finalthetadist*i+nyqst+finalthetanum-initialthetanum);
  }

  // set value
  if (initialthetanum%2==0){
    int numblock = std::ceil(phibatchnum/NUM_THREADS_PER_BLOCK)+1;
    setvalue<<<numblock, 256>>>(thetabatchnum, finalthetadist, nyqst, initialthetanum, finalthetanum, thrust::raw_pointer_cast(d_vec_result.data()));
  }

  // need check
  // inverse theta
  checkCudaErrors(cufftExecC2C(plan_inverse_theta, thrust::raw_pointer_cast(d_vec_result.data()),
    thrust::raw_pointer_cast(d_vec.data()), CUFFT_INVERSE));

  // need to scale down by N
  int finalnum = finalthetanum*finalphinum;
  thrust::transform(d_vec.begin(), d_vec.begin()+finalnum, d_vec.begin(), Scale_by_constant((float)(initialthetanum)));

  // final result is stored in d_vec!
}

void copyandinterp(const devicef2vec &field, int index, int numdir, devicef2vec &tmp1, devicef2vec &tmp2, int numdirthislevel, 
    cufftHandle plan_phi_0, cufftHandle plan_inverse_phi_0, cufftHandle plan_theta_0, cufftHandle plan_inverse_theta_0, 
    int initialphinum, int finalphinum, int initialthetanum, int finalthetanum, int phibatchnum, int thetabatchnum){

  thrust::fill(tmp1.begin(),tmp1.end(), zerofloat2);

  thrust::copy(field.begin()+index*numdir, field.begin()+(index+1)*numdir, tmp1.begin());

  interpFFT(plan_phi_0, plan_inverse_phi_0, plan_theta_0, plan_inverse_theta_0, tmp1, tmp2, 
  initialphinum, finalphinum, initialthetanum, finalthetanum, phibatchnum, thetabatchnum);

}