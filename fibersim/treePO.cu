#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sys/stat.h>
#include <sys/time.h>
#include <cufft.h>
#include <cufftXt.h>
#include <bits/stdc++.h>

#include "../util/util.h"
#include "../util/tangentplane.h"
#include "../util/fields.h"
#include "../util/interpolation.h"
#include "../util/geometry.h"
#include "../util/tree.h"


////////////////////////////////////////////////////////////////////////////////
// Main entry point.
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {

  float delta;
  struct timeval start, end;
  gettimeofday(&start, NULL);

  int lambdaindexstart = atoi(argv[1]);
  int lambdaindexend = atoi(argv[2]);
  int thetaindexstart = atoi(argv[3]);
  int phiiindexstart = atoi(argv[4]);
  int fiberindex = atoi(argv[5]);
  int max_depth = atoi(argv[6]);
  int startindex = atoi(argv[7]); // start index in one ring of the fiber
  int endindex = atoi(argv[8]);   // end index in one ring of the fiber

  std::cout<<"lambda "<<lambdaindexstart<<" "<<lambdaindexend<<" thetaindexstart "<<thetaindexstart<<
  " phiiindexstart "<<phiiindexstart<<std::endl;

  std::string outputdir = "output";
  char dir_array[outputdir.length()];
  strcpy(dir_array, outputdir.c_str());
  mkdir(dir_array, 0777);

  float periodvec[4]  = { 20 * 1e-6, 50 * 1e-6, 400 * 1e-6, 400 * 1e-6 };
  float radius1vec[4] = { 20 * 1e-6, 40 * 1e-6,  40 * 1e-6,  40 * 1e-6 };
  float radius2vec[4] = { 20 * 1e-6, 40 * 1e-6,  35 * 1e-6,  35 * 1e-6 };
  float sigmavec[4]   = { 10 * 1e-6, 10 * 1e-6, 150 * 1e-6, 150 * 1e-6 };
  int numelvec[4]     = { 3000, 5026, 4420, 4720 };
  int numelvertvec[4] = {  478, 1000, 8000, 8000 };
  int thetaovec[4] = {  450,  900, 7200, 7200 };
  int phiovec[4]   = {  880, 1440, 1440, 1440 }; 

  // geometry
  float period  =  periodvec[fiberindex];
  float radius1 = radius1vec[fiberindex];
  float radius2 = radius2vec[fiberindex];
  float radius  = (radius1 + radius2) / 2;
  float gaussiansigma = sigmavec[fiberindex];
  int numel = numelvec[fiberindex];
  int numelvert = numelvertvec[fiberindex];
  float unitz = period / numelvert;

  int thetainum = 180;
  int phiinum = 360;
  int thetaonum = thetaovec[fiberindex];
  int phionum = phiovec[fiberindex];
  int thetaofinal = 180;
  int phiofinal = 360;  
  int obnum = thetaonum * phionum;
  int obfinal = thetaofinal * phiofinal;
  float thetastartoffset = 1e-10;

  int lambdanum = 25;
  float wavelength_min = 400;
  float wavelength_max = 700;

  float Dis = 1;  

  // read in rough surface file
  std::cout<<"read in surface height"<<std::endl;
  std::string filename = "mesh/fiber"+std::to_string(fiberindex)+".binary";
  std::ifstream myfile(filename, std::ios::in|std::ios::binary);
  int datanum = numel*numelvert*2;
  std::cout<<"datanum "<<datanum<<std::endl;
  float *surfaceheight = new float[datanum];
  myfile.read((char*)&surfaceheight[0], datanum * sizeof(float));
  std::cout<<"finish reading file"<<std::endl;
  std::cout<<"surfaceheight[0] "<<surfaceheight[0]<<" surfaceheight[1] "<<surfaceheight[1]<<std::endl;


  // tree parameters
  float beta = 6.0;
  float samplingenlargeratio = 1.2;
  const int min_points_per_node = 16;
  float enlargeratio = 1.1;
  float xrange = radius1 * enlargeratio;
  float yrange = radius2 * enlargeratio;
  float zrange = period * 0.5 * enlargeratio;
  int max_nodes = (pow(8, max_depth+1)-1)/7;
  int nonleafnodes = (pow(8, max_depth)-1)/7;
  std::cout<<"max_nodes "<<max_nodes<<std::endl;

  // Initialize all the points
  //int sourcenum = numel * numelvert;
  // int startindex = std::floor(numel/4.0 - 5);
  // int endindex = std::floor(3.0*numel/4.0 + 5);
  int sourcenum = (endindex - startindex + 1) * numelvert;

  std::cout<<"sourcenum "<<sourcenum<<std::endl;

  hostFvec x0(sourcenum);
  hostFvec y0(sourcenum);
  hostFvec z0(sourcenum);
  hostFvec x1(sourcenum);
  hostFvec y1(sourcenum);
  hostFvec z1(sourcenum);

  // normals
  hostFvec normalx0(sourcenum);
  hostFvec normalx1(sourcenum);
  hostFvec normaly0(sourcenum);
  hostFvec normaly1(sourcenum);
  hostFvec normalz0(sourcenum);
  hostFvec normalz1(sourcenum);

  // area
  hostFvec area0(sourcenum);
  hostFvec area1(sourcenum);

  std::cout<<"set_sourcepoint"<<std::endl;
  //set_sourcepoint(radius1, radius2, period, unitz, x0, y0, z0, area0, normalx0, normaly0, normalz0, numel, numelvert, surfaceheight);
  set_sourcepoint(radius1, radius2, period, unitz, x0, y0, z0, area0, normalx0, normaly0, normalz0, numel, numelvert, surfaceheight, startindex, endindex);
 

  // Initialize all the points
  std::cout<<"deviceFvec"<<std::endl;
  deviceFvec x_d0=x0;
  deviceFvec x_d1=x1;
  deviceFvec y_d0=y0;
  deviceFvec y_d1=y1;
  deviceFvec z_d0=z0;
  deviceFvec z_d1=z1;

  deviceFvec normalx_d0=normalx0;
  deviceFvec normalx_d1=normalx1;
  deviceFvec normaly_d0=normaly0;
  deviceFvec normaly_d1=normaly1;
  deviceFvec normalz_d0=normalz0;
  deviceFvec normalz_d1=normalz1;

  deviceFvec area_d0=area0;
  deviceFvec area_d1=area1;

  std::cout<<"create points"<<std::endl;
  // Host structures to analyze the device ones, cast from sourcevec
  Points points_init[2];
  points_init[0].set(thrust::raw_pointer_cast(&x_d0[0]),
                     thrust::raw_pointer_cast(&y_d0[0]),
                     thrust::raw_pointer_cast(&z_d0[0]));
  points_init[1].set(thrust::raw_pointer_cast(&x_d1[0]),
                     thrust::raw_pointer_cast(&y_d1[0]),
                     thrust::raw_pointer_cast(&z_d1[0]));

  std::cout<<"allocate points"<<std::endl;
  // Allocate memory to store points.
  Points *points;
  checkCudaErrors(cudaMalloc((void **)&points, 2 * sizeof(Points)));
  checkCudaErrors(cudaMemcpy(points, points_init, 2 * sizeof(Points),
                             cudaMemcpyHostToDevice));

  // Host structures to analyze the device ones (normals).
  std::cout<<"create normals"<<std::endl;
  Points normals_init[2];
  normals_init[0].set(thrust::raw_pointer_cast(&normalx_d0[0]),
                      thrust::raw_pointer_cast(&normaly_d0[0]),
                      thrust::raw_pointer_cast(&normalz_d0[0]));
  normals_init[1].set(thrust::raw_pointer_cast(&normalx_d1[0]),
                      thrust::raw_pointer_cast(&normaly_d1[0]),
                      thrust::raw_pointer_cast(&normalz_d1[0]));

  std::cout<<"allocate normals"<<std::endl;
  // Allocate memory to store normals.
  Points *normals;
  checkCudaErrors(cudaMalloc((void **)&normals, 2 * sizeof(Points)));
  checkCudaErrors(cudaMemcpy(normals, normals_init, 2 * sizeof(Points),
                             cudaMemcpyHostToDevice));

  // Host structures to analyze the device ones (area).
  std::cout<<"create areas"<<std::endl;
  Area area_init[2];
  area_init[0].set(thrust::raw_pointer_cast(&area_d0[0]));
  area_init[1].set(thrust::raw_pointer_cast(&area_d1[0]));

  std::cout<<"allocate areas"<<std::endl;
  // Allocate memory to store areas.
  Area *areavec;
  checkCudaErrors(cudaMalloc((void **)&areavec, 2 * sizeof(Area)));
  checkCudaErrors(cudaMemcpy(areavec, area_init, 2 * sizeof(Area),
                             cudaMemcpyHostToDevice));

  printf("Find/set the device.");
  // The test requires an architecture SM35 or greater (CDP capable).
  int cuda_device = findCudaDevice(argc, (const char **)argv);
  cudaDeviceProp deviceProps;
  checkCudaErrors(cudaGetDeviceProperties(&deviceProps, cuda_device));
  int cdpCapable = (deviceProps.major == 3 && deviceProps.minor >= 5) ||
    deviceProps.major >= 4;

  printf("GPU device %s has compute capabilities (SM %d.%d)\n",
         deviceProps.name, deviceProps.major, deviceProps.minor);

  if (!cdpCapable) {
    std::cerr << "cdpOcTree requires SM 3.5 or higher to use CUDA Dynamic "
      "Parallelism.  Exiting...\n"
              << std::endl;
    exit(EXIT_WAIVED);
  }

  // Allocate memory to store the tree.
  Octree_node root;
  root.set_range(0, sourcenum);

  // add set range
  root.set_bounding_box(-xrange, -yrange, -zrange, xrange, yrange, zrange);

  Octree_node *nodes;
  checkCudaErrors(
                  cudaMalloc((void **)&nodes, max_nodes * sizeof(Octree_node)));
  checkCudaErrors(
                  cudaMemcpy(nodes, &root, sizeof(Octree_node), cudaMemcpyHostToDevice));


  // We set the recursion limit for CDP to max_depth.
  cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, max_depth);

  // Build the octree.
  int warp_size = deviceProps.warpSize;
  Parameters params(max_depth, min_points_per_node);
  std::cout << "Launching CDP kernel to build the octree" << std::endl;
  const int NUM_THREADS_PER_BLOCK = 256;  // Do not use less than 128 threads.
  const int NUM_WARPS_PER_BLOCK = NUM_THREADS_PER_BLOCK / warp_size;

  std::cout<<"NUM_WARPS_PER_BLOCK "<<NUM_WARPS_PER_BLOCK<<std::endl;


  const size_t smem_size = 8 * NUM_WARPS_PER_BLOCK * sizeof(int);
  build_octree_kernel<
    NUM_THREADS_PER_BLOCK><<<1, NUM_THREADS_PER_BLOCK, smem_size>>>(
                                                                    nodes, points, normals, areavec, params);
  checkCudaErrors(cudaGetLastError());

  std::cout<<"after building tree"<<std::endl;

  // convert nodes back to cpu
  // Copy nodes to CPU.
  Octree_node *host_nodes = new Octree_node[max_nodes];
  checkCudaErrors(cudaMemcpy(host_nodes, nodes,
                             max_nodes * sizeof(Octree_node),
                             cudaMemcpyDeviceToHost));


  // Precompute a non-empty node list
  std::vector<std::vector<int>> nonemptynodes;
  std::vector<int> nonemptynumbylevel(max_depth);      // leaf to top order
  // Go up the tree and loop through the nodes
  for (int level = max_depth; level>0; level--){
    int nonemptythislevel = 0;
    int nodeindexstart = (pow(8, level)-1)/7;
    int nodeindexend = (pow(8, level+1)-1)/7;

    nonemptynodes.push_back(std::vector<int>());

    for (int i = nodeindexstart; i < nodeindexend; ++i){
      int myshift = i-nodeindexstart;
      // leaf level
      if (level == max_depth){
        Octree_node node = host_nodes[i];
        int startindex = node.points_begin();
        int endindex = node.points_end();
        if (startindex<endindex){
          host_nodes[i].active=true;
          nonemptynodes[max_depth-level].push_back(i);
          nonemptythislevel++;
        }
        else
          host_nodes[i].active=false;

      }else{
        // non leaf levels
        Octree_node node = host_nodes[i];
        int startindex = node.points_begin();
        int endindex = node.points_end();
        if (startindex<endindex){
          // check children active or not
          bool Imactive = false;
          for (int childindex = 0; childindex < 8; ++childindex){
            int truechildindex = nodeindexend+myshift*8+childindex;
            if (host_nodes[truechildindex].active){
              Imactive = true;
              break;
            }
          }
          if (Imactive){
            host_nodes[i].active=true;
            nonemptynodes[max_depth-level].push_back(i);
            nonemptythislevel++;
          }else
            host_nodes[i].active=false;

        }else
          host_nodes[i].active=false;
      }
    }
    nonemptynumbylevel[max_depth-level] = nonemptythislevel;
  }

  // d0 uses to collect from children (or source points), my parent's center
  hostFvec centerx_h0(sourcenum, 0.0);
  hostFvec centery_h0(sourcenum, 0.0);
  hostFvec centerz_h0(sourcenum, 0.0);

  std::cout<<"nonleafnodes "<<nonleafnodes<<std::endl;
  // assign centers on cpu, make this on gpu!!!
  float3 center;
  for (int i = nonleafnodes; i < max_nodes; ++i){
    host_nodes[i].bounding_box().compute_center(center);
    int startindex = host_nodes[i].points_begin();
    int endindex = host_nodes[i].points_end();
    for (int j = startindex; j<endindex; ++j){
      centerx_h0[j] = center.x;
      centery_h0[j] = center.y;
      centerz_h0[j] = center.z;
    }
  }
  deviceFvec centerx_d0 = centerx_h0;
  deviceFvec centery_d0 = centery_h0;
  deviceFvec centerz_d0 = centerz_h0;

 
  deviceC6vec currentvec_dev(sourcenum);
  std::cout<<"max_depth "<<max_depth<<std::endl;
  int thetaonumvec[max_depth+1];
  int phionumvec[max_depth+1];
  int anglenumvec[max_depth+1];
  int anglesum_nonleaf = obnum;
  thetaonumvec[0] = thetaonum;
  phionumvec[0] = phionum;
  anglenumvec[0] = obnum;
  int maxnum = obnum;
  std::cout<<"level "<<0<<" thetaonum "<<thetaonumvec[0]<<" phionum "<<phionumvec[0]<<
    " anglenumvec[0] "<<anglenumvec[0]<<std::endl;

  for (int i = 1; i < max_depth+1; ++i){
    float d_theta = pow(0.5, i) * (period * samplingenlargeratio);
    int thetares = std::ceil(2*M_PI/(0.4e-6)*d_theta + beta*pow(2*M_PI/(0.4e-6)*d_theta, 1.0/3.0));
    thetaonumvec[i] = thetares;
    float d_phi = pow(0.5, i)*2.0*(std::max(radius1, radius2))* samplingenlargeratio;
    int phires = std::ceil(2*M_PI/(0.4e-6)*d_phi + beta*pow(2*M_PI/(0.4e-6)*d_phi, 1.0/3.0));
    phionumvec[i] = phires;

    anglenumvec[i] = thetaonumvec[i] * phionumvec[i];

    if (i<max_depth){
      anglesum_nonleaf += thetaonumvec[i] * phionumvec[i];
    }

    int curnum = nonemptynumbylevel[max_depth-i] * anglenumvec[i];
    if (curnum>maxnum)
      maxnum = curnum;

    std::cout<<"level "<<i<<" thetaonum "<<thetaonumvec[i]<<" phionum "<<phionumvec[i]<<
      " anglenumvec[i] "<<anglenumvec[i]<<" maxnum "<<maxnum<<std::endl;
  }
  std::cout<<"anglesum_nonleaf "<<anglesum_nonleaf<<std::endl;
  std::cout<<"maxnum "<<maxnum<<std::endl;


  hostFvec dirvec(obnum*3);
  for (int thetaoindex = 0; thetaoindex < thetaonum; thetaoindex++){
    float thetao = -M_PI/2+thetastartoffset+M_PI/thetaonum*thetaoindex;
    for (int phioindex = 0; phioindex < phionum; phioindex++){
      float phio = 2*M_PI/phionum*phioindex;
      int dirindex = thetaoindex*phionum + phioindex;
      float dirx = cos(thetao)*cos(phio);
      float diry = cos(thetao)*sin(phio);
      float dirz = sin(thetao);
      dirvec[dirindex*3]   = dirx;
      dirvec[dirindex*3+1] = diry;
      dirvec[dirindex*3+2] = dirz;
    }
  }
  deviceFvec dirvec_d=dirvec;

  // make fft plans
  std::vector<cufftHandle> plan_phi(max_depth);
  std::vector<cufftHandle> plan_inverse_phi(max_depth);
  std::vector<cufftHandle> plan_theta(max_depth);
  std::vector<cufftHandle> plan_inverse_theta(max_depth);

  for (int i = 0; i < max_depth; ++i){
    int rank       = 1;
    int batch      = thetaonumvec[i+1];
    int num[1]     = {phionumvec[i+1]};
    int inembed[1] = {anglenumvec[i+1]};
    int istride    = 1;
    int idist      = phionumvec[i+1];
    int onembed[1] = {anglenumvec[i+1]};
    int ostride    = 1;
    int odist      = phionumvec[i+1];
    cufftHandle plan_phi_;
    cufftPlanMany(&plan_phi_, rank, num, inembed,
                  istride, idist, onembed, ostride,
                  odist, CUFFT_C2C, batch);
    plan_phi[i] = plan_phi_;

    num[0]     = phionumvec[i];
    istride    = 1;
    idist      = phionumvec[i];
    ostride    = 1;
    odist      = phionumvec[i];
    cufftHandle plan_inverse_phi_;
    cufftPlanMany(&plan_inverse_phi_, rank, num, inembed,
                  istride, idist, onembed, ostride,
                  odist, CUFFT_C2C, batch);;
    plan_inverse_phi[i] = plan_inverse_phi_;

    batch      = phionumvec[i];   // phi already interpolated
    inembed[0] = {anglenumvec[i]};
    onembed[0] = {anglenumvec[i]};
    num[0]     = thetaonumvec[i+1];
    istride    = phionumvec[i];
    idist      = 1;
    ostride    = 1;                 // make it connected after forward FFT
    odist      = thetaonumvec[i+1];
    cufftHandle plan_theta_;  // from level 1 to 0

    std::cout<<"plan_theta_ inembed[0] "<<inembed[0]<<" onembed[0] "<<onembed[0]<<" num[0] "<<num[0]<<
      " istride "<<phionumvec[i]<<" idist "<<1<<" ostride "<<1<<" odist "<<thetaonumvec[i+1]<<" batch "<<batch<<std::endl;

    cufftPlanMany(&plan_theta_, rank, num, inembed,
                  istride, idist, onembed, ostride,
                  odist, CUFFT_C2C, batch);
    plan_theta[i] = plan_theta_;

    num[0]     = thetaonumvec[i];
    istride    = 1;
    idist      = thetaonumvec[i];
    ostride    = phionumvec[i];
    odist      = 1;
    cufftHandle plan_inverse_theta_;
    cufftPlanMany(&plan_inverse_theta_, rank, num, inembed,
                  istride, idist, onembed, ostride,
                  odist, CUFFT_C2C, batch);
    plan_inverse_theta[i] = plan_inverse_theta_;
  }

  float6 zerofloat6;
  thrust::get<0>(zerofloat6) = make_cuFloatComplex(0.0,0.0);
  thrust::get<1>(zerofloat6) = make_cuFloatComplex(0.0,0.0);
  thrust::get<2>(zerofloat6) = make_cuFloatComplex(0.0,0.0);
  thrust::get<3>(zerofloat6) = make_cuFloatComplex(0.0,0.0);
  thrust::get<4>(zerofloat6) = make_cuFloatComplex(0.0,0.0);
  thrust::get<5>(zerofloat6) = make_cuFloatComplex(0.0,0.0);

  devicef2vec Ex_d0(maxnum);
  devicef2vec Ey_d0(maxnum);
  devicef2vec Ez_d0(maxnum);
  devicef2vec Hx_d0(maxnum);
  devicef2vec Hy_d0(maxnum);
  devicef2vec Hz_d0(maxnum);

  int num = sourcenum>maxnum ? sourcenum : maxnum;
  std::cout<<"num "<<num<<" sourcenum "<<sourcenum<<" maxnum "<<maxnum<<std::endl;
  devicef2vec Ex_d1(num);
  devicef2vec Ey_d1(num);
  devicef2vec Ez_d1(num);
  devicef2vec Hx_d1(num);
  devicef2vec Hy_d1(num);
  devicef2vec Hz_d1(num);

  devicef2vec tmp1_ex(obnum);
  devicef2vec tmp1_ey(obnum);
  devicef2vec tmp1_ez(obnum);
  devicef2vec tmp1_hx(obnum);
  devicef2vec tmp1_hy(obnum);
  devicef2vec tmp1_hz(obnum);

  devicef2vec tmp2(obnum);

  hostf2vec Ex_final(obnum);
  hostf2vec Ey_final(obnum);
  hostf2vec Ez_final(obnum);
  hostf2vec Hx_final(obnum);
  hostf2vec Hy_final(obnum);
  hostf2vec Hz_final(obnum);

  deviceFvec scattering(obnum);
  hostFvec scattering_h(obnum);

  // loop through wavelengths
  for (int lambdaindex = lambdaindexstart; lambdaindex < lambdaindexend; ++lambdaindex){
    std::cout<<"lambdaindex "<<lambdaindex<<std::endl;

    float wavelength = wavelength_min + (wavelength_max-wavelength_min)/(float)lambdanum*lambdaindex;
    wavelength *= 1e-9;
    float k0 = 2 * M_PI / wavelength;
    float omega = k0 * c0;
    comThr Efactor = -cunit*omega*mu0/(4*M_PI)*exp(-cunit*k0*Dis)/Dis;
    comThr Hfactor = -cunit*omega*eps0/(4*M_PI)*exp(-cunit*k0*Dis)/Dis;

    // try to precomopute factors for center to center calculation, 8*num_dir for each non top level, 
    // num_dir is the number of directions for parent level. This calculation is per wavelength
    // arbitray levels (move this to GPU?)
    hostf2vec factor1(8*anglesum_nonleaf);
    int index = 0;
    float3 mycenter, childcenter;
    for (int levelindex =0; levelindex<max_depth;++levelindex){
      int nodesofar = (pow(8, levelindex)-1)/7;
      int nodeincludethislevel = (pow(8, levelindex+1)-1)/7;
      int curthetaonum = thetaonumvec[levelindex];
      int curphionum = phionumvec[levelindex];

      std::cout<<"curindex "<<index<<std::endl;
      std::cout<<"levelindex "<<levelindex<<std::endl;
      std::cout<<"curthetaonum "<<curthetaonum<<std::endl;
      std::cout<<"curphionum "<<curphionum<<std::endl;
      host_nodes[nodesofar].bounding_box().compute_center(mycenter);
      // loop for 8 children
      for (int i = 0; i < 8; ++i){
        host_nodes[nodeincludethislevel+i].bounding_box().compute_center(childcenter);
        float xdiff = childcenter.x-mycenter.x;
        float ydiff = childcenter.y-mycenter.y;
        float zdiff = childcenter.z-mycenter.z;
        // loop for theta directions
        for (int j = 0; j < curthetaonum; ++j){
          float thetao = -M_PI/2+thetastartoffset+M_PI/curthetaonum*j;
          // loor for phi directions
          for (int k = 0; k < curphionum; ++k){
            float phio = 2*M_PI/curphionum*k;
            // direction
            float dirx = cos(thetao)*cos(phio);
            float diry = cos(thetao)*sin(phio);
            float dirz = sin(thetao);
            float dot   = xdiff*dirx+ydiff*diry+zdiff*dirz;
            comThr green = exp(cunit*k0*dot);

            factor1[index].x = green.real();
            factor1[index].y = green.imag();
            index++;
          }
        }
      }
    }
    devicef2vec factor1_d=factor1;


    // loop through incident theta angles
    char mode;
    for (int thetaiindex = thetaindexstart; thetaiindex<thetaindexstart+1; thetaiindex++){
      float thetai = -M_PI/2+thetastartoffset+M_PI/thetainum*thetaiindex;
      std::cout<<"thetaiindex "<<thetaiindex<<" thetai "<<thetai<<std::endl;

      for (int phiiindex = phiiindexstart; phiiindex<phiiindexstart+1; phiiindex++){
        float phii = 2 * M_PI / phiinum * phiiindex;

        float sigma = 0;
        float kappa = sigma * wavelength / (4*M_PI*radius);
        comThr eta = 1.55 + kappa * cunit;
        float krho = k0 * cos(thetai);
        float kz   = k0 * sin(thetai);

        hostFvec scattering_final_h(obfinal, 0.f);
        for (int polarindex = 0; polarindex<1; ++polarindex){

          if (polarindex == 0)
            mode = 'M';
          else
            mode = 'E';

          // E H direction
          Float3 wavedir(3), Edir(3), Hdir(3);
          thrust::get<0>(wavedir) = cos(thetai)*cos(phii);
          thrust::get<1>(wavedir) = cos(thetai)*sin(phii);
          thrust::get<2>(wavedir) = sin(thetai);
          if (mode=='M'){
            thrust::get<0>(Edir) = -sin(thetai)*cos(phii);
            thrust::get<1>(Edir) = -sin(thetai)*sin(phii);
            thrust::get<2>(Edir) = cos(thetai);

            thrust::get<0>(Hdir) = sin(phii);
            thrust::get<1>(Hdir) = -cos(phii);
            thrust::get<2>(Hdir) = 0;
          }else{
            thrust::get<0>(Edir) = -sin(phii);
            thrust::get<1>(Edir) = cos(phii);
            thrust::get<2>(Edir) = 0;

            thrust::get<0>(Hdir) = -sin(thetai)*cos(phii);
            thrust::get<1>(Hdir) = -sin(thetai)*sin(phii);
            thrust::get<2>(Hdir) = cos(thetai);
          }

          computeLocalTangentPlane(x_d0, y_d0, z_d0, normalx_d0, normaly_d0, normalz_d0, currentvec_dev, 
            wavedir, Edir, Hdir, comThr(1.0,0.0), eta, Z0, cos(thetai), gaussiansigma, k0, mode);

          field_times_area(currentvec_dev, area_d0);
          multiply_current_factor(currentvec_dev, Efactor, Hfactor);

          gettimeofday(&end, NULL);
          delta = ((end.tv_sec  - start.tv_sec) * 1000000u +
                   end.tv_usec - start.tv_usec) / 1.e6;
          std::cout<<"after fresnel, time used: "<<delta<<std::endl;

          for (int i = max_depth; i >= 0; i--){
            int nodeindexstart = (pow(8, i)-1)/7;
            int nodeindexend = (pow(8, i+1)-1)/7;
            int curthetaonum = thetaonumvec[i];
            int curphionum = phionumvec[i];

            std::cout<<"level "<<i<<" nodeindexstart "<<nodeindexstart<<" nodeindexend "<<nodeindexend<<
              " curthetaonum "<<curthetaonum<<" curphionum "<<curphionum<<std::endl;

            // lowest level
            if (i==max_depth){
              for (int thetaoindex = 0; thetaoindex < curthetaonum; thetaoindex++){
                //std::cout<<"thetaoindex "<<thetaoindex<<" thetaonum "<<curthetaonum<<std::endl;
                //for (int thetaoindex = 0; thetaoindex < 1; thetaoindex++){
                float thetao = -M_PI/2+thetastartoffset+M_PI/curthetaonum*thetaoindex;
                //float thetao = 0;
                for (int phioindex = 0; phioindex < curphionum; phioindex++){
                  //for (int phioindex = 0; phioindex < 1; phioindex++){
                  //std::cout<<"phioindex "<<phioindex<<std::endl;
                  int dirindex = thetaoindex*curphionum + phioindex;

                  float phio = 2*M_PI/(curphionum)*phioindex;
                  //float phio = M_PI/3;
                  float dirx = cos(thetao)*cos(phio);
                  float diry = cos(thetao)*sin(phio);
                  float dirz = sin(thetao);
                  Float3 dir = Float3(dirx, diry, dirz);

                  current_times_green_center(dir, k0, x_d0, y_d0, z_d0, centerx_d0, centery_d0, centerz_d0, 
                    currentvec_dev, Ex_d1, Ey_d1, Ez_d1, Hx_d1, Hy_d1, Hz_d1, max_nodes);

                  // loop through parent node, reduce base on range
                  // loop through active nodes
                  int loopend;
                  if (i==0){
                    loopend = 1;
                  }else
                    loopend = nonemptynodes[0].size();
                  for (int vecindex = 0; vecindex < loopend; ++vecindex){
                    int nodeindex;
                    if (i==0)
                      nodeindex = 0;
                    else
                      nodeindex = nonemptynodes[0][vecindex];  // this is gauranteed to be an active node
                    // reduce field
                    const Octree_node &node = host_nodes[nodeindex]; // on cpu, make it gpu?
                    int startindex = node.points_begin();
                    int endindex = node.points_end();
                    float6 field = thrust::reduce(thrust::device,
                                                  thrust::make_zip_iterator(thrust::make_tuple(Ex_d1.begin()+startindex, 
                                                  Ey_d1.begin()+startindex, Ez_d1.begin()+startindex, 
                                                  Hx_d1.begin()+startindex, Hy_d1.begin()+startindex, Hz_d1.begin()+startindex)),
                                                  thrust::make_zip_iterator(thrust::make_tuple(Ex_d1.begin()+endindex, 
                                                  Ey_d1.begin()+endindex, Ez_d1.begin()+endindex, 
                                                  Hx_d1.begin()+endindex, Hy_d1.begin()+endindex, Hz_d1.begin()+endindex)),
                                                  zerofloat6,
                                                  add6());

                    // store this value
                    // last level, start index needs to count how many entries in previous level
                    int index = vecindex*anglenumvec[i]+dirindex;
                    thrust::fill(thrust::device,
                                 thrust::make_zip_iterator(thrust::make_tuple(Ex_d0.begin()+index, 
                                 Ey_d0.begin()+index, Ez_d0.begin()+index, 
                                 Hx_d0.begin()+index, Hy_d0.begin()+index, Hz_d0.begin()+index)),
                                 thrust::make_zip_iterator(thrust::make_tuple(Ex_d0.begin()+index+1, 
                                 Ey_d0.begin()+index+1, Ez_d0.begin()+index+1, 
                                 Hx_d0.begin()+index+1, Hy_d0.begin()+index+1, Hz_d0.begin()+index+1)),
                                 field);
                  }
                }
              }
            }else{

              if ((max_depth-i)%2==1){
                // zero out field_all_dev2
                thrust::fill(Ex_d1.begin(), Ex_d1.end(), zerofloat2);
                thrust::fill(Ey_d1.begin(), Ey_d1.end(), zerofloat2);
                thrust::fill(Ez_d1.begin(), Ez_d1.end(), zerofloat2);
                thrust::fill(Hx_d1.begin(), Hx_d1.end(), zerofloat2);
                thrust::fill(Hy_d1.begin(), Hy_d1.end(), zerofloat2);
                thrust::fill(Hz_d1.begin(), Hz_d1.end(), zerofloat2);

                // non leaf level
                int numdirnextlevel = anglenumvec[i+1];
                int numdirthislevel = anglenumvec[i]; // #dirthislevel should be greater than #dirnextlevel
                int numdirsofar=0;
                for (int levelindex=0; levelindex<i;++levelindex){
                  numdirsofar+=anglenumvec[levelindex];
                }
                int leveloffset = 8 * numdirsofar;

                // loop through active nodes
                int curactivenum;
                if (i>0)
                  curactivenum = nonemptynodes[max_depth-i].size();
                else
                  curactivenum = 1; // top level

                for (int vecindex = 0; vecindex < curactivenum; ++vecindex){
                  int nodeindex;
                  if (i>0)
                    nodeindex= nonemptynodes[max_depth-i][vecindex];
                  else
                    nodeindex = 0;

                  int parentstart = vecindex*numdirthislevel;

                  // loop through 8 children and find active ones to do interp, translation and sum
                  int numactivechildren = 0;

                  for (int childindex = 0; childindex < 8; ++childindex){
                    int truechildindex = nodeindexend+(nodeindex-nodeindexstart)*8+childindex;

                    // for each child index, search whether it is in the next level active list
                    std::vector<int> activechildlist = nonemptynodes[max_depth-i-1];
                    for (int searchindex = 0; searchindex < activechildlist.size(); ++searchindex){

                      if (activechildlist[searchindex] == truechildindex){
                        // this child is active!!
                        int index = searchindex;   // index in the active child list is the index we need to copy the fields to tmp1 vector

                        copyandinterp(Ex_d0, index, numdirnextlevel, tmp1_ex, tmp2, numdirthislevel,
                                      plan_phi[i], plan_inverse_phi[i], plan_theta[i], plan_inverse_theta[i],
                                      phionumvec[i+1], phionumvec[i], thetaonumvec[i+1], thetaonumvec[i],
                                      thetaonumvec[i+1], phionumvec[i]);

                        copyandinterp(Ey_d0, index, numdirnextlevel, tmp1_ey, tmp2, numdirthislevel,
                                      plan_phi[i], plan_inverse_phi[i], plan_theta[i], plan_inverse_theta[i],
                                      phionumvec[i+1], phionumvec[i], thetaonumvec[i+1], thetaonumvec[i],
                                      thetaonumvec[i+1], phionumvec[i]);

                        copyandinterp(Ez_d0, index, numdirnextlevel, tmp1_ez, tmp2, numdirthislevel,
                                      plan_phi[i], plan_inverse_phi[i], plan_theta[i], plan_inverse_theta[i],
                                      phionumvec[i+1], phionumvec[i], thetaonumvec[i+1], thetaonumvec[i],
                                      thetaonumvec[i+1], phionumvec[i]);

                        copyandinterp(Hx_d0, index, numdirnextlevel, tmp1_hx, tmp2, numdirthislevel,
                                      plan_phi[i], plan_inverse_phi[i], plan_theta[i], plan_inverse_theta[i],
                                      phionumvec[i+1], phionumvec[i], thetaonumvec[i+1], thetaonumvec[i],
                                      thetaonumvec[i+1], phionumvec[i]);

                        copyandinterp(Hy_d0, index, numdirnextlevel, tmp1_hy, tmp2, numdirthislevel,
                                      plan_phi[i], plan_inverse_phi[i], plan_theta[i], plan_inverse_theta[i],
                                      phionumvec[i+1], phionumvec[i], thetaonumvec[i+1], thetaonumvec[i],
                                      thetaonumvec[i+1], phionumvec[i]);

                        copyandinterp(Hz_d0, index, numdirnextlevel, tmp1_hz, tmp2, numdirthislevel,
                                      plan_phi[i], plan_inverse_phi[i], plan_theta[i], plan_inverse_theta[i],
                                      phionumvec[i+1], phionumvec[i], thetaonumvec[i+1], thetaonumvec[i],
                                      thetaonumvec[i+1], phionumvec[i]);

                        // translate this child
                        int offset = leveloffset + childindex*numdirthislevel;
                        translation_compute(tmp1_ex, tmp1_ey, tmp1_ez, tmp1_hx, tmp1_hy, tmp1_hz, factor1_d, 0, numdirthislevel, offset);

                        int numberblock = std::ceil(numdirthislevel / NUM_THREADS_PER_BLOCK)+1;
                        sumchildren<<<numberblock, NUM_THREADS_PER_BLOCK>>>
                          (thrust::raw_pointer_cast(tmp1_ex.data()),
                           thrust::raw_pointer_cast(tmp1_ey.data()),
                           thrust::raw_pointer_cast(tmp1_ez.data()),
                           thrust::raw_pointer_cast(tmp1_hx.data()),
                           thrust::raw_pointer_cast(tmp1_hy.data()),
                           thrust::raw_pointer_cast(tmp1_hz.data()),
                           thrust::raw_pointer_cast(Ex_d1.data())+parentstart,
                           thrust::raw_pointer_cast(Ey_d1.data())+parentstart,
                           thrust::raw_pointer_cast(Ez_d1.data())+parentstart,
                           thrust::raw_pointer_cast(Hx_d1.data())+parentstart,
                           thrust::raw_pointer_cast(Hy_d1.data())+parentstart,
                           thrust::raw_pointer_cast(Hz_d1.data())+parentstart,
                           numdirthislevel);

                        numactivechildren++;

                        break; // end the search

                      } // if child is active

                    } // searching in the active child list

                  } // loop through 8 children

                }// loop through active nodes on this level

              }else{
                // zero out field_all_dev2
                thrust::fill(Ex_d0.begin(), Ex_d0.end(), zerofloat2);
                thrust::fill(Ey_d0.begin(), Ey_d0.end(), zerofloat2);
                thrust::fill(Ez_d0.begin(), Ez_d0.end(), zerofloat2);
                thrust::fill(Hx_d0.begin(), Hx_d0.end(), zerofloat2);
                thrust::fill(Hy_d0.begin(), Hy_d0.end(), zerofloat2);
                thrust::fill(Hz_d0.begin(), Hz_d0.end(), zerofloat2);

                // non leaf level
                int numdirnextlevel = anglenumvec[i+1];
                int numdirthislevel = anglenumvec[i]; // #dirthislevel should be greater than #dirnextlevel
                int numdirsofar=0;
                for (int levelindex=0; levelindex<i;++levelindex){
                  numdirsofar+=anglenumvec[levelindex];
                  std::cout<<"levelindex "<<levelindex<<" anglenumvec[levelindex] "<<anglenumvec[levelindex]<<std::endl;
                }
                std::cout<<"numdirsofar "<<numdirsofar<<std::endl;
                int leveloffset = 8 * numdirsofar;

                // loop through active nodes
                int curactivenum;
                if (i>0)
                  curactivenum = nonemptynodes[max_depth-i].size();
                else
                  curactivenum = 1; // top level

                for (int vecindex = 0; vecindex < curactivenum; ++vecindex){
                  int nodeindex;
                  if (i>0)
                    nodeindex= nonemptynodes[max_depth-i][vecindex];
                  else
                    nodeindex = 0;

                  int parentstart = vecindex*numdirthislevel;


                  // loop through 8 children and find active ones to do interp, translation and sum
                  int numactivechildren = 0;

                  for (int childindex = 0; childindex < 8; ++childindex){
                    int truechildindex = nodeindexend+(nodeindex-nodeindexstart)*8+childindex;

                    // for each child index, search whether it is in the next level active list
                    std::vector<int> activechildlist = nonemptynodes[max_depth-i-1];
                    for (int searchindex = 0; searchindex < activechildlist.size(); ++searchindex){

                      if (activechildlist[searchindex] == truechildindex){
                        // this child is active!!
                        int index = searchindex;             // index in the active child list is the index we need to copy the fields to tmp1 vector

                        copyandinterp(Ex_d1, index, numdirnextlevel, tmp1_ex, tmp2, numdirthislevel,
                                      plan_phi[i], plan_inverse_phi[i], plan_theta[i], plan_inverse_theta[i],
                                      phionumvec[i+1], phionumvec[i], thetaonumvec[i+1], thetaonumvec[i],
                                      thetaonumvec[i+1], phionumvec[i]);

                        copyandinterp(Ey_d1, index, numdirnextlevel, tmp1_ey, tmp2, numdirthislevel,
                                      plan_phi[i], plan_inverse_phi[i], plan_theta[i], plan_inverse_theta[i],
                                      phionumvec[i+1], phionumvec[i], thetaonumvec[i+1], thetaonumvec[i],
                                      thetaonumvec[i+1], phionumvec[i]);

                        copyandinterp(Ez_d1, index, numdirnextlevel, tmp1_ez, tmp2, numdirthislevel,
                                      plan_phi[i], plan_inverse_phi[i], plan_theta[i], plan_inverse_theta[i],
                                      phionumvec[i+1], phionumvec[i], thetaonumvec[i+1], thetaonumvec[i],
                                      thetaonumvec[i+1], phionumvec[i]);

                        copyandinterp(Hx_d1, index, numdirnextlevel, tmp1_hx, tmp2, numdirthislevel,
                                      plan_phi[i], plan_inverse_phi[i], plan_theta[i], plan_inverse_theta[i],
                                      phionumvec[i+1], phionumvec[i], thetaonumvec[i+1], thetaonumvec[i],
                                      thetaonumvec[i+1], phionumvec[i]);

                        copyandinterp(Hy_d1, index, numdirnextlevel, tmp1_hy, tmp2, numdirthislevel,
                                      plan_phi[i], plan_inverse_phi[i], plan_theta[i], plan_inverse_theta[i],
                                      phionumvec[i+1], phionumvec[i], thetaonumvec[i+1], thetaonumvec[i],
                                      thetaonumvec[i+1], phionumvec[i]);

                        copyandinterp(Hz_d1, index, numdirnextlevel, tmp1_hz, tmp2, numdirthislevel,
                                      plan_phi[i], plan_inverse_phi[i], plan_theta[i], plan_inverse_theta[i],
                                      phionumvec[i+1], phionumvec[i], thetaonumvec[i+1], thetaonumvec[i],
                                      thetaonumvec[i+1], phionumvec[i]);

                        // translate this child
                        int offset = leveloffset + childindex*numdirthislevel;
                        translation_compute(tmp1_ex, tmp1_ey, tmp1_ez, tmp1_hx, tmp1_hy, tmp1_hz, factor1_d, 0, numdirthislevel, offset);

                        int numberblock = std::ceil(numdirthislevel / NUM_THREADS_PER_BLOCK)+1;
                        sumchildren<<<numberblock, NUM_THREADS_PER_BLOCK>>>
                          (thrust::raw_pointer_cast(tmp1_ex.data()),
                           thrust::raw_pointer_cast(tmp1_ey.data()),
                           thrust::raw_pointer_cast(tmp1_ez.data()),
                           thrust::raw_pointer_cast(tmp1_hx.data()),
                           thrust::raw_pointer_cast(tmp1_hy.data()),
                           thrust::raw_pointer_cast(tmp1_hz.data()),
                           thrust::raw_pointer_cast(Ex_d0.data())+parentstart,
                           thrust::raw_pointer_cast(Ey_d0.data())+parentstart,
                           thrust::raw_pointer_cast(Ez_d0.data())+parentstart,
                           thrust::raw_pointer_cast(Hx_d0.data())+parentstart,
                           thrust::raw_pointer_cast(Hy_d0.data())+parentstart,
                           thrust::raw_pointer_cast(Hz_d0.data())+parentstart,
                           numdirthislevel);

                        numactivechildren++;

                        break; // end the search

                      } // if child is active

                    } // searching in the active child list

                  } // loop through 8 children

                }// loop through active nodes on this level

              }// even or odd?

            } // non leaf level

            // reduce to final sum
            if (i==0){

              // compute scattering on gpu
              int numblock = std::ceil((float)obnum/NUM_THREADS_PER_BLOCK)+1;
              std::cout<<"numblock "<<numblock<<std::endl;

              if ((max_depth-i)%2==1){
                // projection is included here
                computeintensity<<<numblock,
                  NUM_THREADS_PER_BLOCK>>>(thrust::raw_pointer_cast(scattering.data()),
                                           obnum,
                                           thrust::raw_pointer_cast(Ex_d1.data()),
                                           thrust::raw_pointer_cast(Ey_d1.data()),
                                           thrust::raw_pointer_cast(Ez_d1.data()),
                                           thrust::raw_pointer_cast(Hx_d1.data()),
                                           thrust::raw_pointer_cast(Hy_d1.data()),
                                           thrust::raw_pointer_cast(Hz_d1.data()),
                                           thrust::raw_pointer_cast(dirvec_d.data()),
                                           Z0);

              }else{

                // projection is included here
                computeintensity<<<numblock,
                  NUM_THREADS_PER_BLOCK>>>(thrust::raw_pointer_cast(scattering.data()),
                                           obnum,
                                           thrust::raw_pointer_cast(Ex_d0.data()),
                                           thrust::raw_pointer_cast(Ey_d0.data()),
                                           thrust::raw_pointer_cast(Ez_d0.data()),
                                           thrust::raw_pointer_cast(Hx_d0.data()),
                                           thrust::raw_pointer_cast(Hy_d0.data()),
                                           thrust::raw_pointer_cast(Hz_d0.data()),
                                           thrust::raw_pointer_cast(dirvec_d.data()),
                                           Z0);
              } // check even or odd when compute scattering function

              // std::cout<<"copy scattering"<<std::endl;
              thrust::copy(scattering.begin(), scattering.end(), scattering_h.begin());
              // std::cout<<"write out"<<std::endl;
              std::string filename = outputdir + "/fiber_"+std::to_string(fiberindex)+"_lambda"+std::to_string(lambdaindex)+"_T"+mode+"_depth"+std::to_string(max_depth)+".binary";
              std::ofstream out(filename, std::ios::out|std::ios::binary);
              out.write((char *) &scattering_h[0], sizeof(float)*obnum);

            } // top level of the tree

          } // loop through all levels

        } // polarization

      } // inicident phi

    } // incident theta

  } // wavelength

  // Free memory
  // delete cufft plans
  for (int i = 0; i < max_depth; ++i){
    checkCudaErrors(cufftDestroy(plan_phi[i]));
    checkCudaErrors(cufftDestroy(plan_inverse_phi[i]));
    checkCudaErrors(cufftDestroy(plan_theta[i]));
    checkCudaErrors(cufftDestroy(plan_inverse_theta[i]));
  }

  delete[] surfaceheight;
  checkCudaErrors(cudaFree(nodes));
  checkCudaErrors(cudaFree(points));
  checkCudaErrors(cudaFree(normals));
  checkCudaErrors(cudaFree(areavec));

  gettimeofday(&end, NULL);
  delta = ((end.tv_sec  - start.tv_sec) * 1000000u +
           end.tv_usec - start.tv_usec) / 1.e6;
  std::cout<<"total time used: "<<delta<<std::endl;

  return 0;
}
