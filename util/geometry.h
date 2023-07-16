// only given height field
// input hair geometry
void set_sourcepoint(float radius1, float radius2, float period, float unitz,
                     hostFvec& sourcexvec, hostFvec& sourceyvec, hostFvec& sourcezvec,
                     hostFvec& unitareavec, hostFvec &normalxvec, hostFvec &normalyvec, hostFvec &normalzvec, 
                     int numel, int numelvert, float *surfaceheight)
{
  float x, y, z, znext;
  float phiunit = 2 * M_PI / numel;
  Float3 p1(3), p2(3), p3(3), p4(3), areavec1(3), areavec2(3), areavec(3);
  int index;

  for (int i = 0; i < numelvert; ++i){
    z = i * unitz - period*0.5;
    znext = (i+1)*unitz - period*0.5;

    for (int j = 0; j < numel; ++j){
      index = i*numel+j;

      int planeindex, zindex;
      if (j<numel-1)
        planeindex=j+1;
      else
        planeindex = 0;
      if (i<numelvert-1)
        zindex = i+1;
      else
        zindex = i;

      x = surfaceheight[j*numelvert+i];
      y = surfaceheight[numelvert*numel+j*numelvert+i];

      // source point is the average of the four vertices
      thrust::get<0>(p1) = x;
      thrust::get<1>(p1) = y;
      thrust::get<2>(p1) = z;

      x = surfaceheight[planeindex*numelvert+i];
      y = surfaceheight[numelvert*numel+planeindex*numelvert+i];

      thrust::get<0>(p2) = x;
      thrust::get<1>(p2) = y;
      thrust::get<2>(p2) = z;

      x = surfaceheight[j*numelvert+zindex];
      y = surfaceheight[numelvert*numel+j*numelvert+zindex];
      thrust::get<0>(p3) = x;
      thrust::get<1>(p3) = y;
      thrust::get<2>(p3) = znext;

      x = surfaceheight[planeindex*numelvert+zindex];
      y = surfaceheight[numelvert*numel+planeindex*numelvert+zindex];
      thrust::get<0>(p4) = x;
      thrust::get<1>(p4) = y;
      thrust::get<2>(p4) = znext;

      sourcexvec[index] = (thrust::get<0>(p1)+thrust::get<0>(p2)+thrust::get<0>(p3)+thrust::get<0>(p4))/4.0;
      sourceyvec[index] = (thrust::get<1>(p1)+thrust::get<1>(p2)+thrust::get<1>(p3)+thrust::get<1>(p4))/4.0;
      sourcezvec[index] = (thrust::get<2>(p1)+thrust::get<2>(p2)+thrust::get<2>(p3)+thrust::get<2>(p4))/4.0;

      areavec1 = crossproduct(subtract(p2, p1), subtract(p3, p1));
      areavec2 = crossproduct(subtract(p4, p2), subtract(p3, p2));
      unitareavec[index] = 0.5*(vecnorm(areavec1)+vecnorm(areavec2));

      areavec = normalize(add(areavec1, areavec2));

      normalxvec[index] = thrust::get<0>(areavec);
      normalyvec[index] = thrust::get<1>(areavec);
      normalzvec[index] = thrust::get<2>(areavec);
    }
  }
}


// only given height field
// input hair geometry
// only assign half of the source points
 void set_sourcepoint(float radius1, float radius2, float period, float unitz,
                      hostFvec& sourcexvec, hostFvec& sourceyvec, hostFvec& sourcezvec,
                      hostFvec& unitareavec, hostFvec &normalxvec, hostFvec &normalyvec, 
                      hostFvec &normalzvec, int numel, int numelvert, float *surfaceheight, 
                      int startindex, int endindex)
 {
   float x, y, z, znext;
   float phiunit = 2 * M_PI / numel;
   Float3 p1(3), p2(3), p3(3), p4(3), areavec1(3), areavec2(3), areavec(3);
   int index;

   for (int i = 0; i < numelvert; ++i){
     //for (int i = 0; i < 1; ++i){
     z = i * unitz - period*0.5;
     znext = (i+1)*unitz - period*0.5;

     for (int j = 0; j < numel; ++j){
       if (j < startindex || j > endindex)
         continue;     

       int planeindex, zindex;
       if (j<numel-1)
         planeindex=j+1;
       else
         planeindex = 0;
       if (i<numelvert-1)
         zindex = i+1;
       else
         zindex = i;

       x = surfaceheight[j*numelvert+i];
       y = surfaceheight[numelvert*numel+j*numelvert+i];

       if (x==0 && y==0){
         std::cout<<"error: surfaceheight is zero, p1"<<std::endl;
         std::cout<<"j: "<<j<<", i: "<<i<<std::endl;
         std::cout<<"x: "<<x<<", y: "<<y<<std::endl;
         exit(1);
       }

       // source point is the average of the four vertices
       thrust::get<0>(p1) = x;
       thrust::get<1>(p1) = y;
       thrust::get<2>(p1) = z;

       x = surfaceheight[planeindex*numelvert+i];
       y = surfaceheight[numelvert*numel+planeindex*numelvert+i];

       if (x==0 && y==0){
         std::cout<<"error: surfaceheight is zero, p2"<<std::endl;
         std::cout<<"j: "<<j<<", i: "<<i<<std::endl;
         std::cout<<"x: "<<x<<", y: "<<y<<std::endl;
         exit(1);
       }

       thrust::get<0>(p2) = x;
       thrust::get<1>(p2) = y;
       thrust::get<2>(p2) = z;

       x = surfaceheight[j*numelvert+zindex];
       y = surfaceheight[numelvert*numel+j*numelvert+zindex];
       thrust::get<0>(p3) = x;
       thrust::get<1>(p3) = y;
       thrust::get<2>(p3) = znext;

       if (x==0 &&y==0){
         std::cout<<"error: surfaceheight is zero, p3"<<std::endl;
         std::cout<<"j: "<<j<<", i: "<<i<<std::endl;
         std::cout<<"x: "<<x<<", y: "<<y<<std::endl;
         exit(1);
       }

       x = surfaceheight[planeindex*numelvert+zindex];
       y = surfaceheight[numelvert*numel+planeindex*numelvert+zindex];
       thrust::get<0>(p4) = x;
       thrust::get<1>(p4) = y;
       thrust::get<2>(p4) = znext;

       if (x==0 &&y==0){
         std::cout<<"error: surfaceheight is zero, p4"<<std::endl;
         std::cout<<"j: "<<j<<", i: "<<i<<std::endl;
         std::cout<<"x: "<<x<<", y: "<<y<<std::endl;
         exit(1);
       }
       index = i*(endindex-startindex+1)+j-startindex;

       sourcexvec[index] = (thrust::get<0>(p1)+thrust::get<0>(p2)+thrust::get<0>(p3)+thrust::get<0>(p4))/4.0;
       sourceyvec[index] = (thrust::get<1>(p1)+thrust::get<1>(p2)+thrust::get<1>(p3)+thrust::get<1>(p4))/4.0;
       sourcezvec[index] = (thrust::get<2>(p1)+thrust::get<2>(p2)+thrust::get<2>(p3)+thrust::get<2>(p4))/4.0;

       areavec1 = crossproduct(subtract(p2, p1), subtract(p3, p1));
       areavec2 = crossproduct(subtract(p4, p2), subtract(p3, p2));
       unitareavec[index] = 0.5*(vecnorm(areavec1)+vecnorm(areavec2));

       areavec = normalize(add(areavec1, areavec2));

       normalxvec[index] = thrust::get<0>(areavec);
       normalyvec[index] = thrust::get<1>(areavec);
       normalzvec[index] = thrust::get<2>(areavec);
     }
   }
 }