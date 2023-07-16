// Build a octree on the GPU. Use CUDA Dynamic Parallelism.
// The code is modified from CUDA's quadtree example.
// The copyright of the original quadtree code is as follows:

/* Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


namespace cg = cooperative_groups;

////////////////////////////////////////////////////////////////////////////////
// A structure of the area assoiciated with each point (structure of arrays).
////////////////////////////////////////////////////////////////////////////////
class Area {
public:
  float *m_x;

  // Constructor.
  __host__ __device__ Area() : m_x(NULL){}

  // Constructor.
  __host__ __device__ Area(float *x) : m_x(x){}

  // Get a point.
  __host__ __device__ __forceinline__ float get_area(int idx) const {
    return m_x[idx];
  }

  // Set a point.
  __host__ __device__ __forceinline__ void set_area(int idx, float x) {
    m_x[idx] = x;
  }

  // Set the pointers.
  __host__ __device__ __forceinline__ void set(float *x) {
    m_x = x;
  }
};

////////////////////////////////////////////////////////////////////////////////
// A structure of 3D points (structure of arrays).
////////////////////////////////////////////////////////////////////////////////
class Points {
 public:
  float *m_x;
  float *m_y;
  float *m_z;

  // Constructor.
  __host__ __device__ Points() : m_x(NULL), m_y(NULL), m_z(NULL) {}

  // Constructor.
  __host__ __device__ Points(float *x, float *y, float *z) : m_x(x), m_y(y), m_z(z) {}

  // Get a point.
  __host__ __device__ __forceinline__ float3 get_point(int idx) const {
    return make_float3(m_x[idx], m_y[idx], m_z[idx]);
  }

  // Set a point.
  __host__ __device__ __forceinline__ void set_point(int idx, const float3 &p) {
    m_x[idx] = p.x;
    m_y[idx] = p.y;
    m_z[idx] = p.z;
  }

  // Set the pointers.
  __host__ __device__ __forceinline__ void set(float *x, float *y, float *z) {
    m_x = x;
    m_y = y;
    m_z = z;
  }
};

////////////////////////////////////////////////////////////////////////////////
// A 3D bounding box
////////////////////////////////////////////////////////////////////////////////
class Bounding_box {
 public:
  // Extreme points of the bounding box.
  float3 m_p_min;
  float3 m_p_max;
  //float3 center;

  // Constructor. Create a unit box.
  __host__ __device__ Bounding_box() {
    m_p_min = make_float3(0.0f, 0.0f, 0.0f);
    m_p_max = make_float3(1.0f, 1.0f, 1.0f);
  }

  //Compute the center of the bounding-box.
  __host__ __device__ void compute_center(float3 &_center) const {
    _center.x = 0.5f * (m_p_min.x + m_p_max.x);
    _center.y = 0.5f * (m_p_min.y + m_p_max.y);
    _center.z = 0.5f * (m_p_min.z + m_p_max.z);
  }

  // The points of the box.
  __host__ __device__ __forceinline__ const float3 &get_max() const {
    return m_p_max;
  }

  __host__ __device__ __forceinline__ const float3 &get_min() const {
    return m_p_min;
  }

  // Does a box contain a point.
  __host__ __device__ bool contains(const float3 &p) const {
    return p.x >= m_p_min.x && p.x < m_p_max.x && p.y >= m_p_min.y &&
      p.y < m_p_max.y  && p.z >= m_p_min.z &&
      p.z < m_p_max.z;
  }

  // Define the bounding box.
  __host__ __device__ void set(float min_x, float min_y, float min_z, float max_x,
                               float max_y, float max_z) {
    m_p_min.x = min_x;
    m_p_min.y = min_y;
    m_p_min.z = min_z;
    m_p_max.x = max_x;
    m_p_max.y = max_y;
    m_p_max.z = max_z;
  }
};

////////////////////////////////////////////////////////////////////////////////
// A node of a octree.
////////////////////////////////////////////////////////////////////////////////
class Octree_node {
 public:
  // The identifier of the node.
  int m_id;
  // The bounding box of the tree.
  Bounding_box m_bounding_box;
  // The range of points.
  int m_begin, m_end;
  bool active;

  // Constructor.
  __host__ __device__ Octree_node() : m_id(0), m_begin(0), m_end(0), active(1) {}

  // The ID of a node at its level.
  __host__ __device__ int id() const { return m_id; }

  // The ID of a node at its level.
  __host__ __device__ void set_id(int new_id) { m_id = new_id; }

  // The bounding box.
  __host__ __device__ __forceinline__ const Bounding_box &bounding_box() const {
    return m_bounding_box;
  }

  // Set the bounding box.
  __host__ __device__ __forceinline__ void set_bounding_box(float min_x,
                                                            float min_y,
                                                            float min_z,
                                                            float max_x,
                                                            float max_y,
                                                            float max_z) {
    m_bounding_box.set(min_x, min_y, min_z, max_x, max_y, max_z);
  }

  // The number of points in the tree.
  __host__ __device__ __forceinline__ int num_points() const {
    return m_end - m_begin;
  }

  // The range of points in the tree.
  __host__ __device__ __forceinline__ int points_begin() const {
    return m_begin;
  }

  __host__ __device__ __forceinline__ int points_end() const { return m_end; }

  // Define the range for that node.
  __host__ __device__ __forceinline__ void set_range(int begin, int end) {
    m_begin = begin;
    m_end = end;
  }
};

////////////////////////////////////////////////////////////////////////////////
// Algorithm parameters.
////////////////////////////////////////////////////////////////////////////////
struct Parameters {
  // Choose the right set of points to use as in/out.
  int point_selector;
  // The number of nodes at a given level (8^k for level k).
  int num_nodes_at_this_level;
  // The recursion depth.
  int depth;
  // The max value for depth.
  const int max_depth;
  // The minimum number of points in a node to stop recursion.
  const int min_points_per_node;

  // Constructor set to default values.
  __host__ __device__ Parameters(int max_depth, int min_points_per_node)
      : point_selector(0),
        num_nodes_at_this_level(1),
        depth(0),
        max_depth(max_depth),
        min_points_per_node(min_points_per_node) {}

  // Copy constructor. Changes the values for next iteration.
  __host__ __device__ Parameters(const Parameters &params, bool)
      : point_selector((params.point_selector + 1) % 2),
        num_nodes_at_this_level(8 * params.num_nodes_at_this_level),
        depth(params.depth + 1),
        max_depth(params.max_depth),
        min_points_per_node(params.min_points_per_node) {}
};



////////////////////////////////////////////////////////////////////////////////
//
// The algorithm works as follows. The host (CPU) launches one block of
// NUM_THREADS_PER_BLOCK threads. That block will do the following steps:
//
// 1- Check the number of points and its depth.
//
// We impose a maximum depth to the tree and a minimum number of points per
// node. If the maximum depth is exceeded, the threads in the block exit.
//
// Before exiting, they perform a buffer swap if it is needed. Indeed, the
// algorithm uses two buffers to permute the points and make sure they are
// properly distributed in the octree. By design we want all points to be
// in the first buffer of points at the end of the algorithm. It is the reason
// why we may have to swap the buffer before leaving (if the points are in the
// 2nd buffer).
//
// 2- Count the number of points in each child.
//
// If the depth is not too high and the number of points is sufficient, the
// block has to dispatch the points into eight geometrical buckets: Its
// children. For that purpose, we compute the center of the bounding box and
// count the number of points in each octant.
//
// The set of points is divided into sections. Each section is given to a
// warp of threads (32 threads). Warps use __ballot and __popc intrinsics
// to count the points. See the Programming Guide for more information about
// those functions.
//
// 3- Scan the warps' results to know the "global" numbers.
//
// Warps work independently from each other. At the end, each warp knows the
// number of points in its section. To know the numbers for the block, the
// block has to run a scan/reduce at the block level. It's a traditional
// approach. The implementation in the sample is not as optimized as what
// could be found in fast radix sorts, for example, but it relies on the same
// idea.
//
// 4- Move points.
//
// Now that the block knows how many points go in each of its 8 children and 
// will dispatch the points. 
//
// 5- Launch new blocks.
//
// The block launches eight new blocks: one per child. Each of the eight blocks
// will apply the same algorithm.
////////////////////////////////////////////////////////////////////////////////

template <int NUM_THREADS_PER_BLOCK>
__global__ void build_octree_kernel(Octree_node *nodes, Points *points, Points *normals, Area *unitareavec, Parameters params) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  // The number of warps in a block.
  const int NUM_WARPS_PER_BLOCK = NUM_THREADS_PER_BLOCK / warpSize;

  // Shared memory to store the number of points.
  extern __shared__ int smem[];

  // s_num_pts[8][NUM_WARPS_PER_BLOCK];
  // Addresses of shared memory.
  volatile int *s_num_pts[8];

  for (int i = 0; i < 8; ++i)
    s_num_pts[i] = (volatile int *)&smem[i * NUM_WARPS_PER_BLOCK];

  // Compute the coordinates of the threads in the block.
  const int warp_id = threadIdx.x / warpSize;
  const int lane_id = threadIdx.x % warpSize;

  // Mask for compaction.
  // Same as: asm( "mov.u32 %0, %%lanemask_lt;" : "=r"(lane_mask_lt) );  ????
  int lane_mask_lt = (1 << lane_id) - 1;

  // The current node.
  Octree_node &node = nodes[blockIdx.x];

  // The number of points in the node.
  int num_points = node.num_points();

  float3 center;
  int range_begin, range_end;
  int warp_cnts[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  //
  // 1- Check the number of points and its depth.
  //

  // Stop the recursion here. Make sure points[0] contains all the points.
  // if (params.depth >= params.max_depth ||
  //     num_points <= params.min_points_per_node) {

  if (params.depth >= params.max_depth) {
    if (params.point_selector == 1) {
      int it = node.points_begin(), end = node.points_end();

      for (it += threadIdx.x; it < end; it += NUM_THREADS_PER_BLOCK){
        if (it < end){
          points[0].set_point(it, points[1].get_point(it));
          normals[0].set_point(it, normals[1].get_point(it));
          unitareavec[0].set_area(it, unitareavec[1].get_area(it));
        }
      }
    }
    return;
  }

  // Compute the center of the bounding box of the points.
  const Bounding_box &bbox = node.bounding_box();
  bbox.compute_center(center);
  //center = bbox.get_center();

  // Find how many points to give to each warp.
  // check this number
  int num_points_per_warp = max(
      warpSize, (num_points + NUM_WARPS_PER_BLOCK - 1) / NUM_WARPS_PER_BLOCK);

  // Each warp of threads will compute the number of points to move to each
  // octant.
  range_begin = node.points_begin() + warp_id * num_points_per_warp;
  range_end = min(range_begin + num_points_per_warp, node.points_end());

  //
  // 2- Count the number of points in each child.
  //

  // Input points.
  const Points &in_points = points[params.point_selector];

  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);
  // Compute the number of points.
  for (int range_it = range_begin + tile32.thread_rank();
       tile32.any(range_it < range_end); range_it += warpSize) {
    // Is it still an active thread?
    bool is_active = range_it < range_end;

    // Load the coordinates of the point.
    float3 p = is_active ? in_points.get_point(range_it) : make_float3(0.0f, 0.0f, 0.0f);

    // Count top-left points in the first level.
    int num_pts = __popc(tile32.ballot(is_active && p.x < center.x && p.y >= center.y && p.z >= center.z));
    warp_cnts[0] += tile32.shfl(num_pts, 0);

    // Count top-right points in the first level.
    num_pts =
        __popc(tile32.ballot(is_active && p.x >= center.x && p.y >= center.y && p.z >= center.z));
    warp_cnts[1] += tile32.shfl(num_pts, 0);

    // Count bottom-left points in the first level.
    num_pts =
      __popc(tile32.ballot(is_active && p.x < center.x && p.y < center.y && p.z >= center.z));
    warp_cnts[2] += tile32.shfl(num_pts, 0);

    // Count bottom-right points in the first level.
    num_pts =
      __popc(tile32.ballot(is_active && p.x >= center.x && p.y < center.y && p.z >= center.z));
    warp_cnts[3] += tile32.shfl(num_pts, 0);

    // Count top-left points in the second level.
    num_pts = __popc(tile32.ballot(is_active && p.x < center.x && p.y >= center.y && p.z < center.z));
    warp_cnts[4] += tile32.shfl(num_pts, 0);

    // Count top-right points in the second level.
    num_pts =
      __popc(tile32.ballot(is_active && p.x >= center.x && p.y >= center.y && p.z < center.z));
    warp_cnts[5] += tile32.shfl(num_pts, 0);

    // Count bottom-left points in the second level.
    num_pts =
      __popc(tile32.ballot(is_active && p.x < center.x && p.y < center.y && p.z < center.z));
    warp_cnts[6] += tile32.shfl(num_pts, 0);

    // Count bottom-right points in the second level.
    num_pts =
      __popc(tile32.ballot(is_active && p.x >= center.x && p.y < center.y && p.z < center.z));
    warp_cnts[7] += tile32.shfl(num_pts, 0);
  }

  if (tile32.thread_rank() == 0) {
    s_num_pts[0][warp_id] = warp_cnts[0];
    s_num_pts[1][warp_id] = warp_cnts[1];
    s_num_pts[2][warp_id] = warp_cnts[2];
    s_num_pts[3][warp_id] = warp_cnts[3];
    s_num_pts[4][warp_id] = warp_cnts[4];
    s_num_pts[5][warp_id] = warp_cnts[5];
    s_num_pts[6][warp_id] = warp_cnts[6];
    s_num_pts[7][warp_id] = warp_cnts[7];
  }

  // Make sure warps have finished counting.
  cg::sync(cta);

  //
  // 3- Scan the warps' results to know the "global" numbers.
  //

  // First 8 warps scan the numbers of points per child (inclusive scan).
  if (warp_id < 8) {
    int num_pts = tile32.thread_rank() < NUM_WARPS_PER_BLOCK
                      ? s_num_pts[warp_id][tile32.thread_rank()]
                      : 0;
#pragma unroll

    for (int offset = 1; offset < NUM_WARPS_PER_BLOCK; offset *= 2) {
      int n = tile32.shfl_up(num_pts, offset);

      if (tile32.thread_rank() >= offset) num_pts += n;
    }

    if (tile32.thread_rank() < NUM_WARPS_PER_BLOCK)
      s_num_pts[warp_id][tile32.thread_rank()] = num_pts;
  }

  cg::sync(cta);

  // Compute global offsets.
  if (warp_id == 0) {
    int sum = s_num_pts[0][NUM_WARPS_PER_BLOCK - 1];

    for (int row = 1; row < 8; ++row) {
      int tmp = s_num_pts[row][NUM_WARPS_PER_BLOCK - 1];
      cg::sync(tile32);

      if (tile32.thread_rank() < NUM_WARPS_PER_BLOCK)
        s_num_pts[row][tile32.thread_rank()] += sum;

      cg::sync(tile32);
      sum += tmp;
    }
  }

  cg::sync(cta);

  // Make the scan exclusive.
  int val = 0;
  if (threadIdx.x < 8 * NUM_WARPS_PER_BLOCK) {
    val = threadIdx.x == 0 ? 0 : smem[threadIdx.x - 1];
    val += node.points_begin();
  }

  cg::sync(cta);

  if (threadIdx.x < 8 * NUM_WARPS_PER_BLOCK) {
    smem[threadIdx.x] = val;
  }

  cg::sync(cta);

  //
  // 4- Move points.
  //
  // if (!(params.depth >= params.max_depth ||
  //       num_points <= params.min_points_per_node)) {
  if (!(params.depth >= params.max_depth)) {
    // Output points.
    Points &out_points = points[(params.point_selector + 1) % 2];
    Points &out_normals = normals[(params.point_selector + 1) % 2];
    Area &out_unitarea = unitareavec[(params.point_selector + 1) % 2];

    warp_cnts[0] = s_num_pts[0][warp_id];
    warp_cnts[1] = s_num_pts[1][warp_id];
    warp_cnts[2] = s_num_pts[2][warp_id];
    warp_cnts[3] = s_num_pts[3][warp_id];
    warp_cnts[4] = s_num_pts[4][warp_id];
    warp_cnts[5] = s_num_pts[5][warp_id];
    warp_cnts[6] = s_num_pts[6][warp_id];
    warp_cnts[7] = s_num_pts[7][warp_id];

    const Points &in_points = points[params.point_selector];
    const Points &in_normals = normals[params.point_selector];
    const Area &in_unitarea = unitareavec[params.point_selector];
    // Reorder points.
    for (int range_it = range_begin + tile32.thread_rank();
         tile32.any(range_it < range_end); range_it += warpSize) {
      // Is it still an active thread?
      bool is_active = range_it < range_end;

      // Load the coordinates of the point.
      float3 p = is_active ? in_points.get_point(range_it) : make_float3(0.0f, 0.0f, 0.0f);
      float3 nor = is_active ? in_normals.get_point(range_it) : make_float3(0.0f, 0.0f, 0.0f);
      float unitarea = is_active ? in_unitarea.get_area(range_it) : 0.0f;

      // Count top-left points in the first level.
      bool pred = is_active && p.x < center.x && p.y >= center.y && p.z >= center.z;
      int vote = tile32.ballot(pred);
      int dest = warp_cnts[0] + __popc(vote & lane_mask_lt);

      if (pred){
        out_points.set_point(dest, p);
        out_normals.set_point(dest, nor);
        out_unitarea.set_area(dest, unitarea);
      }

      warp_cnts[0] += tile32.shfl(__popc(vote), 0);

      // Count top-right points in the first level.
      pred = is_active && p.x >= center.x && p.y >= center.y && p.z >= center.z;
      vote = tile32.ballot(pred);
      dest = warp_cnts[1] + __popc(vote & lane_mask_lt);

      if (pred){
        out_points.set_point(dest, p);
        out_normals.set_point(dest, nor);
        out_unitarea.set_area(dest, unitarea);
      }

      warp_cnts[1] += tile32.shfl(__popc(vote), 0);

      // Count bottom-left points in the first level.
      pred = is_active && p.x < center.x && p.y < center.y && p.z >= center.z;
      vote = tile32.ballot(pred);
      dest = warp_cnts[2] + __popc(vote & lane_mask_lt);

      if (pred){
        out_points.set_point(dest, p);
        out_normals.set_point(dest, nor);
        out_unitarea.set_area(dest, unitarea);
      }

      warp_cnts[2] += tile32.shfl(__popc(vote), 0);

      // Count bottom-right points in the first level.
      pred = is_active && p.x >= center.x && p.y < center.y && p.z >= center.z;
      vote = tile32.ballot(pred);
      dest = warp_cnts[3] + __popc(vote & lane_mask_lt);

      if (pred){
        out_points.set_point(dest, p);
        out_normals.set_point(dest, nor);
        out_unitarea.set_area(dest, unitarea);
      }

      warp_cnts[3] += tile32.shfl(__popc(vote), 0);

      // Count top-left points in the second level.
      pred = is_active && p.x < center.x && p.y >= center.y && p.z < center.z;
      vote = tile32.ballot(pred);
      dest = warp_cnts[4] + __popc(vote & lane_mask_lt);

      if (pred){
        out_points.set_point(dest, p);
        out_normals.set_point(dest, nor);
        out_unitarea.set_area(dest, unitarea);
      }

      warp_cnts[4] += tile32.shfl(__popc(vote), 0);

      // Count top-right points in the second level.
      pred = is_active && p.x >= center.x && p.y >= center.y && p.z < center.z;
      vote = tile32.ballot(pred);
      dest = warp_cnts[5] + __popc(vote & lane_mask_lt);

      if (pred){
        out_points.set_point(dest, p);
        out_normals.set_point(dest, nor);
        out_unitarea.set_area(dest, unitarea);
      }

      warp_cnts[5] += tile32.shfl(__popc(vote), 0);

      // Count bottom-left points in the second level.
      pred = is_active && p.x < center.x && p.y < center.y && p.z < center.z;
      vote = tile32.ballot(pred);
      dest = warp_cnts[6] + __popc(vote & lane_mask_lt);

      if (pred){
        out_points.set_point(dest, p);
        out_normals.set_point(dest, nor);
        out_unitarea.set_area(dest, unitarea);
      }

      warp_cnts[6] += tile32.shfl(__popc(vote), 0);

      // Count bottom-right points in the second level.
      pred = is_active && p.x >= center.x && p.y < center.y && p.z < center.z;
      vote = tile32.ballot(pred);
      dest = warp_cnts[7] + __popc(vote & lane_mask_lt);

      if (pred){
        out_points.set_point(dest, p);
        out_normals.set_point(dest, nor);
        out_unitarea.set_area(dest, unitarea);
      }

      warp_cnts[7] += tile32.shfl(__popc(vote), 0);
    }
  }

  cg::sync(cta);

  if (tile32.thread_rank() == 0) {
    s_num_pts[0][warp_id] = warp_cnts[0];
    s_num_pts[1][warp_id] = warp_cnts[1];
    s_num_pts[2][warp_id] = warp_cnts[2];
    s_num_pts[3][warp_id] = warp_cnts[3];
    s_num_pts[4][warp_id] = warp_cnts[4];
    s_num_pts[5][warp_id] = warp_cnts[5];
    s_num_pts[6][warp_id] = warp_cnts[6];
    s_num_pts[7][warp_id] = warp_cnts[7];
  }

  cg::sync(cta);

  //
  // 5- Launch new blocks.
  //
  if (!(params.depth >= params.max_depth)) {
    // The last thread launches new blocks.
    if (threadIdx.x == NUM_THREADS_PER_BLOCK - 1) {
      // The children.
      Octree_node *children =
          &nodes[params.num_nodes_at_this_level - (node.id() & ~7)];

      // The offsets of the children at their level.
      int child_offset = 8 * node.id();

      // Set IDs.
      children[child_offset + 0].set_id(8 * node.id() + 0);
      children[child_offset + 1].set_id(8 * node.id() + 1);
      children[child_offset + 2].set_id(8 * node.id() + 2);
      children[child_offset + 3].set_id(8 * node.id() + 3);
      children[child_offset + 4].set_id(8 * node.id() + 4);
      children[child_offset + 5].set_id(8 * node.id() + 5);
      children[child_offset + 6].set_id(8 * node.id() + 6);
      children[child_offset + 7].set_id(8 * node.id() + 7);


      const Bounding_box &bbox = node.bounding_box();
      // Points of the bounding-box.
      const float3 &p_min = bbox.get_min();
      const float3 &p_max = bbox.get_max();

      // Set the bounding boxes of the children.
      children[child_offset + 0].set_bounding_box(p_min.x, center.y, center.z, center.x,
                                                  p_max.y, p_max.z);  // Top-left, first level.
      children[child_offset + 1].set_bounding_box(center.x, center.y, center.z, p_max.x,
                                                  p_max.y, p_max.z);  // Top-right, first level.
      children[child_offset + 2].set_bounding_box(p_min.x, p_min.y, center.z, center.x,
                                                  center.y, p_max.z);  // Bottom-left, first level.
      children[child_offset + 3].set_bounding_box(center.x, p_min.y, center.z, p_max.x,
                                                  center.y, p_max.z);  // Bottom-right, first level.
      children[child_offset + 4].set_bounding_box(p_min.x, center.y, p_min.z, center.x,
                                                  p_max.y, center.z);  // Top-left, second level.
      children[child_offset + 5].set_bounding_box(center.x, center.y, p_min.z, p_max.x,
                                                  p_max.y, center.z);  // Top-right, second level.
      children[child_offset + 6].set_bounding_box(p_min.x, p_min.y, p_min.z, center.x,
                                                  center.y, center.z);  // Bottom-left, second level.
      children[child_offset + 7].set_bounding_box(center.x, p_min.y, p_min.z, p_max.x,
                                                  center.y, center.z);  // Bottom-right, second level.

      // Set the ranges of the children.

      children[child_offset + 0].set_range(node.points_begin(),
                                           s_num_pts[0][warp_id]);
      children[child_offset + 1].set_range(s_num_pts[0][warp_id],
                                           s_num_pts[1][warp_id]);
      children[child_offset + 2].set_range(s_num_pts[1][warp_id],
                                           s_num_pts[2][warp_id]);
      children[child_offset + 3].set_range(s_num_pts[2][warp_id],
                                           s_num_pts[3][warp_id]);

      children[child_offset + 4].set_range(s_num_pts[3][warp_id],
                                           s_num_pts[4][warp_id]);
      children[child_offset + 5].set_range(s_num_pts[4][warp_id],
                                           s_num_pts[5][warp_id]);
      children[child_offset + 6].set_range(s_num_pts[5][warp_id],
                                           s_num_pts[6][warp_id]);
      children[child_offset + 7].set_range(s_num_pts[6][warp_id],
                                           s_num_pts[7][warp_id]);

      // Launch 8 children.
      build_octree_kernel<NUM_THREADS_PER_BLOCK><<<
          8, NUM_THREADS_PER_BLOCK, 8 * NUM_WARPS_PER_BLOCK * sizeof(int)>>>(
                                                                             &children[child_offset], points, normals, unitareavec, Parameters(params, true));
    }
  }
}