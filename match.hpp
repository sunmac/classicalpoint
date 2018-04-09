// ---------------------------------------------------------
// Copyright (c) 2016, Andy Zeng
//
// This file is part of the 3DMatch Toolbox and is available
// under the terms of the Simplified BSD License provided in
// LICENSE. Please retain this notice and LICENSE if you use
// this file (or any portion of it) in your project.
// ---------------------------------------------------------

#include "utils.hpp"
#include"readfile.cpp"
#define CUDA_NUM_THREADS 512
#define CUDA_MAX_NUM_BLOCKS 2880
// CUDA kernel function: for each voxel center of the voxel grid, find nearest neighbor distance to the surface point cloud
__global__
void Get1NNSearchDist(int num_surface_pts,
                      int voxel_grid_dimX, int voxel_grid_dimY, int voxel_grid_dimZ,
                      float min_bound_3d_x, float min_bound_3d_y, float min_bound_3d_z,
                      float max_bound_3d_x, float max_bound_3d_y, float max_bound_3d_z,
                      float * GPU_surface_pts_3d, float * GPU_closest_pt_dists) {

  int pt_idx = (blockIdx.x * CUDA_NUM_THREADS + threadIdx.x);
  if (pt_idx >= voxel_grid_dimX * voxel_grid_dimY * voxel_grid_dimZ)
    return;

  // Get voxel grid coordinates of current thread
  int z = floorf(pt_idx / (voxel_grid_dimX * voxel_grid_dimY)); //
  int y = floorf((pt_idx - (z * voxel_grid_dimX * voxel_grid_dimY)) / voxel_grid_dimX);
  int x = pt_idx - (z * voxel_grid_dimX * voxel_grid_dimY) - (y * voxel_grid_dimX);

  float voxel_grid_unit_x = (max_bound_3d_x - min_bound_3d_x) / (float)voxel_grid_dimX;
  float voxel_grid_unit_y = (max_bound_3d_y - min_bound_3d_y) / (float)voxel_grid_dimY;
  float voxel_grid_unit_z = (max_bound_3d_z - min_bound_3d_z) / (float)voxel_grid_dimZ;

  // Convert from voxel grid coordinates to camera coordinates
  float pt_cam_x = ((float)x + 0.5f) * voxel_grid_unit_x + min_bound_3d_x;
  float pt_cam_y = ((float)y + 0.5f) * voxel_grid_unit_y + min_bound_3d_y;
  float pt_cam_z = ((float)z + 0.5f) * voxel_grid_unit_z + min_bound_3d_z;

  // Compute distance from voxel center to closest surface point
  float closest_dist = -1.0f;
  for (int i = 0; i < num_surface_pts; ++i) {
    float query_pt_x = GPU_surface_pts_3d[3 * i + 0];
    float query_pt_y = GPU_surface_pts_3d[3 * i + 1];
    float query_pt_z = GPU_surface_pts_3d[3 * i + 2];
    float query_dist = sqrtf((pt_cam_x - query_pt_x) * (pt_cam_x - query_pt_x) +
                             (pt_cam_y - query_pt_y) * (pt_cam_y - query_pt_y) +
                             (pt_cam_z - query_pt_z) * (pt_cam_z - query_pt_z));
    if (closest_dist == -1.0f || query_dist < closest_dist)
      closest_dist = query_dist;
  }
  GPU_closest_pt_dists[pt_idx] = closest_dist;
}


__global__
void ComputeTDF(int CUDA_LOOP_IDX, float * voxel_grid_occ, float * voxel_grid_TDF,
                int voxel_grid_dim_x, int voxel_grid_dim_y, int voxel_grid_dim_z,
                float voxel_grid_origin_x, float voxel_grid_origin_y, float voxel_grid_origin_z,
                float voxel_size, float trunc_margin) {

  int voxel_idx = CUDA_LOOP_IDX * CUDA_NUM_THREADS * CUDA_MAX_NUM_BLOCKS + blockIdx.x * CUDA_NUM_THREADS + threadIdx.x;
  if (voxel_idx > (voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z))
    return;

  int pt_grid_z = (int)floor((float)voxel_idx / ((float)voxel_grid_dim_x * (float)voxel_grid_dim_y));
  int pt_grid_y = (int)floor(((float)voxel_idx - ((float)pt_grid_z * (float)voxel_grid_dim_x * (float)voxel_grid_dim_y)) / (float)voxel_grid_dim_x);
  int pt_grid_x = (int)((float)voxel_idx - ((float)pt_grid_z * (float)voxel_grid_dim_x * (float)voxel_grid_dim_y) - ((float)pt_grid_y * (float)voxel_grid_dim_x));

  int search_radius = (int)round(trunc_margin / voxel_size);

  if (voxel_grid_occ[voxel_idx] > 0) {
    voxel_grid_TDF[voxel_idx] = 1.0f; // on surface
    return;
  }

  // Find closest surface point
  for (int iix = max(0, pt_grid_x - search_radius); iix < min(voxel_grid_dim_x, pt_grid_x + search_radius + 1); ++iix)
    for (int iiy = max(0, pt_grid_y - search_radius); iiy < min(voxel_grid_dim_y, pt_grid_y + search_radius + 1); ++iiy)
      for (int iiz = max(0, pt_grid_z - search_radius); iiz < min(voxel_grid_dim_z, pt_grid_z + search_radius + 1); ++iiz) {
        int iidx = iiz * voxel_grid_dim_x * voxel_grid_dim_y + iiy * voxel_grid_dim_x + iix;
        if (voxel_grid_occ[iidx] > 0) {
          float xd = (float)(pt_grid_x - iix);
          float yd = (float)(pt_grid_y - iiy);
          float zd = (float)(pt_grid_z - iiz);
          float dist = sqrtf(xd * xd + yd * yd + zd * zd) / (float)search_radius;
          if ((1.0f - dist) > voxel_grid_TDF[voxel_idx])
            voxel_grid_TDF[voxel_idx] = 1.0f - dist;
        }
      }
}


// 3DMatch data layer for Marvin
template <class T>
class MatchDataLayer : public DataLayer {
  std::future<void> lock;

  std::vector<StorageT*> data_CPU;
  std::vector<StorageT*> data_GPU;
  std::vector<StorageT*> label_CPU;
  std::vector<StorageT*> label_GPU;

public:

  std::string data_path;
  std::string data;
  std::string dataset;
  int batch_size;

  // Pre-defined parameters
  int im_height = 480;
  int im_width = 640;
  int volume_dim = 30;
  int num_pts=0;
  float* keypts_grid;
  int voxel_grid_dim_z;
  int voxel_grid_dim_x;
  int voxel_grid_dim_y;
  float * voxel_grid_TDF ;
  int numofitems() { return 0; };
  std::vector<struct set> set;
  void init() {
    train_me = true;
    std::cout << "MatchDataLayer: " << std::endl;
    data_CPU.resize(3);
    data_GPU.resize(3);
    label_CPU.resize(2);
    label_GPU.resize(2);
    readtraining();

    // Compute batch data sizes
    std::vector<int> data_dim;
    data_dim.push_back(batch_size); data_dim.push_back(1); data_dim.push_back(volume_dim); data_dim.push_back(volume_dim); data_dim.push_back(volume_dim);
    data_CPU[0]  = new StorageT[numel(data_dim)];
    data_CPU[1]  = new StorageT[numel(data_dim)];
    data_CPU[2]  = new StorageT[numel(data_dim)];

    // Compute batch label sizes
    std::vector<int> label_dim;
    label_dim.push_back(batch_size); label_dim.push_back(1); label_dim.push_back(1); label_dim.push_back(1); label_dim.push_back(1);
    label_CPU[0] = new StorageT[numel(label_dim)];
    label_CPU[1] = new StorageT[numel(label_dim)];
  };

  MatchDataLayer(std::string name_, Phase phase_, std::string data_path_, int batch_size_):
    DataLayer(name_), data_path(data_path_), batch_size(batch_size_) {
    phase = phase_;
    init();
  };

  MatchDataLayer(JSON* json) {
    SetOrDie(json, name)
    SetValue(json, phase, Training)
    SetOrDie(json, data)
    SetOrDie(json, dataset)
    SetOrDie(json, batch_size)
    init();
  };

  ~MatchDataLayer() {
    if (lock.valid()) lock.wait();
    for (int i = 0; i < data_CPU.size(); ++i)
      if (data_CPU[i] != NULL) delete [] data_CPU[i];
    for (int i = 0; i < label_CPU.size(); ++i)
      if (label_CPU[i] != NULL) delete [] label_CPU[i];
    for (int i = 0; i < data_GPU.size(); ++i)
      if (data_GPU[i] != NULL) checkCUDA(__LINE__, cudaFree(data_GPU[i]));
    for (int i = 0; i < label_GPU.size(); ++i)
      if (label_GPU[i] != NULL) checkCUDA(__LINE__, cudaFree(label_GPU[i]));
    
  };

  void shuffle() {};

  // Given a depth map and the pixel coordinates of a point p, compute the local volumetric voxel grid of TDF values around p
  void GetLocalPointvoxel_grid_TDF(int pix_x, int pix_y, float * cam_K, float * depth_im_p1, int im_height, int im_width, float * voxel_grid_TDF, int voxel_grid_dim, float voxel_size, float trunc_margin) {

    // Project pixel location to 3D point in camera coordinates
    float pt_cam_z = depth_im_p1[pix_y * im_width + pix_x];
    float pt_cam_x = ((float)(pix_x) + 0.5f - cam_K[0 * 3 + 2]) * pt_cam_z / cam_K[0 * 3 + 0];
    float pt_cam_y = ((float)(pix_y) + 0.5f - cam_K[1 * 3 + 2]) * pt_cam_z / cam_K[1 * 3 + 1];



    // Get pixel bounding box of local volume
    float loose_box_radius = ((float)voxel_grid_dim / 2.0f + 3.0f) * voxel_size; // Bounding box margin size: 3 voxels
    float bounds_3d_x[2] = {pt_cam_x + loose_box_radius, pt_cam_x - loose_box_radius};
    float bounds_3d_y[2] = {pt_cam_y + loose_box_radius, pt_cam_y - loose_box_radius};
    float bounds_3d_z[2] = {pt_cam_z + loose_box_radius, pt_cam_z - loose_box_radius};
    float bbox_pts_3d[8][3] = {{bounds_3d_x[0], bounds_3d_y[0], bounds_3d_z[0]},
      {bounds_3d_x[0], bounds_3d_y[0], bounds_3d_z[1]},
      {bounds_3d_x[0], bounds_3d_y[1], bounds_3d_z[0]},
      {bounds_3d_x[0], bounds_3d_y[1], bounds_3d_z[1]},
      {bounds_3d_x[1], bounds_3d_y[0], bounds_3d_z[0]},
      {bounds_3d_x[1], bounds_3d_y[0], bounds_3d_z[1]},
      {bounds_3d_x[1], bounds_3d_y[1], bounds_3d_z[0]},
      {bounds_3d_x[1], bounds_3d_y[1], bounds_3d_z[1]}
    };
    float min_bounds_2d[2] = {(float)im_width, (float)im_height}; // x,y
    float max_bounds_2d[2] = {0};
    for (int i = 0; i < 8; ++i) {
      float tmp_pix_x = std::round((bbox_pts_3d[i][0] * cam_K[0 * 3 + 0] / bbox_pts_3d[i][2]) + cam_K[0 * 3 + 2] - 0.5f);
      float tmp_pix_y = std::round((bbox_pts_3d[i][1] * cam_K[1 * 3 + 1] / bbox_pts_3d[i][2]) + cam_K[1 * 3 + 2] - 0.5f);
      min_bounds_2d[0] = std::min(tmp_pix_x, min_bounds_2d[0]);
      min_bounds_2d[1] = std::min(tmp_pix_y, min_bounds_2d[1]);
      max_bounds_2d[0] = std::max(tmp_pix_x, max_bounds_2d[0]);
      max_bounds_2d[1] = std::max(tmp_pix_y, max_bounds_2d[1]);
    }
    min_bounds_2d[0] = std::max(min_bounds_2d[0], 0.0f);
    min_bounds_2d[1] = std::max(min_bounds_2d[1], 0.0f);
    max_bounds_2d[0] = std::min(max_bounds_2d[0], (float)im_width - 1.0f);
    max_bounds_2d[1] = std::min(max_bounds_2d[1], (float)im_height - 1.0f);

    // Project pixels in image bounding box to 3D
    int num_local_region_pts = (max_bounds_2d[0] - min_bounds_2d[0] + 1) * (max_bounds_2d[1] - min_bounds_2d[1] + 1);
    // std::cout << num_local_region_pts << std::endl;
    float * num_local_region_pts_3d = new float[3 * num_local_region_pts];
    int num_local_region_pts_3dIdx = 0;
    for (int y = min_bounds_2d[1]; y <= max_bounds_2d[1]; ++y)
      for (int x = min_bounds_2d[0]; x <= max_bounds_2d[0]; ++x) {
        float tmp_pt_cam_z = depth_im_p1[y * im_width + x];
        float tmp_pt_cam_x = ((float)x + 0.5f - cam_K[0 * 3 + 2]) * tmp_pt_cam_z / cam_K[0 * 3 + 0];
        float tmp_pt_cam_y = ((float)y + 0.5f - cam_K[1 * 3 + 2]) * tmp_pt_cam_z / cam_K[1 * 3 + 1];
        num_local_region_pts_3d[3 * num_local_region_pts_3dIdx + 0] = tmp_pt_cam_x;
        num_local_region_pts_3d[3 * num_local_region_pts_3dIdx + 1] = tmp_pt_cam_y;
        num_local_region_pts_3d[3 * num_local_region_pts_3dIdx + 2] = tmp_pt_cam_z;
        num_local_region_pts_3dIdx++;
      }

    // FILE *fp = fopen("test.txt", "w");
    // for (int i = 0; i < num_local_region_pts; ++i) {
    //     std::cout << num_local_region_pts_3d[3 * i + 0] << " " << num_local_region_pts_3d[3 * i + 1] << " " << num_local_region_pts_3d[3 * i + 2] << std::endl;
    //     float tmpx = num_local_region_pts_3d[3 * i + 0];
    //     float tmpy = num_local_region_pts_3d[3 * i + 1];
    //     float tmpz = num_local_region_pts_3d[3 * i + 2];
    //     int iret = fprintf(fp, "%f %f %f\n",tmpx,tmpy,tmpz);
    // }
    // for (int i = 0; i < 8; ++i) {
    //     float tmpx = bbox_pts_3d[i][0];
    //     float tmpy = bbox_pts_3d[i][1];
    //     float tmpz = bbox_pts_3d[i][2];
    //     int iret = fprintf(fp, "%f %f %f\n",tmpx,tmpy,tmpz);
    // }
    // fclose(fp);

    // Prepare GPU variables
    int voxel_grid_dimX = voxel_grid_dim;
    int voxel_grid_dimY = voxel_grid_dim;
    int voxel_grid_dimZ = voxel_grid_dim;
    int num_grid_pts = voxel_grid_dimX * voxel_grid_dimY * voxel_grid_dimZ;
    float * closest_pt_dists = new float[num_grid_pts];
    float * GPU_local_region_pts_3d;
    float * GPU_closest_pt_dists;
    cudaMalloc(&GPU_local_region_pts_3d, 3 * num_local_region_pts * sizeof(float));
    cudaMalloc(&GPU_closest_pt_dists, num_grid_pts * sizeof(float));
    checkCUDA(__LINE__, cudaGetLastError());
    cudaMemcpy(GPU_local_region_pts_3d, num_local_region_pts_3d, 3 * num_local_region_pts * sizeof(float), cudaMemcpyHostToDevice);
    checkCUDA(__LINE__, cudaGetLastError());

    // Compute xyz range of 3d local voxel grid in camera coordinates
    float tight_box_radius = ((float)voxel_grid_dim / 2.0f) * voxel_size;
    float min_bound_3d_x = pt_cam_x - tight_box_radius;
    float max_bound_3d_x = pt_cam_x + tight_box_radius;
    float min_bound_3d_y = pt_cam_y - tight_box_radius;
    float max_bound_3d_y = pt_cam_y + tight_box_radius;
    float min_bound_3d_z = pt_cam_z - tight_box_radius;
    float max_bound_3d_z = pt_cam_z + tight_box_radius;

    // Compute voxel grid of TDF values (use CUDA GPU kernel function)
    int TMP_CUDA_NUM_BLOCKS = std::ceil((float)(voxel_grid_dim * voxel_grid_dim * voxel_grid_dim) / (float)CUDA_NUM_THREADS);
    Get1NNSearchDist <<< TMP_CUDA_NUM_BLOCKS, CUDA_NUM_THREADS >>> (num_local_region_pts,
        voxel_grid_dimX, voxel_grid_dimY, voxel_grid_dimZ,
        min_bound_3d_x, min_bound_3d_y, min_bound_3d_z,
        max_bound_3d_x, max_bound_3d_y, max_bound_3d_z,
        GPU_local_region_pts_3d, GPU_closest_pt_dists);
    checkCUDA(__LINE__, cudaGetLastError());
    cudaMemcpy(closest_pt_dists, GPU_closest_pt_dists, num_grid_pts * sizeof(float), cudaMemcpyDeviceToHost);
    checkCUDA(__LINE__, cudaGetLastError());

    // Apply truncation, normalize between 0 and 1, and flip so 1 is near surface and 0 is far away from surface
    for (int i = 0; i < num_grid_pts; ++i)
      voxel_grid_TDF[i] = 1.0f - (std::min(closest_pt_dists[i], trunc_margin) / trunc_margin);

    // std::cout << pt_cam_x << " " << pt_cam_y << " " << pt_cam_z << std::endl;
    // std::cout << pix_x << " " << pix_y << std::endl;
    // std::cout << min_bounds_2d[0] << " " << min_bounds_2d[1] << " " << max_bounds_2d[0] << " " << max_bounds_2d[1] << std::endl;

    delete [] num_local_region_pts_3d;
    delete [] closest_pt_dists;
    checkCUDA(__LINE__, cudaFree(GPU_local_region_pts_3d));
    checkCUDA(__LINE__, cudaFree(GPU_closest_pt_dists));
  }

  void prefetch() {//修改读取数据方式

    checkCUDA(__LINE__, cudaSetDevice(GPU));



    // Naming convention: p1 and p2 are matches, p1 and p3 are non-matches
    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {


      int num_grid_pts = volume_dim * volume_dim * volume_dim;
      int i_index=rand()%(num_pts);
     // std::cout<<i_index<<" dd"<<std::endl;        
     int p2_index=FINDSAMEINDEX(set,i_index);
     int p3_index=FINDDIFFERENTINDEX(set,i_index); 
     //float * voxel_grid_TDF 
     // std::cout<<batch_idx<<std::endl;   
     
      float * voxel_grid_TDF_p1 = TDFPATCH(keypts_grid,num_pts,i_index, voxel_grid_dim_z,voxel_grid_dim_x,voxel_grid_dim_y,voxel_grid_TDF);
      float * voxel_grid_TDF_p2 = TDFPATCH(keypts_grid,num_pts,p2_index, voxel_grid_dim_z,voxel_grid_dim_x,voxel_grid_dim_y,voxel_grid_TDF);
      float * voxel_grid_TDF_p3 = TDFPATCH(keypts_grid,num_pts,p3_index, voxel_grid_dim_z,voxel_grid_dim_x,voxel_grid_dim_y,voxel_grid_TDF);    
      //std::cout<<i_index<<" "<<p2_index<<" "<<p3_index<<std::endl;
      // Copy to data response
      checkCUDA(__LINE__, cudaMemcpy(&(data_GPU[0][batch_idx * num_grid_pts]), voxel_grid_TDF_p1, num_grid_pts * sizeofStorageT, cudaMemcpyHostToDevice));
      checkCUDA(__LINE__, cudaMemcpy(&(data_GPU[1][batch_idx * num_grid_pts]), voxel_grid_TDF_p2, num_grid_pts * sizeofStorageT, cudaMemcpyHostToDevice));
      checkCUDA(__LINE__, cudaMemcpy(&(data_GPU[2][batch_idx * num_grid_pts]), voxel_grid_TDF_p3, num_grid_pts * sizeofStorageT, cudaMemcpyHostToDevice));
      float pos_label = 1.0f;
      float neg_label = 0.0f;
      checkCUDA(__LINE__, cudaMemcpy(&(label_GPU[0][batch_idx]), &pos_label, sizeofStorageT, cudaMemcpyHostToDevice));
      checkCUDA(__LINE__, cudaMemcpy(&(label_GPU[1][batch_idx]), &neg_label, sizeofStorageT, cudaMemcpyHostToDevice));

      // Clear memory
      //delete [] depth_im_p1;
      delete [] voxel_grid_TDF_p1;
     // delete [] depth_im_p2;
      delete [] voxel_grid_TDF_p2;
     // delete [] depth_im_p3;
      delete [] voxel_grid_TDF_p3;

    }
   // delete[] keypts_grid;

  };

  void forward(Phase phase_) {
    lock.wait();
    std::swap(out[0]->dataGPU, data_GPU[0]);
    std::swap(out[1]->dataGPU, data_GPU[1]);
    std::swap(out[2]->dataGPU, data_GPU[2]);
    std::swap(out[3]->dataGPU, label_GPU[0]);
    std::swap(out[4]->dataGPU, label_GPU[1]);
    lock = std::async(std::launch::async, &MatchDataLayer<T>::prefetch, this);
    // prefetch();
  };


  size_t Malloc(Phase phase_) {
    if (phase == Training && phase_ == Testing) return 0;
    if (out.size() != 5) {
      std::cout << "MatchDataLayer: incorrect # of out's" << std::endl;
      FatalError(__LINE__);
    }
    size_t memory_bytes = 0;
    std::cout << (train_me ? "* " : "  ");
    std::cout << name << std::endl;

    // CPU/GPU malloc data
    std::vector<int> data_dim;
    data_dim.push_back(batch_size); data_dim.push_back(1); data_dim.push_back(30); data_dim.push_back(30); data_dim.push_back(30);

    out[0]->need_diff = false;
    out[0]->receptive_field.resize(data_dim.size() - 2); fill_n(out[0]->receptive_field.begin(), data_dim.size() - 2, 1);
    out[0]->receptive_gap.resize(data_dim.size() - 2);   fill_n(out[0]->receptive_gap.begin(), data_dim.size() - 2, 1);
    out[0]->receptive_offset.resize(data_dim.size() - 2); fill_n(out[0]->receptive_offset.begin(), data_dim.size() - 2, 0);
    memory_bytes += out[0]->Malloc(data_dim);
    checkCUDA(__LINE__, cudaMalloc(&data_GPU[0], numel(data_dim) * sizeofStorageT) );
    memory_bytes += numel(data_dim) * sizeofStorageT;

    out[1]->need_diff = false;
    out[1]->receptive_field.resize(data_dim.size() - 2); fill_n(out[1]->receptive_field.begin(), data_dim.size() - 2, 1);
    out[1]->receptive_gap.resize(data_dim.size() - 2);   fill_n(out[1]->receptive_gap.begin(), data_dim.size() - 2, 1);
    out[1]->receptive_offset.resize(data_dim.size() - 2); fill_n(out[1]->receptive_offset.begin(), data_dim.size() - 2, 0);
    memory_bytes += out[1]->Malloc(data_dim);
    checkCUDA(__LINE__, cudaMalloc(&data_GPU[1], numel(data_dim) * sizeofStorageT) );
    memory_bytes += numel(data_dim) * sizeofStorageT;

    out[2]->need_diff = false;
    out[2]->receptive_field.resize(data_dim.size() - 2); fill_n(out[2]->receptive_field.begin(), data_dim.size() - 2, 1);
    out[2]->receptive_gap.resize(data_dim.size() - 2);   fill_n(out[2]->receptive_gap.begin(), data_dim.size() - 2, 1);
    out[2]->receptive_offset.resize(data_dim.size() - 2); fill_n(out[2]->receptive_offset.begin(), data_dim.size() - 2, 0);
    memory_bytes += out[2]->Malloc(data_dim);
    checkCUDA(__LINE__, cudaMalloc(&data_GPU[2], numel(data_dim) * sizeofStorageT) );
    memory_bytes += numel(data_dim) * sizeofStorageT;

    // CPU/GPU malloc labels
    std::vector<int> label_dim;
    label_dim.push_back(batch_size); label_dim.push_back(1); label_dim.push_back(1); label_dim.push_back(1); label_dim.push_back(1);

    out[3]->need_diff = false;
    memory_bytes += out[3]->Malloc(label_dim);
    checkCUDA(__LINE__, cudaMalloc(&label_GPU[0], numel(label_dim) * sizeofStorageT) );
    memory_bytes += numel(label_dim) * sizeofStorageT;

    out[4]->need_diff = false;
    memory_bytes += out[4]->Malloc(label_dim);
    checkCUDA(__LINE__, cudaMalloc(&label_GPU[1], numel(label_dim) * sizeofStorageT) );
    memory_bytes += numel(label_dim) * sizeofStorageT;

    lock = std::async(std::launch::async, &MatchDataLayer<T>::prefetch, this);
    // prefetch();

    return memory_bytes;
  };
  int FINDSAMEINDEX(std::vector<struct set> set1,int INDEX)
  {
//  int re;
    for(size_t i=0;i<set1.size();i++)
    {
      if(INDEX>=set1[i].begin&&INDEX<=set1[i].end)
      {
     // int re=;
        return rand()%(set1[i].end-set1[i].begin+1)+set1[i].begin;
      }
    }
    return -1;
  }
  int FINDDIFFERENTINDEX(std::vector<struct set> set1,int INDEX)
  {
//  int re;
    while(true)
    {
      int different= rand()%(set1.size());
      if(INDEX<=set1[different].begin||INDEX>=set1[different].end)
      {
        return different;
      }

    }
  }


  float* TDFPATCH(float* grid,int num,int index,int voxel_grid_dim_z,int voxel_grid_dim_x,int voxel_grid_dim_y,float*voxel_grid_TDF,int dim=30)
  {
    float * local_voxel_grid_TDF = new float[dim*dim*dim];
    int keypt_grid_x=grid[index*3];
    int keypt_grid_y=grid[index*3+1];
    int keypt_grid_z=grid[index*3+2];
    int local_voxel_idx=0;
    StorageT * local_voxel_grid_TDF1 = new StorageT[30 * 30 * 30];
    for (int z = keypt_grid_z - 15; z < keypt_grid_z + 15; ++z)
        for (int y = keypt_grid_y - 15; y < keypt_grid_y + 15; ++y)
          for (int x = keypt_grid_x - 15; x < keypt_grid_x + 15; ++x)
          {
           // std::cout<<z * voxel_grid_dim_x * voxel_grid_dim_y + y * voxel_grid_dim_x + x<<std::endl;
            //local_voxel_grid_TDF1[local_voxel_idx] =CPUCompute2StorageT(voxel_grid_TDF[z * voxel_grid_dim_x * voxel_grid_dim_y + y * voxel_grid_dim_x + x]);
            //local_voxel_grid_TDF[local_voxel_idx] = CPUCompute2StorageT(voxel_grid_TDF[z * voxel_grid_dim_x * voxel_grid_dim_y + y * voxel_grid_dim_x + x]);
         //   std::cout<<keypt_grid_x<<" "<<keypt_grid_y<<" "<<keypt_grid_z<<" "<<x<<" "<<y<<" "<<z<<std::endl;
          //  std::cout<<z * voxel_grid_dim_x * voxel_grid_dim_y + y * voxel_grid_dim_x + x<<" "<<voxel_grid_dim_z*voxel_grid_dim_y*voxel_grid_dim_x<<" "<<voxel_grid_dim_x<<" "<<voxel_grid_dim_y
         //   <<" "<<voxel_grid_dim_z<<std::endl;
            local_voxel_grid_TDF[local_voxel_idx] = voxel_grid_TDF[z * voxel_grid_dim_x * voxel_grid_dim_y + y * voxel_grid_dim_x + x];
            local_voxel_idx++;
          }
    return local_voxel_grid_TDF;
  }

void readtraining()
{
  //std::cout<<"名称"<<data<<" "<<dataset<<std::endl;
    READS(dataset,set);
    float * pts = READPLOUD(data,num_pts); // Nx3 matrix saved as float array (row-major order)
    std::cout << "Loaded point cloud with " << num_pts << " points!" << std::endl;
    float voxel_size = 1;
    float trunc_margin = voxel_size * 5;
    int voxel_grid_padding = 15; // in voxels
 
  // Compute point cloud coordinates of the origin voxel (0,0,0) of the voxel grid
    float voxel_grid_origin_x, voxel_grid_origin_y, voxel_grid_origin_z;
    float voxel_grid_max_x, voxel_grid_max_y, voxel_grid_max_z;
    voxel_grid_origin_x = pts[0]; voxel_grid_max_x = pts[0];
    voxel_grid_origin_y = pts[1]; voxel_grid_max_y = pts[1];
    voxel_grid_origin_z = pts[2]; voxel_grid_max_z = pts[2];
    for (int pt_idx = 0; pt_idx < num_pts; ++pt_idx) {
        voxel_grid_origin_x = min(voxel_grid_origin_x, pts[pt_idx * 3 + 0]);
        voxel_grid_origin_y = min(voxel_grid_origin_y, pts[pt_idx * 3 + 1]);
        voxel_grid_origin_z = min(voxel_grid_origin_z, pts[pt_idx * 3 + 2]);
        voxel_grid_max_x = max(voxel_grid_max_x, pts[pt_idx * 3 + 0]);
        voxel_grid_max_y = max(voxel_grid_max_y, pts[pt_idx * 3 + 1]);
        voxel_grid_max_z = max(voxel_grid_max_z, pts[pt_idx * 3 + 2]);
    }

    voxel_grid_dim_x = round((voxel_grid_max_x - voxel_grid_origin_x) / voxel_size) + 1 + voxel_grid_padding * 2;
    voxel_grid_dim_y = round((voxel_grid_max_y - voxel_grid_origin_y) / voxel_size) + 1 + voxel_grid_padding * 2;
    voxel_grid_dim_z = round((voxel_grid_max_z - voxel_grid_origin_z) / voxel_size) + 1 + voxel_grid_padding * 2;

    voxel_grid_origin_x = voxel_grid_origin_x - voxel_grid_padding * voxel_size + voxel_size / 2;
    voxel_grid_origin_y = voxel_grid_origin_y - voxel_grid_padding * voxel_size + voxel_size / 2;
    voxel_grid_origin_z = voxel_grid_origin_z - voxel_grid_padding * voxel_size + voxel_size / 2;

    std::cout << "Size of TDF voxel grid: " << voxel_grid_dim_x << " x " << voxel_grid_dim_y << " x " << voxel_grid_dim_z << std::endl;
    std::cout << "Computing TDF voxel grid..." << std::endl;

  // Compute surface occupancy grid
    float * voxel_grid_occ = new float[voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z];
    memset(voxel_grid_occ, 0, sizeof(float) * voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z);
    for (int pt_idx = 0; pt_idx < num_pts; ++pt_idx) {
        int pt_grid_x = round((pts[pt_idx * 3 + 0] - voxel_grid_origin_x) / voxel_size);
        int pt_grid_y = round((pts[pt_idx * 3 + 1] - voxel_grid_origin_y) / voxel_size);
        int pt_grid_z = round((pts[pt_idx * 3 + 2] - voxel_grid_origin_z) / voxel_size);
        voxel_grid_occ[pt_grid_z * voxel_grid_dim_y * voxel_grid_dim_x + pt_grid_y * voxel_grid_dim_x + pt_grid_x] = 1.0f;
    }

  // Initialize TDF voxel grid
    voxel_grid_TDF = new float[voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z];
    memset(voxel_grid_TDF, 0, sizeof(float) * voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z);
  // Copy voxel grids to GPU memory
    float * gpu_voxel_grid_occ;
    float * gpu_voxel_grid_TDF;
    cudaMalloc(&gpu_voxel_grid_occ, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float));
    cudaMalloc(&gpu_voxel_grid_TDF, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float));
    marvin::checkCUDA(__LINE__, cudaGetLastError());
    cudaMemcpy(gpu_voxel_grid_occ, voxel_grid_occ, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_voxel_grid_TDF, voxel_grid_TDF, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float), cudaMemcpyHostToDevice);
    marvin::checkCUDA(__LINE__, cudaGetLastError());
   // delete [] voxel_grid_TDF;
    int CUDA_NUM_LOOPS = (int)ceil((float)(voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z) / (float)(CUDA_NUM_THREADS * CUDA_MAX_NUM_BLOCKS));

    for (int CUDA_LOOP_IDX = 0; CUDA_LOOP_IDX < CUDA_NUM_LOOPS; ++CUDA_LOOP_IDX) {
        ComputeTDF <<< CUDA_MAX_NUM_BLOCKS, CUDA_NUM_THREADS >>>(CUDA_LOOP_IDX, gpu_voxel_grid_occ, gpu_voxel_grid_TDF,
        voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z,
        voxel_grid_origin_x, voxel_grid_origin_y, voxel_grid_origin_z,
        voxel_size, trunc_margin);
    }


    cudaMemcpy(voxel_grid_TDF, gpu_voxel_grid_TDF, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float), cudaMemcpyDeviceToHost);
    marvin::checkCUDA(__LINE__, cudaGetLastError());   
    /*
    for(int i=0;i<voxel_grid_dim_x;i++)
    {
      for(int j=0;j<voxel_grid_dim_y;j++)
      {
        for(int k=0;k<voxel_grid_dim_z;k++)
        {
          CPUCompute2StorageT(voxel_grid_TDF[i * voxel_grid_dim_x * voxel_grid_dim_y + j * voxel_grid_dim_x + k]);
       //   if(voxel_grid_TDF[i * voxel_grid_dim_x * voxel_grid_dim_y + j * voxel_grid_dim_x + k]>0)
          //  std::cout<<voxel_grid_TDF[i * voxel_grid_dim_x * voxel_grid_dim_y + j * voxel_grid_dim_x + k]<<" "<<i<<" "<<j<<" "<<k<<std::endl;
        }
      }
    }*/
       // Compute random surface keypoints in point cloud coordinates and voxel grid coordinates

    std::cout << "Finding random surface points..." << std::endl;
    int num_keypts = num_pts;

    //公共变量  记得free
    float * keypts = new float[num_keypts * 3];
    keypts_grid = new float[num_keypts * 3];
    for (int keypt_idx = 0; keypt_idx < num_keypts; ++keypt_idx) {
        int rand_idx=keypt_idx;
       // int rand_fix=dis1(random);
        keypts[keypt_idx * 3 + 0] = pts[rand_idx * 3 + 0];
        keypts[keypt_idx * 3 + 1] = pts[rand_idx * 3 + 1];
        keypts[keypt_idx * 3 + 2] = pts[rand_idx * 3 + 2];
        keypts_grid[keypt_idx * 3 + 0] = round((pts[rand_idx * 3 + 0] - voxel_grid_origin_x) / voxel_size);
        keypts_grid[keypt_idx * 3 + 1] = round((pts[rand_idx * 3 + 1] - voxel_grid_origin_y) / voxel_size);
        keypts_grid[keypt_idx * 3 + 2] = round((pts[rand_idx * 3 + 2] - voxel_grid_origin_z) / voxel_size);
        //std::cout<<keypts_grid[keypt_idx * 3 + 0] <<" "<<keypts_grid[keypt_idx * 3 + 1] <<" "<<keypts_grid[keypt_idx * 3 + 2] <<std::endl;
    }
    
    
    delete [] voxel_grid_occ;
  //  return 0;
    delete[] pts;
}
};
