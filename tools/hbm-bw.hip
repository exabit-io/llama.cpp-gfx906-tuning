// hbm-bw: device-local HBM bandwidth probe. Usage: hbm-bw <hip device> <seconds> [bytes]
// Runs a float4 copy kernel (read+write) and a read-only reduction on a large buffer, reports GB/s (median of timed reps) plus the PCI bus id.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <chrono>
#define CHK(x) do{hipError_t e=(x); if(e!=hipSuccess){fprintf(stderr,"HIP error %s at %s:%d\n",hipGetErrorString(e),__FILE__,__LINE__); exit(2);} }while(0)
__global__ void copyk(const float4* __restrict__ a, float4* __restrict__ b, size_t n){
  size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x, s=(size_t)gridDim.x*blockDim.x; for(;i<n;i+=s) b[i]=a[i]; }
__global__ void readk(const float4* __restrict__ a, float* __restrict__ out, size_t n){
  size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x, s=(size_t)gridDim.x*blockDim.x; float4 acc=make_float4(0,0,0,0);
  for(;i<n;i+=s){ float4 v=a[i]; acc.x+=v.x; acc.y+=v.y; acc.z+=v.z; acc.w+=v.w; }
  float r=acc.x+acc.y+acc.z+acc.w; if(r==12345.678f) out[0]=r; /* keep the loads alive, never true for zeroed input */ }
static double med(std::vector<double> v){ std::sort(v.begin(),v.end()); return v[v.size()/2]; }
int main(int argc,char**argv){
  int dev=argc>1?atoi(argv[1]):0; double secs=argc>2?atof(argv[2]):5.0; size_t bytes=argc>3?strtoull(argv[3],0,10):(size_t)1<<30;
  CHK(hipSetDevice(dev)); char pci[32]; CHK(hipDeviceGetPCIBusId(pci,32,dev)); hipDeviceProp_t p; CHK(hipGetDeviceProperties(&p,dev));
  size_t n=bytes/sizeof(float4); float4 *a,*b; float* o; CHK(hipMalloc(&a,bytes)); CHK(hipMalloc(&b,bytes)); CHK(hipMalloc(&o,64)); CHK(hipMemset(a,0,bytes)); CHK(hipMemset(b,0,bytes));
  int grid=p.multiProcessorCount*8, block=256; hipEvent_t e0,e1; CHK(hipEventCreate(&e0)); CHK(hipEventCreate(&e1));
  for(int w=0;w<3;w++){ copyk<<<grid,block>>>(a,b,n); readk<<<grid,block>>>(a,o,n); } CHK(hipDeviceSynchronize());
  std::vector<double> vc,vr; auto t0=std::chrono::steady_clock::now(); float ms;
  while(std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count()<secs){
    CHK(hipEventRecord(e0)); copyk<<<grid,block>>>(a,b,n); CHK(hipEventRecord(e1)); CHK(hipEventSynchronize(e1)); CHK(hipEventElapsedTime(&ms,e0,e1)); vc.push_back(2.0*bytes/ms/1e6);
    CHK(hipEventRecord(e0)); readk<<<grid,block>>>(a,o,n); CHK(hipEventRecord(e1)); CHK(hipEventSynchronize(e1)); CHK(hipEventElapsedTime(&ms,e0,e1)); vr.push_back(1.0*bytes/ms/1e6); }
  printf("dev %d pci %s %s reps %zu copy_GBs %.1f read_GBs %.1f copy_max %.1f read_max %.1f\n",dev,pci,p.gcnArchName,vc.size(),med(vc),med(vr),*std::max_element(vc.begin(),vc.end()),*std::max_element(vr.begin(),vr.end()));
  return 0; }
