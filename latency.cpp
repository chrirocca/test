#include "hip/hip_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NUM_SM 120
#define BLOCK_SIZE 64 // related with tid*N°
#define S_SIZE ((8*1024)*1024)/16 // must be smaller than L2 cache size
#define ITERATION 5

typedef unsigned int uint;

#define hipCheckError() {                                          \
    hipError_t e=hipGetLastError();                                 \
    if(e!=hipSuccess) {                                              \
        printf("HIP failure %s:%d: '%s'\n",__FILE__,__LINE__,hipGetErrorString(e));           \
        exit(0); \
    }                                                                 \
}


// hip.amdgcn.bc - device routine
/*
  HW_ID Register bit structure for RDNA2 & RDNA3
  WAVE_ID     4:0     Wave id within the SIMD.
  SIMD_ID     9:8     SIMD_ID within the WGP: [0] = row, [1] = column.
  WGP_ID      13:10   Physical WGP ID.
  SA_ID       16      Shader Array ID
  SE_ID       20:18   Shader Engine the wave is assigned to for gfx11
  SE_ID       19:18   Shader Engine the wave is assigned to for gfx10
  DP_RATE     31:29   Number of double-precision float units per SIMD

  HW_ID Register bit structure for GCN and CDNA
  WAVE_ID     3:0     Wave buffer slot number. 0-9.
  SIMD_ID     5:4     SIMD which the wave is assigned to within the CU.
  PIPE_ID     7:6     Pipeline from which the wave was dispatched.
  CU_ID       11:8    Compute Unit the wave is assigned to.
  SH_ID       12      Shader Array (within an SE) the wave is assigned to.
  SE_ID       15:13   Shader Engine the wave is assigned to for gfx908, gfx90a
              14:13   Shader Engine the wave is assigned to for gfx940-942
  TG_ID       19:16   Thread-group ID
  VM_ID       23:20   Virtual Memory ID
  QUEUE_ID    26:24   Queue from which this wave was dispatched.
  STATE_ID    29:27   State ID (graphics only, not compute).
  ME_ID       31:30   Micro-engine ID.

  XCC_ID Register bit structure for gfx940
  XCC_ID      3:0     XCC the wave is assigned to.
 */

#if (defined (__GFX10__) || defined (__GFX11__))
  #define HW_ID               23
#else
  #define HW_ID               4
#endif

#if (defined(__GFX10__) || defined(__GFX11__))
  #define HW_ID_WGP_ID_SIZE   4
  #define HW_ID_WGP_ID_OFFSET 10
  #if (defined(__AMDGCN_CUMODE__))
    #define HW_ID_CU_ID_SIZE    1
    #define HW_ID_CU_ID_OFFSET  8
  #endif
#else
  #define HW_ID_CU_ID_SIZE    4
  #define HW_ID_CU_ID_OFFSET  8
#endif

#if (defined(__gfx908__) || defined(__gfx90a__) || \
     defined(__GFX11__))
  #define HW_ID_SE_ID_SIZE    3
#else //4 SEs/XCC for gfx940-942
  #define HW_ID_SE_ID_SIZE    2
#endif
#if (defined(__GFX10__) || defined(__GFX11__))
  #define HW_ID_SE_ID_OFFSET  18
  #define HW_ID_SA_ID_OFFSET  16
  #define HW_ID_SA_ID_SIZE    1
#else
  #define HW_ID_SE_ID_OFFSET  13
#endif

#if (defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__))
  #define XCC_ID                   20
  #define XCC_ID_XCC_ID_SIZE       4
  #define XCC_ID_XCC_ID_OFFSET     0
#endif

#if (!defined(__HIP_NO_IMAGE_SUPPORT) && \
    (defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)))
  #define __HIP_NO_IMAGE_SUPPORT   1
#endif

/*
   Encoding of parameter bitmask
   HW_ID        5:0     HW_ID
   OFFSET       10:6    Range: 0..31
   SIZE         15:11   Range: 1..32
 */

#define GETREG_IMMED(SZ,OFF,REG) (((SZ) << 11) | ((OFF) << 6) | (REG))

/*
  __smid returns the wave's assigned Compute Unit and Shader Engine.
  The Compute Unit, CU_ID returned in bits 3:0, and Shader Engine, SE_ID in bits 5:4.
  Note: the results vary over time.
  SZ minus 1 since SIZE is 1-based.
*/
__device__
inline
unsigned __smid(void)
{
    unsigned se_id = __builtin_amdgcn_s_getreg(
            GETREG_IMMED(HW_ID_SE_ID_SIZE-1, HW_ID_SE_ID_OFFSET, HW_ID));
    #if (defined(__GFX10__) || defined(__GFX11__))
      unsigned wgp_id = __builtin_amdgcn_s_getreg(
            GETREG_IMMED(HW_ID_WGP_ID_SIZE - 1, HW_ID_WGP_ID_OFFSET, HW_ID));
      unsigned sa_id = __builtin_amdgcn_s_getreg(
            GETREG_IMMED(HW_ID_SA_ID_SIZE - 1, HW_ID_SA_ID_OFFSET, HW_ID));
      #if (defined(__AMDGCN_CUMODE__))
        unsigned cu_id = __builtin_amdgcn_s_getreg(
            GETREG_IMMED(HW_ID_CU_ID_SIZE - 1, HW_ID_CU_ID_OFFSET, HW_ID));
      #endif
    #else
      #if (defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__))
      unsigned xcc_id = __builtin_amdgcn_s_getreg(
            GETREG_IMMED(XCC_ID_XCC_ID_SIZE - 1, XCC_ID_XCC_ID_OFFSET, XCC_ID));
      #endif
      unsigned cu_id = __builtin_amdgcn_s_getreg(
            GETREG_IMMED(HW_ID_CU_ID_SIZE - 1, HW_ID_CU_ID_OFFSET, HW_ID));
    #endif
    #if (defined(__GFX10__) || defined(__GFX11__))
      unsigned temp = se_id;
      temp = (temp << HW_ID_SA_ID_SIZE) | sa_id;
      temp = (temp << HW_ID_WGP_ID_SIZE) | wgp_id;
      #if (defined(__AMDGCN_CUMODE__))
        temp = (temp << HW_ID_CU_ID_SIZE) | cu_id;
      #endif
      return temp;
      //TODO : CU Mode impl
    #elif (defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__))
      unsigned temp = xcc_id;
      temp = (temp << HW_ID_SE_ID_SIZE) | se_id;
      temp = (temp << HW_ID_CU_ID_SIZE) | cu_id;
      return temp;
    #else
      return (se_id << HW_ID_CU_ID_SIZE) + cu_id;
    #endif
}

__global__ void k(unsigned int *a0, unsigned int *a1, unsigned int start_idx, unsigned int sm_chosen) {
    unsigned int i;
    unsigned int sm_id = get_smid();
    unsigned int start, latency;

    if(sm_id == sm_chosen && threadIdx.x == 0) 
    {

        for (i = 0; i < ITERATION; i ++)
        {

            start = clock();

            a0[start_idx] += a1[0];

            latency = clock() - start;

            printf("%u\n",latency);


        }
        
    }

}
    
int main(int argc, char * argv[]) {
    unsigned int * h_arr;
    unsigned int * d_a0, * d_a1;
    int i,start_idx, sm_chosen;
    
    hipSetDevice(0);
    start_idx = atoi(argv[1])*64;
    sm_chosen = atoi(argv[2]);

    h_arr = (unsigned int *)malloc(sizeof(unsigned int) * S_SIZE);

    hipMalloc((void**)&d_a0, sizeof(unsigned int) * S_SIZE);
    hipCheckError();
    hipMalloc((void**)&d_a1, sizeof(unsigned int) * S_SIZE);
    hipCheckError();

    for (i = 0; i < S_SIZE; i++) {
        h_arr[i] = i;
    }

    hipMemcpy(d_a0, h_arr, sizeof(unsigned int) * S_SIZE, hipMemcpyHostToDevice);
    hipCheckError();
    hipMemcpy(d_a1, h_arr, sizeof(unsigned int) * S_SIZE, hipMemcpyHostToDevice);
    hipCheckError();

    k<<<NUM_SM, BLOCK_SIZE>>>(d_a0,d_a1,start_idx,sm_chosen);
    hipCheckError();
    hipDeviceSynchronize();
    hipCheckError();


    free(h_arr);
    hipFree(d_a0); 
    hipFree(d_a1); 

    return 0;
}


