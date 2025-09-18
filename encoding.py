void hd_enc_lvid_packed_cuda(
                    unsigned int* __restrict__ id_hvs_packed, 
                    unsigned int* __restrict__ level_hvs_packed, 
                    int* __restrict__ feature_indices, 
                    float* __restrict__ feature_values, 
                    int max_peaks_used, 
                    unsigned int* hv_matrix, 
                    int N, int Q, int D, int packLength) 
                    {
                    const int d = threadIdx.x + blockIdx.x * blockDim.x;
                    if (d >= D)
                        return;
                    for (int sample_idx = blockIdx.y; sample_idx < N; sample_idx += blockDim.y * gridDim.y) 
                    {
                        // we traverse [start, end-1]
                        float encoded_hv_e = 0.0;
                        unsigned int start_range = sample_idx*max_peaks_used;
                        unsigned int end_range = (sample_idx + 1)*max_peaks_used;
                        #pragma unroll 1
                        for (int f = start_range; f < end_range; ++f) {
                            if(feature_values[f] != -1)
                                encoded_hv_e += get2d_bin(level_hvs_packed, (int)(feature_values[f] * Q), D, d) * \
                                                get2d_bin(id_hvs_packed, feature_indices[f], D, d);
                        }
                        
                        // hv_matrix[sample_idx*D+d] = (encoded_hv_e > 0)? 1 : -1;
                        int tid = threadIdx.x;
                        int lane = tid % warpSize;
                        int bitPattern=0;
                        if (d < D)
                            bitPattern = __ballot_sync(0xFFFFFFFF, encoded_hv_e > 0);
                        if (lane == 0) {
                            hv_matrix[sample_idx * packLength + (d / warpSize)] = bitPattern;
                        }
                    }
                }
                ''', 'hd_enc_lvid_packed_cuda')
                
    threads = 1024
    max_block = cp.cuda.runtime.getDeviceProperties(0)['maxGridSize'][1]
    hd_enc_lvid_packed_cuda_kernel(
        ((D + threads - 1) // threads, min(N, max_block)), (threads,), 
        (id_hvs_packed, lv_hvs_packed, spectra_mz, spectra_intensity, max_peaks_used, encoded_spectra, N, Q, D, packed_dim))