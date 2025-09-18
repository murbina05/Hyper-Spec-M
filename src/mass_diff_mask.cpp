#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include "/home/zebra/blaflen/htslib/htslib-QPL/qpl_lib/include/qpl/qpl.h"

namespace py = pybind11;

constexpr float SCALE = 1000.0f;
qpl_path_t execution_path = qpl_path_software;

// std::vector<uint8_t> mass_diff_mask_qpl(
//     const std::vector<float>& mz,
//     const std::vector<float>& remove_mz,
//     float tol,
//     bool mode_is_da
// ) {
//     size_t n = mz.size();
//     std::vector<uint8_t> mask(n, 0xFF);
//     std::vector<int32_t> mz_scaled(n);
//     std::vector<int32_t> diff_scaled(n);
//     std::vector<uint8_t> mask_i(n);

//     for (size_t i = 0; i < n; ++i) {
//         mz_scaled[i] = static_cast<int32_t>(mz[i] * SCALE);
//     }
//     qpl_path_t execution_path = qpl_path_software;
//     uint32_t job_size = 0;
//     qpl_get_job_size(execution_path, &job_size);
//     std::vector<uint8_t> job_buffer(job_size);
//     qpl_job* job = reinterpret_cast<qpl_job*>(job_buffer.data());
//     qpl_init_job(execution_path, job);

//     for (float rmz : remove_mz) {
//         int32_t ref = static_cast<int32_t>(rmz * SCALE);

//         for (size_t i = 0; i < n; ++i) {
//             diff_scaled[i] = std::abs(mz_scaled[i] - ref);
//         }

//         job->op = qpl_op_compare;
//         job->param_low = static_cast<int32_t>(tol * SCALE);
//         job->param_high = 0;
//         job->param_middle = 0;
//         job->src1_ptr = reinterpret_cast<uint8_t*>(diff_scaled.data());
//         job->available_src1 = n * sizeof(int32_t);
//         job->src1_bit_width = qpl_ow_32;
//         job->cmp_operation = qpl_cmp_gt;
//         job->dst_ptr = mask_i.data();
//         job->available_dst = n;

//         qpl_execute_job(job);
//         if (job->status != QPL_STS_OK) {
//             qpl_fini_job(job);
//             throw std::runtime_error("QPL compare job failed");
//         }

//         for (size_t i = 0; i < n; ++i) {
//             mask[i] = mask[i] & mask_i[i];
//         }
//     }

//     qpl_fini_job(job);
//     return mask;
// }



std::vector<uint8_t> get_mz_mask_qpl(
    const std::vector<float>& mz,
    float min_mz,
    float max_mz
) {
    qpl_status status;
    size_t n = mz.size();
    std::vector<uint32_t> mz_scaled(n);
    std::vector<uint8_t> mask(n);

    // Fixed-point scaling
    for (size_t i = 0; i < n; ++i) {
        mz_scaled[i] = static_cast<int32_t>(mz[i] * SCALE);
    }

    uint32_t scaled_min = static_cast<int32_t>(min_mz * SCALE);
    uint32_t scaled_max = static_cast<int32_t>(max_mz * SCALE);

    // QPL job allocation
    uint32_t job_size = 0;
    qpl_get_job_size(execution_path, &job_size);
    std::vector<uint8_t> job_buffer(job_size);
    qpl_job* job = reinterpret_cast<qpl_job*>(job_buffer.data());
    qpl_init_job(execution_path, job);

    // Setup scan-range job
    job->next_in_ptr        = reinterpret_cast<uint8_t*>(mz_scaled.data());
    job->available_in       = static_cast<uint32_t>(mz_scaled.size());
    job->next_out_ptr       = reinterpret_cast<uint8_t*>(mask.data());
    job->available_out      = static_cast<uint32_t>(mask.size());
    job->op                 = qpl_op_scan_range;
    job->src1_bit_width     = qpl_ow_32;
    job->num_input_elements = static_cast<uint32_t>(mz_scaled.size());
    job->out_bit_width      = qpl_ow_nom;
    job->param_low          = scaled_min;
    job->param_high         = scaled_max;


    status = qpl_execute_job(job);
    if (status != QPL_STS_OK) {
        qpl_fini_job(job);
        throw std::runtime_error("QPL scan-range job failed");
    }

    qpl_fini_job(job);
    return mask;
}




// // pybind11 wrapper
// PYBIND11_MODULE(massdiff_qpl, m) {
//     m.def("mass_diff_mask", &mass_diff_mask_qpl,
//         py::arg("mz"),
//         py::arg("remove_mz"),
//         py::arg("tol"),
//         py::arg("mode_is_da") = true,
//         "Vectorized mass difference mask using QPL");
// }

// pybind11 wrapper
PYBIND11_MODULE(get_mz_mask_qpl, m) {
    m.def("mz_diff_mask", &get_mz_mask_qpl,
        py::arg("mz"),
        py::arg("min_mz"),
        py::arg("max_mz"),
        "Vectorized mz difference mask using QPL");
}
