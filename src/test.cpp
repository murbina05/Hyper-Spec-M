#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <iostream>
#include <qpl/qpl.h>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // if using std::vector

// #include "/home/zebra/blaflen/htslib/htslib-QPL/qpl_lib/include/qpl/qpl.h"
constexpr float SCALE = 1000.0f;
qpl_path_t execution_path = qpl_path_software;

namespace py = pybind11;

// typedef float float;

struct Spectrum {
    int dummy = -1;
    int charge;
    float pepmass;
    std::string filename;
    int scans;
    float rtinsecs;
    std::vector<float> mz;
    std::vector<float> intensity;
};


void pad_vectors(std::vector<float>& vec, int target_size) {
    if (vec.size() < static_cast<size_t>(target_size)) {
        vec.resize(target_size, -1.0f);
    }
}


void print_vector(const std::vector<float>& vec) {
    for (const auto& val : vec) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
}

bool _check_spectrum_valid(
    const std::vector<float>& spectrum_mz,
    int min_peaks,
    float min_mz_range
){
    if (spectrum_mz.size() < static_cast<size_t>(min_peaks)) {
        return false;
    }

    // Assumes spectrum_mz is sorted in ascending order
    float mz_range = spectrum_mz.back() - spectrum_mz.front();
    return mz_range >= min_mz_range;
}


inline float mass_diff(float mz1, float mz2, bool mode_is_da) {
    return mode_is_da ? (mz1 - mz2) : ((mz1 - mz2) / mz2) * 1e6;
}

// Declare helper functions (implement separately)
bool is_invalid(const char* line);
bool is_begin(const char* line);
bool is_end(const char* line);
bool is_scans(const char* line);
bool is_rtins(const char* line);
bool is_pepmass(const char* line);
bool is_charge(const char* line);
void fast_parse(const char* start, const char* end, float* mz_out, float* intensity_out);

// Main function
std::vector<Spectrum> load_mgf_file(const std::string& full_filename) {
    FILE* fp = fopen(full_filename.c_str(), "r");
    if (fp == NULL) {
        std::cerr << "File open failed!" << std::endl;
        exit(1);
    }

    // Extract just the filename (no path, no extension)
    std::string filename = full_filename;
    size_t last_slash = filename.find_last_of('/');
    size_t last_dot = filename.find_last_of('.');
    if (last_slash != std::string::npos) filename = filename.substr(last_slash + 1);
    if (last_dot != std::string::npos) filename = filename.substr(0, last_dot);

    char* line = NULL;
    size_t len = 0;
    ssize_t read_len;
    size_t line_num = 0;
    long peak_i = 0;

    float rtinsecs = -1, pepmass = -1;
    int charge = -1, scans = -1, spec_index = 0;
    float mz_temp, intensity_temp;
    std::vector<float> mz, intensity;
    mz.reserve(2000);
    intensity.reserve(2000);

    std::vector<Spectrum> read_spectra_list;

    while ((read_len = getline(&line, &len, fp)) != -1) {
        ++line_num;

        if (is_invalid(line)) {
            continue;
        } else if (is_begin(line)) {
            peak_i = 0;
            charge = -1;
            pepmass = -1;
            scans = -1;
            rtinsecs = -1;
            mz.clear();
            intensity.clear();
            continue;
        } else if (is_scans(line)) {
            scans = atoi(line + 6);
            continue;
        } else if (is_rtins(line)) {
            rtinsecs = atof(line + 12);
            continue;
        } else if (is_pepmass(line)) {
            pepmass = atof(line + 8);
            continue;
        } else if (is_charge(line)) {
            charge = atoi(line + 7);
            while ((read_len = getline(&line, &len, fp)) != -1) {
                if (is_end(line)) {
                    Spectrum s;
                    s.charge = charge;
                    s.pepmass = pepmass;
                    s.filename = filename;
                    s.scans = scans;
                    s.rtinsecs = rtinsecs;
                    s.mz = mz;
                    s.intensity = intensity;

                    read_spectra_list.push_back(s);
                    ++spec_index;
                    break;
                } else {
                    fast_parse(line, line + read_len, &mz_temp, &intensity_temp);
                    mz.push_back(mz_temp);
                    intensity.push_back(intensity_temp);
                    ++peak_i;
                }
            }
        }
    }

    free(line);
    fclose(fp);

    return read_spectra_list;
}



std::vector<Spectrum> load_mgf_bytes(const std::vector<uint8_t>& data, const std::string& full_filename) {
    // Extract filename (no path, no extension)
    std::string filename = full_filename;
    size_t last_slash = filename.find_last_of('/');
    size_t last_dot = filename.find_last_of('.');
    if (last_slash != std::string::npos) filename = filename.substr(last_slash + 1);
    if (last_dot != std::string::npos) filename = filename.substr(0, last_dot);

    float rtinsecs = -1, pepmass = -1;
    int charge = -1, scans = -1, spec_index = 0;
    float mz_temp, intensity_temp;
    std::vector<float> mz, intensity;
    mz.reserve(2000);
    intensity.reserve(2000);

    std::vector<Spectrum> read_spectra_list;

    const char* ptr = reinterpret_cast<const char*>(data.data());
    const char* end = ptr + data.size();

    auto next_line = [&]() -> std::string_view {
        const char* line_start = ptr;
        while (ptr < end && *ptr != '\n') ++ptr;
        std::string_view line(line_start, ptr - line_start);
        if (ptr < end && *ptr == '\n') ++ptr;
        return line;
    };

    while (ptr < end) {
        std::string_view line = next_line();
        if (line.empty() || is_invalid(line.data())) {
            continue;
        } else if (is_begin(line.data())) {
            charge = -1;
            pepmass = -1;
            scans = -1;
            rtinsecs = -1;
            mz.clear();
            intensity.clear();
            continue;
        } else if (is_scans(line.data())) {
            scans = atoi(line.data() + 6);
            continue;
        } else if (is_rtins(line.data())) {
            rtinsecs = atof(line.data() + 12);
            continue;
        } else if (is_pepmass(line.data())) {
            pepmass = atof(line.data() + 8);
            continue;
        } else if (is_charge(line.data())) {
            charge = atoi(line.data() + 7);

            while (ptr < end) {
                std::string_view peak_line = next_line();
                if (is_end(peak_line.data())) {
                    Spectrum s;
                    s.charge = charge;
                    s.pepmass = pepmass;
                    s.filename = filename;
                    s.scans = scans;
                    s.rtinsecs = rtinsecs;
                    s.mz = mz;
                    s.intensity = intensity;
                    read_spectra_list.push_back(s);
                    ++spec_index;
                    break;
                } else {
                    fast_parse(peak_line.data(), peak_line.data() + peak_line.size(), &mz_temp, &intensity_temp);
                    mz.push_back(mz_temp);
                    intensity.push_back(intensity_temp);
                }
            }
        }
    }

    return read_spectra_list;
}


inline bool starts_with(const char* buf, const char* prefix, size_t prefix_len) {
    return strncmp(buf, prefix, prefix_len) == 0;
}

inline bool is_begin(const char* buf) {
    return starts_with(buf, "BEGIN", 5);
}

inline bool is_end(const char* buf) {
    return starts_with(buf, "END", 3);
}

inline bool is_invalid(const char* buf) {
    return buf[0] == '#';
}

inline bool is_title(const char* buf) {
    return starts_with(buf, "TITLE", 5);
}

inline bool is_scans(const char* buf) {
    return starts_with(buf, "SCANS", 5);
}

inline bool is_rtins(const char* buf) {
    return starts_with(buf, "RTINS", 5);
}

inline bool is_pepmass(const char* buf) {
    return starts_with(buf, "PEPMASS", 5) || starts_with(buf, "PEPMA", 5);
}

inline bool is_charge(const char* buf) {
    return starts_with(buf, "CHARGE", 6);
}

inline void fast_parse(const char* start, const char* end, float* mz_out, float* intensity_out) {
    char* mid = nullptr;
    *mz_out = strtof(start, &mid);         // parse first float
    *intensity_out = strtof(mid, nullptr); // parse second float
}





std::vector<uint8_t> get_mz_mask_qpl(
    const std::vector<float>& mz,
    float min_mz,
    float max_mz
) {
    qpl_status status;
    size_t n = mz.size();
    std::vector<uint32_t> mz_scaled(n);
    std::vector<uint8_t> mask((n + 7) / 8, 0);

    // Fixed-point scaling
    for (size_t i = 0; i < n; ++i) {
        mz_scaled[i] = static_cast<int32_t>(mz[i]);
    }

    uint32_t scaled_min = static_cast<int32_t>(min_mz);
    uint32_t scaled_max = static_cast<int32_t>(max_mz);

    // QPL job allocation
    uint32_t job_size = 0;
    qpl_get_job_size(execution_path, &job_size);
    std::vector<uint8_t> job_buffer(job_size);
    qpl_job* job = reinterpret_cast<qpl_job*>(job_buffer.data());
    qpl_init_job(execution_path, job);


    // Setup scan-range job
    job->next_in_ptr        = reinterpret_cast<uint8_t*>(mz_scaled.data());
    job->available_in       = static_cast<uint32_t>(mz_scaled.size() * sizeof(uint32_t));;
    job->next_out_ptr       = mask.data();
    job->available_out      = static_cast<uint32_t>(mask.size());
    job->op                 = qpl_op_scan_range;
    job->src1_bit_width     = 32;
    job->num_input_elements = static_cast<uint32_t>(n);
    job->out_bit_width      = qpl_ow_nom;
    job->param_low          = scaled_min;
    job->param_high         = scaled_max;


    status = qpl_execute_job(job);
    if (status != QPL_STS_OK) {
        qpl_fini_job(job);
        throw std::runtime_error("QPL scan-range job failed");
    }
    // std::cout << "Mask bits (as bytes):" << std::endl;
    // for (size_t byte_i = 0; byte_i < mask.size(); ++byte_i) {
    //     uint8_t b = mask[byte_i];
    //     for (int bit = 0; bit <8 ; ++bit) {
    //         std::cout << ((b >> bit) & 1);
    //     }
    //     std::cout << " ";
    // }
    qpl_fini_job(job);
    return mask;
}

Spectrum  _set_mz_range_qpl(
    Spectrum spectrum,
    float min_mz,
    float max_mz
) {
    if (!min_mz && !max_mz) {
        return spectrum;
    }    

        if (spectrum.mz.empty()) {
        return spectrum; // avoid crashing on empty input
    }


    if (!min_mz) {
        min_mz = static_cast<float>(spectrum.mz.front());
    }
    if (!max_mz) {
        max_mz = static_cast<float>(spectrum.mz.back());
    }

    std::vector<uint8_t> mask = get_mz_mask_qpl(spectrum.mz, min_mz, max_mz);

    std::vector<float> new_mz;
    std::vector<float> new_intensity;
    new_mz.reserve(spectrum.mz.size());
    new_intensity.reserve(spectrum.intensity.size());

    // std::cout<<"size: "<<mask.size()<<std::endl;

    for (size_t i = 0; i < mask.size()*8; ++i) {
        if ((mask[(i/8)] & (1 << (i % 8))) != 0) {
            new_mz.push_back(spectrum.mz[i]);
            new_intensity.push_back(spectrum.intensity[i]);
        }
    }

    spectrum.mz = std::move(new_mz);
    spectrum.intensity = std::move(new_intensity);
    return spectrum;
}


constexpr float ADDUCT_MASS = 1.007825f;
constexpr float C_MASS_DIFF = 1.003355f;


void invert_mask(std::vector<uint8_t>& mask) {
    for (auto& byte : mask) {
        byte ^= 0xFF;
    }
}

void and_masks(std::vector<uint8_t>& base_mask, const std::vector<uint8_t>& new_mask) {
    for (size_t i = 0; i < base_mask.size(); ++i) {
        base_mask[i] &= new_mask[i];
    }
}



std::vector<uint8_t> run_qpl_scan_range_mask(const std::vector<uint32_t>& mz_scaled, uint32_t low, uint32_t high) {
    std::vector<uint8_t> mask((mz_scaled.size() + 7) / 8, 0);

    qpl_job* job;
    uint32_t job_size;
    qpl_get_job_size(qpl_path_software, &job_size);
    std::vector<uint8_t> job_buffer(job_size);
    job = reinterpret_cast<qpl_job*>(job_buffer.data());
    qpl_init_job(qpl_path_software, job);
    std::vector<uint32_t> mz_scaledd = mz_scaled;

    job->next_in_ptr        = reinterpret_cast<uint8_t*>(mz_scaledd.data());
    job->available_in       = mz_scaled.size() * sizeof(uint32_t);
    job->next_out_ptr       = mask.data();
    job->available_out      = mask.size();
    job->op                 = qpl_op_scan_range;
    job->src1_bit_width     = qpl_ow_32;
    job->num_input_elements = static_cast<uint32_t>(mz_scaled.size());
    job->out_bit_width      = qpl_ow_nom;
    job->param_low          = low;
    job->param_high         = high;

    qpl_execute_job(job);
    qpl_fini_job(job);

    return mask;
}


std::vector<uint8_t> generate_final_mask(const std::vector<float>& mz, const std::vector<float>& remove_mz, float tol) {
    std::vector<uint32_t> mz_scaled(mz.size());
    for (size_t i = 0; i < mz.size(); i++) {
        mz_scaled[i] = static_cast<uint32_t>(mz[i] * SCALE);
    }

    std::vector<uint8_t> final_mask((mz.size() + 7) / 8, 0xFF);  // Start with all bits 1

    for (float center : remove_mz) {
        uint32_t low = static_cast<uint32_t>((center - tol) * SCALE);
        uint32_t high = static_cast<uint32_t>((center + tol) * SCALE);

        auto mask = run_qpl_scan_range_mask(mz_scaled, low, high);
        invert_mask(mask);
        and_masks(final_mask, mask);
    }
    return final_mask;
}

void apply_mask(const std::vector<uint8_t>& mask, std::vector<float>& mz, std::vector<float>& intensity) {
    std::vector<float> filtered_mz;
    std::vector<float> filtered_intensity;

    for (size_t i = 0; i < mz.size(); i++) {
        if (mask[i / 8] & (1 << (i % 8))) {
            filtered_mz.push_back(mz[i]);
            filtered_intensity.push_back(intensity[i]);
        }
    }

    mz = std::move(filtered_mz);
    intensity = std::move(filtered_intensity);
}

int precursor_to_interval(double mz, int charge, int interval_width) {
    constexpr double hydrogen_mass = 1.00794;
    constexpr double cluster_width = 1.0005079;

    double neutral_mass = (mz - hydrogen_mass) * std::max(std::abs(charge), 1);
    return static_cast<int>(std::round(neutral_mass / cluster_width)) / interval_width;
}


std::vector<float> compute_remove_mz(
    float precursor_mz,
    int precursor_charge,
    int isotope = 0
) {
    float neutral_mass = (precursor_mz - ADDUCT_MASS) * precursor_charge;

    std::vector<float> remove_mz;
    for (int charge = precursor_charge; charge > 0; --charge) {
        for (int iso = 0; iso <= isotope; ++iso) {
            float mz = (neutral_mass + iso * C_MASS_DIFF) / charge + ADDUCT_MASS;
            remove_mz.push_back(mz);
        }
    }
    return remove_mz;
}

// std::vector<uint8_t> _remove_precursor_peak_qpl(
//     std::vector<float>& mz,
//     std::vector<float>& intensity,
//     float precursor_mz,
//     int precursor_charge,
//     float fragment_tol_mass,
//     bool mode_is_da,
//     int isotope = 0
// ) {
//     auto remove_mz = compute_remove_mz(precursor_mz, precursor_charge, isotope);

//     std::vector<uint32_t> mz_scaled(mz.size());
//     for (size_t i = 0; i < mz.size(); i++) {
//         mz_scaled[i] = static_cast<uint32_t>(mz[i] * SCALE);
//     }

//     std::vector<uint8_t> final_mask((mz.size() + 7) / 8, 0xFF);

//     for (float center : remove_mz) {
//         float tol = fragment_tol_mass;
//         if (!mode_is_da) {  // PPM mode
//             tol = center * fragment_tol_mass / 1e6f;
//         }

//         uint32_t low = static_cast<uint32_t>((center - tol) * SCALE);
//         uint32_t high = static_cast<uint32_t>((center + tol) * SCALE);

//         auto mask = run_qpl_scan_range_mask(mz_scaled, low, high);
//         invert_mask(mask);
//         and_masks(final_mask, mask);
//     }

//     apply_mask(final_mask, mz, intensity);
//     return final_mask;
// }




inline double mass_diff(double mz1, double mz2, bool mode_is_da) {
    return mode_is_da ? (mz1 - mz2) : ((mz1 - mz2) / mz2) * 1e6;
}


std::vector<uint8_t> run_qpl_mass_diff_mask(
    const std::vector<float>& mz,
    const std::vector<float>& remove_mz,
    float tol,
    bool mode_is_da
) {
    std::vector<uint32_t> mz_scaled(mz.size());
    for (size_t i = 0; i < mz.size(); i++) {
        mz_scaled[i] = static_cast<uint32_t>(mz[i] * SCALE);
    }

    std::vector<uint8_t> final_mask((mz.size() + 7) / 8, 0xFF);

    for (float center : remove_mz) {
        float low, high;
        if (mode_is_da) {
            low = center - tol;
            high = center + tol;
        } else {
            float ppm_tol = center * tol / 1e6f;
            low = center - ppm_tol;
            high = center + ppm_tol;
        }

        uint32_t scaled_low = static_cast<uint32_t>(low * SCALE);
        uint32_t scaled_high = static_cast<uint32_t>(high * SCALE);

        auto mask = run_qpl_scan_range_mask(mz_scaled, scaled_low, scaled_high);
        invert_mask(mask);
        and_masks(final_mask, mask);
    }

    return final_mask;
}
Spectrum remove_precursor_peak_qpl(Spectrum spectrum, float fragment_tol_mass, bool mode_is_da, int isotope = 0) {
    constexpr float adduct_mass = 1.007825f;
    constexpr float c_mass_diff = 1.003355f;

    float neutral_mass = (spectrum.pepmass - adduct_mass) * static_cast<float>(spectrum.charge);

    std::vector<float> remove_mz;
    for (int charge = spectrum.charge; charge > 0; --charge) {
        for (int iso = 0; iso <= isotope; ++iso) {
            float mz = (neutral_mass + iso * c_mass_diff) / static_cast<float>(charge) + adduct_mass;
            remove_mz.push_back(mz);
        }
    }

    auto mask = run_qpl_mass_diff_mask(spectrum.mz, remove_mz, fragment_tol_mass, mode_is_da);
    apply_mask(mask, spectrum.mz, spectrum.intensity);
    return spectrum;
}

// Get top-N peaks mask by intensity
std::vector<bool> get_intensity_mask(
    const std::vector<float>& intensity,
    float min_intensity_frac,
    int max_num_peaks
) {
    std::vector<size_t> indices(intensity.size());
    std::iota(indices.begin(), indices.end(), 0);

    if (intensity.size() > static_cast<size_t>(max_num_peaks)) {
        std::nth_element(indices.begin(), indices.begin() + max_num_peaks, indices.end(),
            [&](size_t a, size_t b) { return intensity[a] > intensity[b]; });
        indices.resize(max_num_peaks);
    }

    double local_max = 0.0;
    for (auto idx : indices) {
        local_max = std::max(local_max, static_cast<double>(intensity[idx]));
    }

    double threshold = min_intensity_frac * local_max;
    std::vector<bool> mask(intensity.size(), false);
    for (auto idx : indices) {
        if (intensity[idx] > threshold) {
            mask[idx] = true;
        }
    }
    return mask;
}

// Apply the mask to mz and intensity
void filter_intensity(Spectrum& spectrum, float min_intensity_frac, int max_num_peaks) {
    auto mask = get_intensity_mask(spectrum.intensity, min_intensity_frac, max_num_peaks);
    std::vector<float> filtered_mz;
    std::vector<float> filtered_intensity;

    for (size_t i = 0; i < mask.size(); ++i) {
        if (mask[i]) {
            filtered_mz.push_back(spectrum.mz[i]);
            filtered_intensity.push_back(spectrum.intensity[i]);
        }
    }
    spectrum.mz = std::move(filtered_mz);
    spectrum.intensity = std::move(filtered_intensity);
}

// Scaling function: "root", "log", or "rank"
void scale_intensity(std::vector<float>& intensity, const std::string& scaling, int max_rank = -1) {
    if (scaling == "root") {
        for (auto& val : intensity) {
            val = std::sqrt(val);
        }
    } else if (scaling == "log") {
        double log_base = std::log(2.0);
        for (auto& val : intensity) {
            val = std::log1p(val) / log_base;
        }
    } else if (scaling == "rank") {
        if (max_rank < 0) max_rank = static_cast<int>(intensity.size());
        if (max_rank < static_cast<int>(intensity.size())) {
            throw std::runtime_error("max_rank must be >= number of peaks after filtering.");
        }

        std::vector<size_t> indices(intensity.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(),
            [&](size_t a, size_t b) { return intensity[a] > intensity[b]; });

        std::vector<float> new_intensity(intensity.size());
        for (size_t rank = 0; rank < indices.size(); ++rank) {
            new_intensity[indices[rank]] = static_cast<float>(max_rank - rank);
        }
        intensity = std::move(new_intensity);
    }
}

// Normalize by L2 norm
void norm_intensity(std::vector<float>& intensity) {
    double norm = std::sqrt(std::inner_product(intensity.begin(), intensity.end(), intensity.begin(), 0.0));
    if (norm > 0.0) {
        for (auto& val : intensity) {
            val /= norm;
        }
    }
}


std::vector<Spectrum> preprocess_read_spectra_list(
    std::vector<Spectrum> spectra_list,
    int min_peaks = 5,
    float min_mz_range = 250.0,
    int mz_interval = 1,
    float mz_min = 101.0f,
    float mz_max = 1500.0f,
    float remove_precursor_tolerance = 1.5f,
    float min_intensity_frac = 0.01f,
    int max_peaks_used = 50,
    std::string scaling = "off"
) {
    std::vector<size_t> invalid_spec_indices;

    for (size_t i = 0; i < spectra_list.size(); ++i) {
        // Restrict m/z range
        spectra_list[i] = _set_mz_range_qpl(spectra_list[i], mz_min, mz_max);

        if (!_check_spectrum_valid(spectra_list[i].mz, min_peaks, min_mz_range)) {
            invalid_spec_indices.push_back(i);
            continue;
        }

        // Remove precursor peak
        spectra_list[i] = remove_precursor_peak_qpl(spectra_list[i], remove_precursor_tolerance, true, 0);
        if (!_check_spectrum_valid(spectra_list[i].mz, min_peaks, min_mz_range)) {
            invalid_spec_indices.push_back(i);
            continue;
        }

        // Filter intensities
        filter_intensity(spectra_list[i], min_intensity_frac, max_peaks_used);
        if (!_check_spectrum_valid(spectra_list[i].mz, min_peaks, min_mz_range)) {
            invalid_spec_indices.push_back(i);
            continue;
        }

        // Scale intensities
        scale_intensity(spectra_list[i].intensity, scaling, max_peaks_used);

        // Normalize intensities
        norm_intensity(spectra_list[i].intensity);

        // Assign interval index
        int interval_idx = precursor_to_interval(spectra_list[i].pepmass, spectra_list[i].charge, mz_interval);
        spectra_list[i].dummy = interval_idx;

        // Pad mz and intensity arrays
        pad_vectors(spectra_list[i].mz, max_peaks_used);
        pad_vectors(spectra_list[i].intensity, max_peaks_used);
    }

    // Remove invalid spectra
    std::vector<Spectrum> cleaned;
    for (size_t i = 0; i < spectra_list.size(); ++i) {
        if (std::find(invalid_spec_indices.begin(), invalid_spec_indices.end(), i) == invalid_spec_indices.end()) {
            cleaned.push_back(spectra_list[i]);
        }
    }

    return cleaned;
}

std::vector<uint8_t> decomp(const std::string& full_filename, qpl_huffman_table_t& huffman_table) {

    const std::string dataset_path =full_filename;
    const std::string table_path = "/project/max/Hyper-Spec-M/src/tables/huffman_table_S.bin";

    std::ifstream infile(table_path, std::ios::binary);

    // qpl_huffman_table_t huffman_table = nullptr;
    // uint32_t table_size = std::filesystem::file_size(table_path);
    // // std::vector<uint8_t> buffer(size);

    // const std::unique_ptr<uint8_t[]> unique_buffer = std::make_unique<uint8_t[]>(table_size);
    // uint8_t*                         buffer        = reinterpret_cast<uint8_t*>(unique_buffer.get());

    // if (!infile.read(reinterpret_cast<char*>(buffer), table_size)) {
    //     std::cerr << "Failed to read file data.\n";
    //     std::cout<<"error reading file, size: "<<table_size<<std::endl;
    //     infile.close();
    //     throw std::runtime_error("Failed to read file data.");
    // }
    // infile.close();

    // qpl_status status;

    // status = qpl_huffman_table_deserialize(buffer, table_size,DEFAULT_ALLOCATOR_C, &huffman_table);
    // if (status != QPL_STS_OK) {
    //     std::cerr << "Failed to deserialize Huffman table: " << status << std::endl;
    //     qpl_huffman_table_destroy(huffman_table);
    //     std::cout<<"error deserializing table, status: "<<status<<std::endl;
    //     throw std::runtime_error("Failed to read file data.");


    // }
    // std::cout<<"was able to deserlize table I bhope o.O"<<std::endl;

    
    std::ifstream file(full_filename, std::ifstream::binary);

    if (!file.is_open()) {
        std::cout << "Unable to open file in " << dataset_path << '\n';
        file.close();
        throw std::runtime_error("Failed to read file data.");
        // return;
    }

    std::vector<uint8_t> source;
    std::vector<uint8_t> destination;
    // std::vector<uint8_t> reference;

    source.reserve(std::filesystem::file_size(full_filename));
    source.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());

    // // Get compression buffer size estimate
    // const uint32_t compression_size = qpl_get_safe_deflate_compression_buffer_size(source.size());
    // if (compression_size == 0) {
    //     std::cout << "Invalid source size. Source size exceeds the maximum supported size.\n";
    //     return 1;
    // }

    destination.resize(source.size() * 3); 
    // reference.resize(source.size());

    std::unique_ptr<uint8_t[]> job_buffer;
    uint32_t                   size = 0;

    // Job initialization
    qpl_status status = qpl_get_job_size(execution_path, &size);
    if (status != QPL_STS_OK) {
        std::cout << "An error " << status << " acquired during job size getting.\n";
        throw std::runtime_error("Failed to read file data.");
    }

    job_buffer   = std::make_unique<uint8_t[]>(size);
    qpl_job* job = reinterpret_cast<qpl_job*>(job_buffer.get());

    status = qpl_init_job(execution_path, job);
    if (status != QPL_STS_OK) {
        std::cout << "An error " << status << " acquired during job initializing.\n";
        throw std::runtime_error("Failed to read file data.");
    }
    job->op            = qpl_op_decompress;
    job->next_in_ptr   = source.data();
    job->next_out_ptr  = destination.data();
    job->available_in  = source.size();
    job->available_out = static_cast<uint32_t>(destination.size());
    job->flags         = QPL_FLAG_FIRST | QPL_FLAG_LAST | QPL_FLAG_CANNED_MODE;
    job->huffman_table = huffman_table;

    // Compression
    status = qpl_execute_job(job);
    if (status != QPL_STS_OK) {
        std::cout << "An error " << status << " acquired during compression.\n";
        // qpl_huffman_table_destroy(huffman_table);
        throw std::runtime_error("Failed to read file data.");

    }
    status = qpl_fini_job(job);
    if (status != QPL_STS_OK) {
        std::cout << "An error " << status << " acquired during job finalization.\n";
        throw std::runtime_error("Failed to decompress data.");
    }
    return destination; // Return the decompressed data


    
}

qpl_huffman_table_t create_huffman_table(const std::string& table_path) {
    std::ifstream infile(table_path, std::ios::binary);
    if (!infile.is_open()) {
        std::cerr << "Unable to open Huffman table file: " << table_path << '\n';
        throw std::runtime_error("Failed to read Huffman table data.");
    }
   
    uint32_t table_size = std::filesystem::file_size(table_path);
    const std::unique_ptr<uint8_t[]> unique_buffer = std::make_unique<uint8_t[]>(table_size);
    uint8_t*                         buffer        = reinterpret_cast<uint8_t*>(unique_buffer.get());

    if (!infile.read(reinterpret_cast<char*>(buffer), table_size)) {
        std::cerr << "Failed to read file data.\n";
        std::cout<<"error reading file, size: "<<table_size<<std::endl;
        infile.close();
        throw std::runtime_error("Failed to read file data.");
    }
    infile.close();

    qpl_huffman_table_t huffman_table = nullptr;
    qpl_status status = qpl_huffman_table_deserialize(buffer, table_size,DEFAULT_ALLOCATOR_C, &huffman_table);
    if (status != QPL_STS_OK) {
        std::cerr << "Failed to deserialize Huffman table: " << status << std::endl;
        qpl_huffman_table_destroy(huffman_table);
        std::cout<<"error deserializing table, status: "<<status<<std::endl;
        throw std::runtime_error("Failed to read file data.");

    }  
    return huffman_table;
}

int comp(const std::string& full_filename, const std::string& output_filename, qpl_huffman_table_t huffman_table) {
    // Implement compression logic here
    // This is a placeholder function for demonstration purposes
    qpl_status status;
    const std::string dataset_path =full_filename;
    // const std::string table_path = "/project/max/Hyper-Spec-M/src/tables/huffman_table_S.bin";

    // std::ifstream infile(table_path, std::ios::binary);

    // qpl_huffman_table_t huffman_table = nullptr;
    // uint32_t table_size = std::filesystem::file_size(table_path);
    // // std::vector<uint8_t> buffer(size);

    // const std::unique_ptr<uint8_t[]> unique_buffer = std::make_unique<uint8_t[]>(table_size);
    // uint8_t*                         buffer        = reinterpret_cast<uint8_t*>(unique_buffer.get());

    // if (!infile.read(reinterpret_cast<char*>(buffer), table_size)) {
    //     std::cerr << "Failed to read file data.\n";
    //     std::cout<<"error reading file, size: "<<table_size<<std::endl;
    //     infile.close();
    //     throw std::runtime_error("Failed to read file data.");
    // }
    // infile.close();



    // status = qpl_huffman_table_deserialize(buffer, table_size,DEFAULT_ALLOCATOR_C, &huffman_table);
    // if (status != QPL_STS_OK) {
    //     std::cerr << "Failed to deserialize Huffman table: " << status << std::endl;
    //     qpl_huffman_table_destroy(huffman_table);
    //     std::cout<<"error deserializing table, status: "<<status<<std::endl;
    //     throw std::runtime_error("Failed to read file data.");


    // }
    
    std::ifstream file(full_filename, std::ifstream::binary);

    if (!file.is_open()) {
        std::cout << "Unable to open file in " << dataset_path << '\n';
        file.close();
        throw std::runtime_error("Failed to read file data.");
        // return;
    }
    auto t_start = std::chrono::high_resolution_clock::now();
    std::vector<uint8_t> source;
    std::vector<uint8_t> destination;
    // std::vector<uint8_t> reference;

    source.reserve(std::filesystem::file_size(full_filename));
    source.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());

    // Get compression buffer size estimate
    const uint32_t compression_size = qpl_get_safe_deflate_compression_buffer_size(source.size());
    if (compression_size == 0) {
        std::cout << "Invalid source size. Source size exceeds the maximum supported size.\n";
        return 1;
    }

    destination.resize(compression_size); // Allocate enough space for compressed data
    // reference.resize(source.size());

    std::unique_ptr<uint8_t[]> job_buffer;
    uint32_t                   size = 0;
    qpl_histogram              deflate_histogram {};

    // Job initialization
    status = qpl_get_job_size(execution_path, &size);
    if (status != QPL_STS_OK) {
        std::cout << "An error " << status << " acquired during job size getting.\n";
        throw std::runtime_error("Failed to read file data.");
    }

    job_buffer   = std::make_unique<uint8_t[]>(size);
    qpl_job* job = reinterpret_cast<qpl_job*>(job_buffer.get());

    status = qpl_init_job(execution_path, job);
    if (status != QPL_STS_OK) {
        std::cout << "An error " << status << " acquired during job initializing.\n";
        throw std::runtime_error("Failed to read file data.");
    }
    job->op            = qpl_op_compress;
    job->level         = qpl_default_level;
    job->next_in_ptr   = source.data();
    job->next_out_ptr  = destination.data();
    job->available_in  = static_cast<uint32_t>(source.size());
    job->available_out = static_cast<uint32_t>(destination.size());
    job->flags         = QPL_FLAG_FIRST | QPL_FLAG_LAST | QPL_FLAG_CANNED_MODE | QPL_FLAG_OMIT_VERIFY;
    job->huffman_table = huffman_table;


    // Compression
    status = qpl_execute_job(job);
    if (status != QPL_STS_OK) {
        std::cout << "An error " << status << " acquired during compression.\n";
        qpl_huffman_table_destroy(huffman_table);
        throw std::runtime_error("Failed to read file data.");

    }
    status = qpl_fini_job(job);
    if (status != QPL_STS_OK) {
        std::cout << "An error " << status << " acquired during job finalization.\n";
        throw std::runtime_error("Failed to decompress data.");
    }

    std::cout << "Compression function called with: " << full_filename << " and output: " << output_filename << std::endl;
    std::ofstream outfile(output_filename, std::ios::binary);

    auto t_end = std::chrono::high_resolution_clock::now();
    std::cout<<"Time taken to compress file: "<< std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count()<<" ms"<<std::endl;
    outfile.write(reinterpret_cast<const char*>(destination.data()), job->available_out);
    outfile.close();
    std::cout<<job->total_out<<std::endl;
    return 0; // Return appropriate status code
}


PYBIND11_MODULE(spectrum_module, m) {
    py::class_<Spectrum>(m, "Spectrum")
        .def(py::init<>())
        .def_readwrite("dummy", &Spectrum::dummy)
        .def_readwrite("charge", &Spectrum::charge)
        .def_readwrite("pepmass", &Spectrum::pepmass)
        .def_readwrite("filename", &Spectrum::filename)
        .def_readwrite("scans", &Spectrum::scans)
        .def_readwrite("rtinsecs", &Spectrum::rtinsecs)
        .def_readwrite("mz", &Spectrum::mz)
        .def_readwrite("intensity", &Spectrum::intensity);

    m.def("preprocess_read_spectra_list", &preprocess_read_spectra_list,
        py::arg("spectra_list"),
        py::arg("min_peaks") = 5,
        py::arg("min_mz_range") = 250.0f,
        py::arg("mz_interval") = 1,
        py::arg("mz_min") = 101.0f,
        py::arg("mz_max") = 1500.0f,
        py::arg("remove_precursor_tolerance") = 1.5f,
        py::arg("min_intensity_frac") = 0.01f,
        py::arg("max_peaks_used") = 50,
        py::arg("scaling") = "off"
    );
}


int main(int argc, char** argv){
    


    // if(comp(argv[1], "comp.bin") != 0){
    //     std::cerr << "Failed to compress file: " << argv[1] << std::endl;
    //     return 1;
    // }
    qpl_huffman_table_t huffman_table = create_huffman_table("/project/max/Hyper-Spec-M/src/tables/huffman_table_S.bin");
    std::string comp_filename = "/project/max/Hyper-Spec-M/src/comp.bin";

    auto t_start = std::chrono::high_resolution_clock::now();
    std::vector<Spectrum> spec = load_mgf_file(argv[1]);
    auto t_after_load = std::chrono::high_resolution_clock::now();

    comp(argv[1], comp_filename, huffman_table);
    auto t_comp = std::chrono::high_resolution_clock::now();

    std::vector<uint8_t> mgf_data = decomp(comp_filename, huffman_table);
    auto t_decomp = std::chrono::high_resolution_clock::now();

    // if (mgf_data.empty()) {
    //     std::cerr << "Failed to load MGF data from " << argv[1] << std::endl;
    //     return 1;
    // }
    // std::cout << "Loaded MGF data of size: " << mgf_data.size() << " bytes\n";


    // std::vector<Spectrum> spec = load_mgf_bytes(mgf_data, argv[1]);

    auto t_before_mask = std::chrono::high_resolution_clock::now();

    preprocess_read_spectra_list(
    spec,
    5,               // min_peaks
    250.0,           // min_mz_range
    1,               // mz_interval
    101.0,           // mz_min
    1000.0,          // mz_max
    1.5,             // remove_precursor_tolerance
    0.01,            // min_intensity
    50,              // max_peaks_used
    "off"            // scaling
);
    // std::cout<<"hello!"<<argv[1]<<std::endl;
    auto t_after_mask = std::chrono::high_resolution_clock::now();

    // for(auto s: spec){
    //     // std::cout<< s.filename << " " 
    //     //          << s.charge << " "
    //     //          << s.pepmass << " "
    //     //          << s.scans << " "
    //     //          << s.rtinsecs << std::endl;
    //     // print_vector(s.mz);
        
    //     // s = _set_mz_range_qpl(s, 100.0f, 200.0f);
    //     // print_vector(s.mz);
    //     // std::cout<<mask.data()<<std::endl;
    //     break;
    // }
    auto decomp_tim = std::chrono::duration_cast<std::chrono::milliseconds>(t_decomp - t_comp);
    auto load_time = std::chrono::duration_cast<std::chrono::milliseconds>(t_after_load - t_start);
    auto mask_time = std::chrono::duration_cast<std::chrono::milliseconds>(t_after_mask - t_before_mask);

    std::cout << "Time taken to decomp MGF: " << decomp_tim.count() << " ms\n";
    std::cout << "Time taken to load MGF: " << load_time.count() << " ms\n";
    std::cout << "Time taken to apply QPL mask: " << mask_time.count() << " ms\n";


    return 0; 
}