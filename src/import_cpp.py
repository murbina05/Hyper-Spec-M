import spectrum_module

from utils import load_mgf_file, export_mgf_file
import time
from hd_preprocess import preprocess_read_spectra_list
import numpy as np
from typing import Optional


def fast_mgf_parse(filename):
    read_spectra_list = load_mgf_file(filename)
    return read_spectra_list


# Create a dummy Spectrum
s1 = spectrum_module.Spectrum()
s1.dummy = 0
s1.charge = 2
s1.pepmass = 500.0
s1.filename = "test1.mzML"
s1.scans = 100
s1.rtinsecs = 30.0
s1.mz = [101.0, 150.0, 200.0, 300.0]   # example m/z peaks
s1.intensity = [100.0, 200.0, 150.0, 50.0]

# Another Spectrum
s2 = spectrum_module.Spectrum()
s2.dummy = 1
s2.charge = 3
s2.pepmass = 600.0
s2.filename = "test2.mzML"
s2.scans = 120
s2.rtinsecs = 45.0
s2.mz = [110.0, 160.0, 250.0, 400.0]
s2.intensity = [80.0, 160.0, 120.0, 60.0]
spectra_list = [s1, s2]


processed_list = spectrum_module.preprocess_read_spectra_list(
    spectra_list,
    min_peaks=2,
    min_mz_range=100.0,
    mz_interval=1,
    mz_min=100.0,
    mz_max=500.0,
    remove_precursor_tolerance=1.5,
    min_intensity_frac=0.01,
    max_peaks_used=50,
    scaling="off"
)



def load_process_single(
    file: str,
    if_preprocess: bool = True,
    min_peaks: int = 5, min_mz_range: float = 250.0,
    mz_interval: int = 1,
    mz_min: Optional[float] = 101.0,
    mz_max: Optional[float] = 1500.,
    remove_precursor_tolerance: Optional[float] = 1.50,
    min_intensity: Optional[float] = 0.01,
    max_peaks_used: Optional[int] = 50,
    scaling: Optional[str] = 'off',
    file_type: Optional[str] = 'mgf'
):
    spec_list = []
    start_time = time.time()
    spec_list = fast_mgf_parse(file)
    end_time = time.time()
    print(f"Time taken to load {file}: {end_time - start_time} seconds")

    spec_list = preprocess_read_spectra_list(
        spectra_list = spec_list,
        min_peaks = min_peaks, min_mz_range = min_mz_range,
        mz_interval = mz_interval,
        mz_min = mz_min, mz_max = mz_max,
        remove_precursor_tolerance = remove_precursor_tolerance,
        min_intensity = min_intensity,
        max_peaks_used = max_peaks_used,
        scaling = scaling)
    t_filter = time.time()
    print(f"Time taken to preprocess {file}: {t_filter - end_time} seconds")

if __name__ == "__main__":
    start_time = time.time()
    load_process_single("/project/sumukh/raw-ms-dataset/PXD001468/b1906_293T_proteinID_01A_QE3_122212.mgf")
    end_time = time.time()
    print(f"Time taken to load and process: {end_time - start_time} seconds")