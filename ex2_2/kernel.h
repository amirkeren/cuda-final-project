#ifndef MATRIX_FORMAT_PERFORMANCE_GPU_MATRIX_MULTIPLIER_H
#define MATRIX_FORMAT_PERFORMANCE_GPU_MATRIX_MULTIPLIER_H

#include "matrix_converter.h"

measurement_class gpu_ell_spmv (
    const ell_matrix_class<float> &matrix,
    resizable_gpu_memory<float> &A,
    resizable_gpu_memory<unsigned int> &col_ids,
    resizable_gpu_memory<float> &x,
    resizable_gpu_memory<float> &y,

	float*reusable_vector,
    const float*reference_y);


measurement_class gpu_coo_spmv (
    const coo_matrix_class<float> &matrix,
    resizable_gpu_memory<float> &A,
    resizable_gpu_memory<unsigned int> &col_ids,
    resizable_gpu_memory<unsigned int> &row_ids,
    resizable_gpu_memory<float> &x,
    resizable_gpu_memory<float> &y,

	float*reusable_vector,
    const float*reference_y);

#endif // MATRIX_FORMAT_PERFORMANCE_GPU_MATRIX_MULTIPLIER_H