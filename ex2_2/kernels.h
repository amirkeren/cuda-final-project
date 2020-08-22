#include "matrices.h"

measurement_class cpu_csr_spmv_single_thread_naive(const csr_matrix_class& matrix, float* x, float* y);
measurement_class gpu_ell_spmv (const ell_matrix_class& matrix, resizable_gpu_memory<float>& A, resizable_gpu_memory<unsigned int>& col_ids, resizable_gpu_memory<float>& x, resizable_gpu_memory<float>& y, float*reusable_vector, const float*reference_y);
measurement_class gpu_coo_spmv (const coo_matrix_class& matrix, resizable_gpu_memory<float>& A, resizable_gpu_memory<unsigned int>& col_ids, resizable_gpu_memory<unsigned int>& row_ids, resizable_gpu_memory<float>& x, resizable_gpu_memory<float>& y, float* reusable_vector, const float* reference_y);