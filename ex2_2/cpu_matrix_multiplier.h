#ifndef MATRIX_FORMAT_PERFORMANCE_CPU_MATRIX_MULTIPLIER_H
#define MATRIX_FORMAT_PERFORMANCE_CPU_MATRIX_MULTIPLIER_H

#include "measurement_class.h"

template <typename data_type>
class csr_matrix_class;

/// Perform y = Ax
measurement_class cpu_csr_spmv_single_thread_naive(
	const csr_matrix_class<float> &matrix,
	float *x,
	float *y);

#endif // MATRIX_FORMAT_PERFORMANCE_CPU_MATRIX_MULTIPLIER_H