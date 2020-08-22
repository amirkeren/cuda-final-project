#include "cpu_matrix_multiplier.h"
#include "matrix_converter.h"
#include "measurement_class.h"

#include <algorithm>
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <atomic>
#include <array>

#include <immintrin.h>

using namespace std;

measurement_class cpu_csr_spmv_single_thread_naive(
	const csr_matrix_class<float> &matrix,
	float *x,
	float *y)
{
	fill_n(x, matrix.meta.cols_count, 1.0);

	const auto row_ptr = matrix.row_ptr.get();
	const auto col_ids = matrix.columns.get();
	const auto data = matrix.data.get();

	auto begin = chrono::system_clock::now();

	for (unsigned int row = 0; row < matrix.meta.rows_count; row++)
	{
		const auto row_start = row_ptr[row];
		const auto row_end = row_ptr[row + 1];

		float dot = 0;
		for (auto element = row_start; element < row_end; element++)
			dot += data[element] * x[col_ids[element]];
		y[row] = dot;
	}

	auto end = chrono::system_clock::now();
	const double elapsed = chrono::duration<double>(end - begin).count();

	const size_t data_bytes = matrix.meta.non_zero_count * sizeof(float);
	const size_t x_bytes = matrix.meta.non_zero_count * sizeof(float);
	const size_t col_ids_bytes = matrix.meta.non_zero_count * sizeof(unsigned int);
	const size_t row_ids_bytes = 2 * matrix.meta.rows_count * sizeof(unsigned int);
	const size_t y_bytes = matrix.meta.rows_count * sizeof(float);

	const size_t operations_count = matrix.meta.non_zero_count * 2; // + and * per element

	return measurement_class(
		"CPU CSR",
		elapsed,
		data_bytes + x_bytes + col_ids_bytes + row_ids_bytes + y_bytes,
		operations_count);
}