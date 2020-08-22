#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <chrono>

#include "utils.h"
#include "kernels.h"

__global__ void fill_vector(unsigned int n, float *vec, float value)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n)
	{
		vec[i] = value;
	}
}

measurement_class cpu_csr_spmv_single_thread_naive(const csr_matrix_class& matrix, float* x, float* y)
{
	std::fill_n(x, matrix.meta.cols_count, 1.0);

	const auto row_ptr = matrix.row_ptr.get();
	const auto col_ids = matrix.columns.get();
	const auto data = matrix.data.get();

	auto begin = std::chrono::system_clock::now();

	for (unsigned int row = 0; row < matrix.meta.rows_count; row++)
	{
		const auto row_start = row_ptr[row];
		const auto row_end = row_ptr[row + 1];

		float dot = 0;
		for (auto element = row_start; element < row_end; element++)
			dot += data[element] * x[col_ids[element]];
		y[row] = dot;
	}

	auto end = std::chrono::system_clock::now();
	const double elapsed = std::chrono::duration<double>(end - begin).count() * 1000;

	const size_t data_bytes = matrix.meta.non_zero_count * sizeof(float);
	const size_t x_bytes = matrix.meta.non_zero_count * sizeof(float);
	const size_t col_ids_bytes = matrix.meta.non_zero_count * sizeof(unsigned int);
	const size_t row_ids_bytes = 2 * matrix.meta.rows_count * sizeof(unsigned int);
	const size_t y_bytes = matrix.meta.rows_count * sizeof(float);

	const size_t operations_count = matrix.meta.non_zero_count * 2;

	return measurement_class(
		"CPU CSR",
		elapsed,
		data_bytes + x_bytes + col_ids_bytes + row_ids_bytes + y_bytes,
		operations_count);
}

__global__ void ell_spmv_kernel(unsigned int n_rows, unsigned int elements_in_rows, const unsigned int* col_ids, const float* data, const float* x, float* y)
{
	unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < n_rows)
	{
		float dot = 0;
		for (unsigned int element = 0; element < elements_in_rows; element++)
		{
			const unsigned int element_offset = row + element * n_rows;
			dot += data[element_offset] * x[col_ids[element_offset]];
		}
		y[row] = dot;
	}
}

__global__ void coo_spmv_kernel(unsigned int n_elements, const unsigned int* col_ids, const unsigned int* row_ids, const float* data, const float* x, float* y)
{
	unsigned int element = blockIdx.x * blockDim.x + threadIdx.x;

	if (element < n_elements)
	{
		const float dot = data[element] * x[col_ids[element]];
		atomicAdd(y + row_ids[element], dot);
	}
}

measurement_class gpu_ell_spmv(const ell_matrix_class &matrix, resizable_gpu_memory<float>& A, resizable_gpu_memory<unsigned int>& col_ids, resizable_gpu_memory<float>& x, resizable_gpu_memory<float>& y, float* reusable_vector, const float* reference_y)
{
	auto &meta = matrix.meta;

	const size_t A_size = matrix.get_matrix_size();
	const size_t col_ids_size = A_size;
	const size_t x_size = matrix.meta.cols_count;
	const size_t y_size = matrix.meta.rows_count;

	A.resize(A_size);
	col_ids.resize(col_ids_size);
	x.resize(x_size);
	y.resize(y_size);

	cudaMemcpy(A.get(), matrix.data.get(), A_size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(col_ids.get(), matrix.columns.get(), col_ids_size * sizeof(unsigned int), cudaMemcpyHostToDevice);
	
	{
		dim3 block_size = dim3(512);
		dim3 grid_size{};

		grid_size.x = (x_size + block_size.x - 1) / block_size.x;
		fill_vector << <grid_size, block_size >> > (x_size, x.get(), 1.0);

		grid_size.x = (y_size + block_size.x - 1) / block_size.x;
		fill_vector << <grid_size, block_size >> > (y_size, y.get(), 0.0);
	}

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaDeviceSynchronize();
	cudaEventRecord(start);
		
	{
		dim3 block_size = dim3(256);
		dim3 grid_size{};

		grid_size.x = (meta.rows_count + block_size.x - 1) / block_size.x;

		ell_spmv_kernel << <grid_size, block_size >> > (meta.rows_count, matrix.elements_in_rows, col_ids.get(), A.get(), x.get(), y.get());
	}

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	cudaMemcpy(reusable_vector, y.get(), y_size * sizeof(float), cudaMemcpyDeviceToHost);

	compare_results(y_size, reusable_vector, reference_y);

	const double elapsed = milliseconds;

	const unsigned int n_elements = matrix.elements_in_rows * matrix.meta.rows_count;
	const size_t data_bytes = n_elements * sizeof(float);
	const size_t x_bytes = n_elements * sizeof(float);
	const size_t col_ids_bytes = n_elements * sizeof(unsigned int);
	const size_t y_bytes = matrix.meta.rows_count * sizeof(float);

	const size_t operations_count = n_elements * 2;

	return measurement_class("ELL", elapsed, data_bytes + x_bytes + col_ids_bytes + y_bytes, operations_count);
}

measurement_class gpu_coo_spmv(const coo_matrix_class& matrix, resizable_gpu_memory<float>& A, resizable_gpu_memory<unsigned int>& col_ids, resizable_gpu_memory<unsigned int>& row_ids, resizable_gpu_memory<float>& x, resizable_gpu_memory<float>& y, float* reusable_vector, const float* reference_y)
{
	const size_t n_elements = matrix.get_matrix_size();
	const size_t x_size = matrix.meta.cols_count;
	const size_t y_size = matrix.meta.rows_count;

	A.resize(n_elements);
	col_ids.resize(n_elements);
	row_ids.resize(n_elements);
	x.resize(x_size);
	y.resize(y_size);

	cudaMemcpy(A.get(), matrix.data.get(), n_elements * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(col_ids.get(), matrix.cols.get(), n_elements * sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(row_ids.get(), matrix.rows.get(), n_elements * sizeof(unsigned int), cudaMemcpyHostToDevice);

	{
		dim3 block_size = dim3(512);
		dim3 grid_size{};

		grid_size.x = (x_size + block_size.x - 1) / block_size.x;
		fill_vector << <grid_size, block_size >> > (x_size, x.get(), 1.0);

		grid_size.x = (y_size + block_size.x - 1) / block_size.x;
		fill_vector << <grid_size, block_size >> > (y_size, y.get(), 0.0);
	}

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaDeviceSynchronize();
	cudaEventRecord(start);
	
	{
		dim3 block_size = dim3(512);
		dim3 grid_size{};

		grid_size.x = (n_elements + block_size.x - 1) / block_size.x;

		coo_spmv_kernel << <grid_size, block_size >> > (n_elements, col_ids.get(), row_ids.get(), A.get(), x.get(), y.get());
	}

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	cudaMemcpy(reusable_vector, y.get(), y_size * sizeof(float), cudaMemcpyDeviceToHost);

	compare_results(y_size, reusable_vector, reference_y);

	const double elapsed = milliseconds;

	const size_t data_bytes = matrix.meta.non_zero_count * sizeof(float);
	const size_t x_bytes = matrix.meta.non_zero_count * sizeof(float);
	const size_t col_ids_bytes = matrix.meta.non_zero_count * sizeof(unsigned int);
	const size_t row_ids_bytes = matrix.meta.non_zero_count * sizeof(unsigned int);
	const size_t y_bytes = matrix.meta.non_zero_count * sizeof(float);

	const size_t operations_count = matrix.meta.non_zero_count * 2;
	return measurement_class("COO", elapsed, data_bytes + x_bytes + col_ids_bytes + row_ids_bytes + y_bytes, operations_count);
}