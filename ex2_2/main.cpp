#include <fstream>
#include <algorithm>
#include <chrono>

#include "resizable_memory.h"
#include "measurement_class.h"
#include "cpu_matrix_multiplier.h"
#include "kernel.h"

#define CHECK_CPU 0

template <typename data_type>
void perform_measurement(
	const std::string &mtx,
	const matrix_market::reader &reader)
{
	std::unique_ptr<csr_matrix_class<data_type>> csr_matrix;
	std::unique_ptr<ell_matrix_class<data_type>> ell_matrix;
	std::unique_ptr<coo_matrix_class<data_type>> coo_matrix;

	{
		csr_matrix = std::make_unique<csr_matrix_class<data_type>>(reader.matrix());
		std::cout << "Complete converting to CSR" << std::endl;

		if (!CHECK_CPU)
		{
			const size_t csr_matrix_size = csr_matrix->get_matrix_size();
			const size_t ell_matrix_size = ell_matrix_class<data_type>::estimate_size(*csr_matrix);

			const size_t vec_size = std::max(reader.matrix().meta.rows_count, reader.matrix().meta.cols_count) * sizeof(data_type);
			const size_t matrix_size = std::max(csr_matrix_size, ell_matrix_size) * sizeof(data_type);
			const size_t estimated_size = matrix_size + 5 * vec_size;

			ell_matrix = std::make_unique<ell_matrix_class<data_type>>(*csr_matrix);
			std::cout << "Complete converting to ELL" << std::endl;

			coo_matrix = std::make_unique<coo_matrix_class<data_type>>(*csr_matrix);
			std::cout << "Complete converting to COO" << std::endl;
		}
	}

	// CPU
	std::unique_ptr<data_type[]> reference_answer(new data_type[csr_matrix->meta.rows_count]);
	std::unique_ptr<data_type[]> x(new data_type[std::max(csr_matrix->meta.rows_count, csr_matrix->meta.cols_count)]);

	{
		measurement_class cpu_naive = cpu_csr_spmv_single_thread_naive(*csr_matrix, x.get(), reference_answer.get());
		std::cout << cpu_naive.get_elapsed() << std::endl;
	}

	if (CHECK_CPU)
		return;
	
	/// GPU Reusable memory
	resizable_gpu_memory<data_type> A, x_gpu, y;
	resizable_gpu_memory<unsigned int> col_ids, row_ptr;

	/// GPU
	{
		{
			measurement_class gpu_time = gpu_ell_spmv(*ell_matrix, A, col_ids, x_gpu, y, x.get(), reference_answer.get());
			std::cout << "elapsed" << gpu_time.get_elapsed() << "computational_throughput" << gpu_time.get_computational_throughput() << "effective_bandwith" << gpu_time.get_effective_bandwidth() << std::endl;
		}

		{
			measurement_class gpu_time = gpu_coo_spmv(*coo_matrix, A, col_ids, row_ptr, x_gpu, y, x.get(), reference_answer.get());
			std::cout << "elapsed" << gpu_time.get_elapsed() << "computational_throughput" << gpu_time.get_computational_throughput() << "effective_bandwith" << gpu_time.get_effective_bandwidth() << std::endl;
		}
	}
}

std::string get_filename(const std::string &path)
{
	std::size_t i = path.rfind('/', path.length());
	if (i != std::string::npos)
		return (path.substr(i + 1, path.length() - i));
	return path;
}

int main(int argc, char *argv[])
{
	if (argc != 2)
	{
		std::cerr << "Usage: " << argv[0] << " /path/to/mtx_list" << std::endl;
		return 1;
	}

	std::ifstream list(argv[1]);

	std::string mtx;

	while (std::getline(list, mtx))
	{
		std::cout << "Start loading matrix" << mtx;

		std::ifstream is(mtx);
		matrix_market::reader reader(is);
		auto &meta = reader.matrix().meta;
		std::cout << "Complete loading (rows: {}; cols: {}; nnz: {}; nnzpr: {})\n" << meta.rows_count << meta.cols_count << meta.non_zero_count << meta.non_zero_count / meta.rows_count;

		perform_measurement<float>(mtx, reader);
	}

	return 0;
}