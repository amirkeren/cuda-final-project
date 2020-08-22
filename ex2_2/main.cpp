#include <fstream>
#include <algorithm>
#include <chrono>

#include "utils.h"
#include "kernels.h"

const unsigned int MEASUREMENT_COUNT = 20;

void perform_measurement(const std::string& mtx, const matrix_reader::reader& reader)
{
	std::unique_ptr<csr_matrix_class> csr_matrix;
	std::unique_ptr<ell_matrix_class> ell_matrix;
	std::unique_ptr<coo_matrix_class> coo_matrix;

	csr_matrix = std::make_unique<csr_matrix_class>(reader.matrix());
	std::cout << "Converted to CSR" << std::endl;

	const size_t csr_matrix_size = csr_matrix->get_matrix_size();
	const size_t ell_matrix_size = ell_matrix_class::estimate_size(*csr_matrix);

	const size_t vec_size = std::max(reader.matrix().meta.rows_count, reader.matrix().meta.cols_count) * sizeof(float);
	const size_t matrix_size = std::max(csr_matrix_size, ell_matrix_size) * sizeof(float);
	const size_t estimated_size = matrix_size + 5 * vec_size;

	ell_matrix = std::make_unique<ell_matrix_class>(*csr_matrix);
	std::cout << "Converted to ELL" << std::endl;

	coo_matrix = std::make_unique<coo_matrix_class>(*csr_matrix);
	std::cout << "Converted to COO" << std::endl;

	// CPU
	std::unique_ptr<float[]> reference_answer(new float[csr_matrix->meta.rows_count]);
	std::unique_ptr<float[]> x(new float[std::max(csr_matrix->meta.rows_count, csr_matrix->meta.cols_count)]);

	for (unsigned int measurement_id = 0; measurement_id < MEASUREMENT_COUNT; measurement_id++)
	{
		measurement_class cpu_naive = cpu_csr_spmv_single_thread_naive(*csr_matrix, x.get(), reference_answer.get());
		std::cout << "CSR: " << "elapsed [ms] - " << cpu_naive.get_elapsed() << ", computational throughput - " << cpu_naive.get_computational_throughput() << ", effective bandwith - " << cpu_naive.get_effective_bandwidth() << std::endl;
	}

	// GPU
	resizable_gpu_memory<float> A, x_gpu, y;
	resizable_gpu_memory<unsigned int> col_ids, row_ptr;

	for (unsigned int measurement_id = 0; measurement_id < MEASUREMENT_COUNT; measurement_id++)
	{
		measurement_class etll_res = gpu_ell_spmv(*ell_matrix, A, col_ids, x_gpu, y, x.get(), reference_answer.get());
		std::cout << "ELL: " << "elapsed [ms] - " << etll_res.get_elapsed() << ", computational throughput - " << etll_res.get_computational_throughput() << ", effective bandwith - " << etll_res.get_effective_bandwidth() << std::endl;
	}

	for (unsigned int measurement_id = 0; measurement_id < MEASUREMENT_COUNT; measurement_id++)
	{
		measurement_class coo_res = gpu_coo_spmv(*coo_matrix, A, col_ids, row_ptr, x_gpu, y, x.get(), reference_answer.get());
		std::cout << "COO: " << "elapsed [ms] - " << coo_res.get_elapsed() << ", computational throughput - " << coo_res.get_computational_throughput() << ", effective bandwith - " << coo_res.get_effective_bandwidth() << std::endl;
	}
}

int main(int argc, char* argv[])
{
	std::ifstream list("C:\\Users\\amirke\\source\\repos\\ex2_2\\ex2_2\\list.txt");
	std::string mtx;

	while (std::getline(list, mtx))
	{
		std::cout << std::endl << "Start loading matrix - " << mtx << std::endl;

		std::ifstream is(mtx);
		matrix_reader::reader reader(is);
		auto& meta = reader.matrix().meta;
		std::cout << "Complete loading - rows: " << meta.rows_count << ", columns: " << meta.cols_count << ", none zero count: " << meta.non_zero_count << ", none zero percentage: " << meta.non_zero_count / meta.rows_count << std::endl;

		perform_measurement(mtx, reader);
	}

	return 0;
}