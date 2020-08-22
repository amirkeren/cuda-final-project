#include <cuda_runtime.h>

#include "utils.h"

measurement_class::measurement_class (std::string format, double elapsed_arg, double load_store_bytes, double operations_count): elapsed(elapsed_arg), effective_bandwidth(load_store_bytes / (elapsed * giga)), computational_throughput(operations_count / (elapsed * giga)), matrix_format(move(format)) {}

void internal_free(char*& data)
{
	if (data)
	{
		cudaFree(data);
	}
	data = nullptr;
}

void internal_resize(char*& data, size_t new_size)
{
	internal_free(data);
	cudaMalloc(&data, new_size);
}