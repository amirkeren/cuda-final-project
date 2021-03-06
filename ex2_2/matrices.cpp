#include "matrices.h"

#include <vector>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <limits>
#include <chrono>
#include <cmath>

csr_matrix_class::csr_matrix_class(const matrix_reader::matrix_class& matrix, bool row_ptr_only): meta(matrix.meta)
{
  if (meta.matrix_storage_scheme != matrix_reader::matrix_class::storage_scheme::general)
  {
	throw std::runtime_error("Only general matrices are supported");
  }

  if (!row_ptr_only)
  {
    data.reset (new float[meta.non_zero_count]);
    columns.reset (new unsigned int[meta.non_zero_count]);
  }

  row_ptr.reset (new unsigned int[meta.rows_count + 1]);
  std::fill_n (row_ptr.get (), meta.rows_count + 1, 0u);

  auto src_rows = matrix.get_row_ids ();
  auto src_cols = matrix.get_col_ids ();
  auto src_data = matrix.get_dbl_data ();

  for (unsigned int i = 0; i < meta.non_zero_count; i++)
  {
	  row_ptr[src_rows[i]]++;
  }

  unsigned int ptr = 0;
  for (unsigned int row = 0; row < meta.rows_count + 1; row++)
  {
    const unsigned int tmp = row_ptr[row];
    row_ptr[row] = ptr;
    ptr += tmp;
  }

  if (!row_ptr_only)
  {
    std::unique_ptr<unsigned int[]> row_element_id(new unsigned int[meta.rows_count]);
    std::fill_n (row_element_id.get (), meta.rows_count, 0u);

    for (unsigned int i = 0; i < meta.non_zero_count; i++)
    {
      const unsigned int row = src_rows[i];
      const unsigned int element_offset = row_ptr[row] + row_element_id[row]++;
      data[element_offset] = src_data[i];
      columns[element_offset] = src_cols[i];
    }

    std::vector<unsigned int> permutation;
    std::vector<unsigned int> tmp_columns;
    std::vector<float> tmp_data;
    for (unsigned int i = 0; i < meta.rows_count; i++)
    {
      const auto row_begin = row_ptr[i];
      const auto row_end = row_ptr[i + 1];
      const auto n_elements = row_end - row_begin;

      permutation.resize (n_elements);
      tmp_columns.resize (n_elements);
      tmp_data.resize (n_elements);

      std::copy_n (data.get () + row_begin, n_elements, tmp_data.data ());
      std::copy_n (columns.get () + row_begin, n_elements, tmp_columns.data ());

      std::iota (permutation.begin (), permutation.end (), 0);
      std::sort (permutation.begin (), permutation.end (), [&] (const unsigned int &l, const unsigned int &r) {
        return columns[row_begin + l] < columns[row_begin + r];
      });

      for (unsigned int element = 0; element < n_elements; element++)
      {
        data[row_begin + element] = tmp_data[permutation[element]];
        columns[row_begin + element] = tmp_columns[permutation[element]];
      }
    }
  }
}

size_t csr_matrix_class::get_matrix_size() const
{
  return meta.non_zero_count;
}

matrix_rows_statistic get_rows_statistics(const matrix_reader::matrix_class::matrix_meta& meta, const unsigned int* row_ptr)
{
  matrix_rows_statistic statistic {};
  statistic.min_elements_in_rows = std::numeric_limits<unsigned int>::max() - 1;

  unsigned int sum_elements_in_rows = 0;
  for (unsigned int row = 0; row < meta.rows_count; row++)
  {
    const auto elements_in_row = row_ptr[row + 1] - row_ptr[row];

	if (elements_in_row > statistic.max_elements_in_rows)
	{
		statistic.max_elements_in_rows = elements_in_row;
	}

	if (elements_in_row < statistic.min_elements_in_rows)
	{
		statistic.min_elements_in_rows = elements_in_row;
	}

    sum_elements_in_rows += elements_in_row;
  }

  statistic.avg_elements_in_rows = sum_elements_in_rows / meta.rows_count;
  statistic.elements_in_rows_std_deviation = 0.0;

  for (unsigned int row = 0; row < meta.rows_count; row++)
  {
    const auto elements_in_row = row_ptr[row + 1] - row_ptr[row];
    statistic.elements_in_rows_std_deviation += std::pow (static_cast<double> (elements_in_row) - statistic.avg_elements_in_rows, 2);
  }
  statistic.elements_in_rows_std_deviation = std::sqrt (statistic.elements_in_rows_std_deviation / meta.rows_count);

  return statistic;
}

size_t ell_matrix_class::estimate_size(csr_matrix_class& matrix)
{
  const auto row_ptr = matrix.row_ptr.get ();
  matrix_rows_statistic row_statistics = get_rows_statistics (matrix.meta, row_ptr);
  size_t elements_in_rows = row_statistics.max_elements_in_rows;

  return elements_in_rows * matrix.meta.rows_count;
}

ell_matrix_class::ell_matrix_class (csr_matrix_class &matrix): meta(matrix.meta)
{
  if (meta.matrix_storage_scheme != matrix_reader::matrix_class::storage_scheme::general)
  {
	  throw std::runtime_error("Only general matrices are supported");
  }

  const auto row_ptr = matrix.row_ptr.get();
  const auto col_ptr = matrix.columns.get();

  matrix_rows_statistic row_statistics = get_rows_statistics(meta, row_ptr);
  elements_in_rows = row_statistics.max_elements_in_rows;

  const size_t elements_count = elements_in_rows * meta.rows_count;
  data.reset (new float[elements_count]);
  columns.reset (new unsigned int[elements_count]);

  std::fill_n (data.get(), elements_count, 0);
  std::fill_n (columns.get(), elements_count, 0);

  for (unsigned int row = 0; row < meta.rows_count; row++)
  {
    const auto start = row_ptr[row];
    const auto end = row_ptr[row + 1];

    for (auto element = start; element < end; element++)
    {
      data[row + (element - start) * meta.rows_count] = matrix.data[element];
      columns[row + (element - start) * meta.rows_count] = col_ptr[element];
    }
  }
}

ell_matrix_class::ell_matrix_class(csr_matrix_class& matrix, unsigned int elements_in_row_arg): meta(matrix.meta), elements_in_rows(elements_in_row_arg)
{
  const auto row_ptr = matrix.row_ptr.get ();
  const auto col_ptr = matrix.columns.get ();

  const unsigned int elements_count = get_matrix_size();
  data.reset (new float[elements_count]);
  columns.reset (new unsigned int[elements_count]);

  std::fill_n (data.get (), elements_count, 0);
  std::fill_n (columns.get (), elements_count, 0);

  for (unsigned int row = 0; row < meta.rows_count; row++)
  {
    const auto start = row_ptr[row];
    const auto end = row_ptr[row + 1];

    /// Skip extra elements
    for (auto element = start; element < std::min (start + elements_in_row_arg, end); element++)
    {
      data[row + (element - start) * meta.rows_count] = matrix.data[element];
      columns[row + (element - start) * meta.rows_count] = col_ptr[element];
    }
  }
}

size_t ell_matrix_class::get_matrix_size() const
{
  return meta.rows_count * elements_in_rows;
}

coo_matrix_class::coo_matrix_class(csr_matrix_class& matrix): meta(matrix.meta), elements_count(meta.non_zero_count)
{
  if (meta.matrix_storage_scheme != matrix_reader::matrix_class::storage_scheme::general)
  {
	  throw std::runtime_error("Only general matrices are supported");
  }

  data.reset (new float[meta.non_zero_count]);
  cols.reset (new unsigned int[meta.non_zero_count]);
  rows.reset (new unsigned int[meta.non_zero_count]);

  const auto row_ptr = matrix.row_ptr.get();
  const auto col_ptr = matrix.columns.get();

  unsigned int id = 0;
  for (unsigned int row = 0; row < meta.rows_count; row++)
  {
    const auto start = row_ptr[row];
    const auto end = row_ptr[row + 1];

    for (auto element = start; element < end; element++)
    {
      data[id] = matrix.data[element];
      cols[id] = col_ptr[element];
      rows[id] = row;
      id++;
    }
  }
}

coo_matrix_class::coo_matrix_class (csr_matrix_class& matrix, unsigned int element_start): meta(matrix.meta)
{
  const auto row_ptr = matrix.row_ptr.get();
  const auto col_ptr = matrix.columns.get();

  for (unsigned int row = 0; row < meta.rows_count; row++)
  {
    const auto start = row_ptr[row];
    const auto end = row_ptr[row + 1];

    for (auto element = start; element < end; element++)
	   if (element - start >= element_start)
	   {
	       elements_count++;
	   }
  }

  data.reset (new float[get_matrix_size()]);
  cols.reset (new unsigned int[get_matrix_size()]);
  rows.reset (new unsigned int[get_matrix_size()]);

  unsigned int id = 0;
  for (unsigned int row = 0; row < meta.rows_count; row++)
  {
    const auto start = row_ptr[row];
    const auto end = row_ptr[row + 1];

    for (auto element = start; element < end; element++)
    {
      if (element - start >= element_start)
      {
        data[id] = matrix.data[element];
        cols[id] = col_ptr[element];
        rows[id] = row;
        id++;
      }
    }
  }
}

size_t coo_matrix_class::get_matrix_size() const
{
  return elements_count;
}