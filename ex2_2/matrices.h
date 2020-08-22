#include "matrix_reader.h"

#include <memory>

struct matrix_rows_statistic
{
  unsigned int min_elements_in_rows {};
  unsigned int max_elements_in_rows {};
  unsigned int avg_elements_in_rows {};
  double elements_in_rows_std_deviation {};
};

matrix_rows_statistic get_rows_statistics (const matrix_reader::matrix_class::matrix_meta& meta, const unsigned int* row_ptr);

class csr_matrix_class
{
public:
  csr_matrix_class () = delete;
  explicit csr_matrix_class (const matrix_reader::matrix_class& matrix, bool row_ptr_only = false);

  const matrix_reader::matrix_class::matrix_meta meta;

  size_t get_matrix_size() const;

public:
  std::unique_ptr<float[]> data;
  std::unique_ptr<unsigned int[]> columns;
  std::unique_ptr<unsigned int[]> row_ptr;
};

class ell_matrix_class
{
public:
  ell_matrix_class() = delete;
  explicit ell_matrix_class (csr_matrix_class& matrix);
  ell_matrix_class (csr_matrix_class& matrix, unsigned int elements_in_row_arg);

  static size_t estimate_size (csr_matrix_class& matrix);

  const matrix_reader::matrix_class::matrix_meta meta;

  size_t get_matrix_size() const;

public:
  std::unique_ptr<float[]> data;
  std::unique_ptr<unsigned int[]> columns;

  unsigned int elements_in_rows = 0;
};

class coo_matrix_class
{
public:
  coo_matrix_class() = delete;
  explicit coo_matrix_class (csr_matrix_class& matrix);
  coo_matrix_class (csr_matrix_class& matrix, unsigned int element_start);

  const matrix_reader::matrix_class::matrix_meta meta;

  size_t get_matrix_size () const;

public:
  std::unique_ptr<float[]> data;
  std::unique_ptr<unsigned int[]> cols;
  std::unique_ptr<unsigned int[]> rows;

private:
  size_t elements_count {};
};