////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2022, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory.
// Written by the LBANN Research Team (B. Van Essen, et al.) listed in
// the CONTRIBUTORS file. <lbann-dev@llnl.gov>
//
// LLNL-CODE-697807.
// All rights reserved.
//
// This file is part of LBANN: Livermore Big Artificial Neural Network
// Toolkit. For details, see http://software.llnl.gov/LBANN or
// https://github.com/LLNL/LBANN.
//
// Licensed under the Apache License, Version 2.0 (the "Licensee"); you
// may not use this file except in compliance with the License.  You may
// obtain a copy of the License at:
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the license.
////////////////////////////////////////////////////////////////////////////////

namespace lbann {
namespace gpu_lib {
#if defined LBANN_HAS_CUDA
using namespace cuda;
#elif defined LBANN_HAS_ROCM
using namespace rocm;
#endif // LBANN_HAS_CUDA

// -------------------------------------------------------------
// Device properties
// -------------------------------------------------------------

inline void clip_grid_dims(dim3& grid_dims)
{
  const auto max_grid_dims_ = max_grid_dims();
  grid_dims.x = std::min(grid_dims.x, max_grid_dims_.x);
  grid_dims.y = std::min(grid_dims.y, max_grid_dims_.y);
  grid_dims.z = std::min(grid_dims.z, max_grid_dims_.z);
}

// -------------------------------------------------------------
// Device functions
// -------------------------------------------------------------
#if defined __CUDACC__ || defined __HIPCC__

// Unary math functions
#define WRAP_UNARY_MATH_FUNCTION(func)                                         \
  __device__ __forceinline__ float func(const float& x)                        \
  {                                                                            \
    return ::func##f(x);                                                       \
  }                                                                            \
  __device__ __forceinline__ double func(const double& x)                      \
  {                                                                            \
    return ::func(x);                                                          \
  }
template <typename T>
__device__ __forceinline__ T abs(const T& x)
{
  return x >= static_cast<T>(0) ? x : -x;
}
__device__ __forceinline__ float abs(const float& x) { return ::fabsf(x); }
__device__ __forceinline__ double abs(const double& x) { return ::fabs(x); }
WRAP_UNARY_MATH_FUNCTION(round)
WRAP_UNARY_MATH_FUNCTION(ceil)
WRAP_UNARY_MATH_FUNCTION(floor)
WRAP_UNARY_MATH_FUNCTION(sqrt)
WRAP_UNARY_MATH_FUNCTION(rsqrt)
WRAP_UNARY_MATH_FUNCTION(exp)
WRAP_UNARY_MATH_FUNCTION(expm1)
WRAP_UNARY_MATH_FUNCTION(log)
WRAP_UNARY_MATH_FUNCTION(log1p)
WRAP_UNARY_MATH_FUNCTION(cos)
WRAP_UNARY_MATH_FUNCTION(sin)
WRAP_UNARY_MATH_FUNCTION(tan)
WRAP_UNARY_MATH_FUNCTION(acos)
WRAP_UNARY_MATH_FUNCTION(asin)
WRAP_UNARY_MATH_FUNCTION(atan)
WRAP_UNARY_MATH_FUNCTION(cosh)
WRAP_UNARY_MATH_FUNCTION(sinh)
WRAP_UNARY_MATH_FUNCTION(tanh)
WRAP_UNARY_MATH_FUNCTION(acosh)
WRAP_UNARY_MATH_FUNCTION(asinh)
WRAP_UNARY_MATH_FUNCTION(atanh)
WRAP_UNARY_MATH_FUNCTION(erf)
WRAP_UNARY_MATH_FUNCTION(erfinv)
#undef WRAP_UNARY_MATH_FUNCTION

template <typename T>
__device__ __forceinline__ bool isfinite(T const& x)
{
  return ::isfinite(x);
}
template <typename T>
__device__ __forceinline__ bool isinf(T const& x)
{
  return ::isinf(x);
}
template <typename T>
__device__ __forceinline__ bool isnan(T const& x)
{
  return ::isnan(x);
}

// Binary math functions
#define WRAP_BINARY_MATH_FUNCTION(func)                                        \
  __device__ __forceinline__ float func(const float& x, const float& y)        \
  {                                                                            \
    return ::func##f(x, y);                                                    \
  }                                                                            \
  __device__ __forceinline__ double func(const double& x, const double& y)     \
  {                                                                            \
    return ::func(x, y);                                                       \
  }
template <typename T>
__device__ __forceinline__ T min(const T& x, const T& y)
{
  return y < x ? y : x;
}
__device__ __forceinline__ float min(const float& x, const float& y)
{
  return ::fminf(x, y);
}
__device__ __forceinline__ double min(const double& x, const double& y)
{
  return ::fmin(x, y);
}
template <typename T>
__device__ __forceinline__ T max(const T& x, const T& y)
{
  return y > x ? y : x;
}
__device__ __forceinline__ float max(const float& x, const float& y)
{
  return ::fmaxf(x, y);
}
__device__ __forceinline__ double max(const double& x, const double& y)
{
  return ::fmax(x, y);
}
__device__ __forceinline__ float mod(const float& x, const float& y)
{
  return ::fmodf(x, y);
}
__device__ __forceinline__ double mod(const double& x, const double& y)
{
  return ::fmod(x, y);
}
WRAP_BINARY_MATH_FUNCTION(pow)
#undef WRAP_BINARY_MATH_FUNCTION

__device__ __forceinline__ __half pow(const __half& x, const __half& y)
{
  return pow(float(x), float(y));
}

__device__ __forceinline__ __half mod(const __half& x, const __half& y)
{
  return mod(float(x), float(y));
}

// FIXME (TRB): I think this is right? Borrowed the values from the
// sourceforge half library.
template <>
__device__ __forceinline__ __half min<__half>()
{
  return __short_as_half(0x0400);
}
template <>
__device__ __forceinline__ __half max<__half>()
{
  return __short_as_half(0x7BFF);
}
template <>
__device__ __forceinline__ __half epsilon<__half>()
{
  return __short_as_half(0x1400);
}
template <>
__device__ __forceinline__ __half infinity<__half>()
{
  return __short_as_half(0x7C00);
}

// Array member functions
template <typename T, size_t N>
__host__ __device__ __forceinline__ size_t array<T, N>::size() const
{
  return N;
}
template <typename T, size_t N>
__host__ __device__ __forceinline__ T& array<T, N>::operator[](size_t i)
{
  return vals[i];
}
template <typename T, size_t N>
__host__ __device__ __forceinline__ const T&
array<T, N>::operator[](size_t i) const
{
  return vals[i];
}

#endif // __CUDACC__ || __HIPCC__

// -------------------------------------------------------------
// Helper functions for entrywise operations
// -------------------------------------------------------------
#if defined __CUDACC__ || defined __HIPCC__

namespace apply_entrywise_operator_impl {

/** @brief Apply entry-wise unary operator to 1D data
 *
 *  Block dims: bsize x 1 x 1
 *
 *  Grid dims: (size/bsize) x 1 x 1
 */
template <template <typename> class UnaryOperator, typename TensorDataType>
__global__ void unary_1d_kernel(size_t size,
                                const TensorDataType* __restrict__ input,
                                TensorDataType* __restrict__ output)
{
  const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t nthreads = blockDim.x * gridDim.x;
  UnaryOperator<TensorDataType> op;
  for (size_t i = gid; i < size; i += nthreads) {
    output[i] = op(input[i]);
  }
}

/** @brief Apply entry-wise unary operator to 2D data
 *
 *  Block dims: bsizex x bsizey x 1
 *
 *  Grid dims: (height/bsizex) x (width/bsizey) x 1
 */
template <template <typename> class UnaryOperator, typename TensorDataType>
__global__ void unary_2d_kernel(size_t height,
                                size_t width,
                                const TensorDataType* __restrict__ input,
                                size_t input_ldim,
                                TensorDataType* __restrict__ output,
                                size_t output_ldim)
{
  const size_t gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t gidy = threadIdx.y + blockIdx.y * blockDim.y;
  const size_t nthreadsx = blockDim.x * gridDim.x;
  const size_t nthreadsy = blockDim.y * gridDim.y;
  UnaryOperator<TensorDataType> op;
  for (size_t j = gidy; j < width; j += nthreadsy) {
    for (size_t i = gidx; i < height; i += nthreadsx) {
      const auto& x = input[i + j * input_ldim];
      auto& y = output[i + j * output_ldim];
      y = op(x);
    }
  }
}

/** @brief Apply entry-wise binary operator to 1D data
 *
 *  Block dims: bsize x 1 x 1
 *
 *  Grid dims: (size/bsize) x 1 x 1
 */
template <template <typename> class BinaryOperator, typename TensorDataType>
__global__ void binary_1d_kernel(size_t size,
                                 const TensorDataType* __restrict__ input1,
                                 const TensorDataType* __restrict__ input2,
                                 TensorDataType* __restrict__ output)
{
  const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t nthreads = blockDim.x * gridDim.x;
  BinaryOperator<TensorDataType> op;
  for (size_t i = gid; i < size; i += nthreads) {
    output[i] = op(input1[i], input2[i]);
  }
}

/** @brief Apply entry-wise binary operator to 2D data
 *
 *  Block dims: bsizex x bsizey x 1
 *
 *  Grid dims: (height/bsizex) x (width/bsizey) x 1
 */
template <template <typename> class BinaryOperator, typename TensorDataType>
__global__ void binary_2d_kernel(size_t height,
                                 size_t width,
                                 const TensorDataType* __restrict__ input1,
                                 size_t input1_ldim,
                                 const TensorDataType* __restrict__ input2,
                                 size_t input2_ldim,
                                 TensorDataType* __restrict__ output,
                                 size_t output_ldim)
{
  const size_t gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t gidy = threadIdx.y + blockIdx.y * blockDim.y;
  const size_t nthreadsx = blockDim.x * gridDim.x;
  const size_t nthreadsy = blockDim.y * gridDim.y;
  BinaryOperator<TensorDataType> op;
  for (size_t j = gidy; j < width; j += nthreadsy) {
    for (size_t i = gidx; i < height; i += nthreadsx) {
      const auto& x1 = input1[i + j * input1_ldim];
      const auto& x2 = input2[i + j * input2_ldim];
      auto& y = output[i + j * output_ldim];
      y = op(x1, x2);
    }
  }
}

} // namespace apply_entrywise_operator_impl

/** @brief Apply an entry-wise unary operator to GPU data.
 *
 *  The input and output data must be on GPU and must have the same
 *  dimensions.
 */
template <template <typename> class UnaryOp, typename TensorDataType>
void apply_entrywise_unary_operator(
  const El::AbstractMatrix<TensorDataType>& input,
  El::AbstractMatrix<TensorDataType>& output)
{

  // Check that input and output are valid
  if (input.GetDevice() != El::Device::GPU) {
    LBANN_ERROR("input is not on GPU");
  }
  else if (output.GetDevice() != El::Device::GPU) {
    LBANN_ERROR("output is not on GPU");
  }
  else if (input.Height() != output.Height() ||
           input.Width() != output.Width()) {
    LBANN_ERROR("input matrix dimensions "
                "(",
                input.Height(),
                " x ",
                input.Width(),
                ")"
                "don't match output matrix dimensions "
                "(",
                output.Height(),
                " x ",
                output.Width(),
                ")");
  }

  // Return immediately if no compute is required
  if (output.IsEmpty()) {
    return;
  }

  // Launch GPU kernel
  if (input.Contiguous() && output.Contiguous()) {
    dim3 block_dims, grid_dims;
    block_dims.x = 256;
    grid_dims.x =
      (output.Height() * output.Width() + block_dims.x - 1) / block_dims.x;
    gpu_lib::clip_grid_dims(grid_dims);
    auto multisync =
      El::MakeMultiSync(gpu::get_sync_info(output), gpu::get_sync_info(input));
    hydrogen::gpu::LaunchKernel(
      apply_entrywise_operator_impl::unary_1d_kernel<UnaryOp, TensorDataType>,
      grid_dims,
      block_dims,
      0,
      multisync,
      output.Height() * output.Width(),
      input.LockedBuffer(),
      output.Buffer());
  }
  else {
    dim3 block_dims, grid_dims;
    block_dims.x = 256;
    block_dims.y = 1;
    grid_dims.x = (output.Height() + block_dims.x - 1) / block_dims.x;
    grid_dims.y = (output.Width() + block_dims.y - 1) / block_dims.y;
    gpu_lib::clip_grid_dims(grid_dims);
    auto multisync =
      El::MakeMultiSync(gpu::get_sync_info(output), gpu::get_sync_info(input));
    hydrogen::gpu::LaunchKernel(
      apply_entrywise_operator_impl::unary_2d_kernel<UnaryOp, TensorDataType>,
      grid_dims,
      block_dims,
      0,
      multisync,
      input.Height(),
      input.Width(),
      input.LockedBuffer(),
      input.LDim(),
      output.Buffer(),
      output.LDim());
  }
}

/** @brief Apply an entry-wise binary operator to GPU data.
 *
 *  The input and output data must be on GPU and must have the same
 *  dimensions.
 */
template <template <typename> class BinaryOp, typename TensorDataType>
void apply_entrywise_binary_operator(
  const El::AbstractMatrix<TensorDataType>& input1,
  const El::AbstractMatrix<TensorDataType>& input2,
  El::AbstractMatrix<TensorDataType>& output)
{

  // Check that input and output are valid
  if (input1.GetDevice() != El::Device::GPU ||
      input2.GetDevice() != El::Device::GPU) {
    LBANN_ERROR("input is not on GPU");
  }
  else if (output.GetDevice() != El::Device::GPU) {
    LBANN_ERROR("output is not on GPU");
  }
  else if (input1.Height() != input2.Height() ||
           input1.Width() != input2.Width() ||
           input1.Height() != output.Height() ||
           input1.Width() != output.Width()) {
    LBANN_ERROR("input matrix dimensions "
                "(",
                input1.Height(),
                " x ",
                input1.Width(),
                ", ",
                input2.Height(),
                " x ",
                input2.Width(),
                ")"
                "don't match output matrix dimensions "
                "(",
                output.Height(),
                " x ",
                output.Width(),
                ")");
  }

  // Return immediately if no compute is required
  if (output.IsEmpty()) {
    return;
  }

  // Launch GPU kernel
  if (input1.Contiguous() && input2.Contiguous() && output.Contiguous()) {
    dim3 block_dims, grid_dims;
    block_dims.x = 256;
    grid_dims.x =
      (output.Height() * output.Width() + block_dims.x - 1) / block_dims.x;
    gpu_lib::clip_grid_dims(grid_dims);
    auto multisync = El::MakeMultiSync(gpu::get_sync_info(output),
                                       gpu::get_sync_info(input1),
                                       gpu::get_sync_info(input2));
    hydrogen::gpu::LaunchKernel(
      apply_entrywise_operator_impl::binary_1d_kernel<BinaryOp, TensorDataType>,
      grid_dims,
      block_dims,
      0,
      multisync,
      output.Height() * output.Width(),
      input1.LockedBuffer(),
      input2.LockedBuffer(),
      output.Buffer());
  }
  else {
    dim3 block_dims, grid_dims;
    block_dims.x = 256;
    block_dims.y = 1;
    grid_dims.x = (output.Height() + block_dims.x - 1) / block_dims.x;
    grid_dims.y = (output.Width() + block_dims.y - 1) / block_dims.y;
    gpu_lib::clip_grid_dims(grid_dims);
    auto multisync = El::MakeMultiSync(gpu::get_sync_info(output),
                                       gpu::get_sync_info(input1),
                                       gpu::get_sync_info(input2));
    hydrogen::gpu::LaunchKernel(
      apply_entrywise_operator_impl::binary_2d_kernel<BinaryOp, TensorDataType>,
      grid_dims,
      block_dims,
      0,
      multisync,
      output.Height(),
      output.Width(),
      input1.LockedBuffer(),
      input1.LDim(),
      input2.LockedBuffer(),
      input2.LDim(),
      output.Buffer(),
      output.LDim());
  }
}

/** Apply an entry-wise unary operator to GPU data.
 *  The input and output data must be on GPU, have the same
 *  dimensions, and be aligned.
 */
template <template <typename> class UnaryOperator, typename TensorDataType>
void apply_entrywise_unary_operator(
  const El::AbstractDistMatrix<TensorDataType>& input,
  El::AbstractDistMatrix<TensorDataType>& output)
{
  if (input.Height() != output.Height() || input.Width() != output.Width()) {
    LBANN_ERROR("input matrix dimensions "
                "(",
                input.Height(),
                " x ",
                input.Width(),
                ")"
                "don't match output matrix dimensions "
                "(",
                output.Height(),
                " x ",
                output.Width(),
                ")");
  }
  else if (input.DistData() != output.DistData()) {
    LBANN_ERROR("input and output matrix distributions don't match");
  }
  apply_entrywise_unary_operator<UnaryOperator>(input.LockedMatrix(),
                                                output.Matrix());
}

/** Apply an entry-wise binary operator to GPU data.
 *  The input and output data must be on GPU, have the same
 *  dimensions, and be aligned.
 */
template <template <typename> class BinaryOperator, typename TensorDataType>
void apply_entrywise_binary_operator(
  const El::AbstractDistMatrix<TensorDataType>& input1,
  const El::AbstractDistMatrix<TensorDataType>& input2,
  El::AbstractDistMatrix<TensorDataType>& output)
{
  if (input1.Height() != input2.Height() || input1.Width() != input2.Width() ||
      input1.Height() != output.Height() || input1.Width() != output.Width()) {
    LBANN_ERROR("input matrix dimensions "
                "(",
                input1.Height(),
                " x ",
                input1.Width(),
                ", ",
                input2.Height(),
                " x ",
                input2.Width(),
                ")"
                "don't match output matrix dimensions "
                "(",
                output.Height(),
                " x ",
                output.Width(),
                ")");
  }
  else if (input1.DistData() != input2.DistData() ||
           input1.DistData() != output.DistData()) {
    LBANN_ERROR("input and output matrix distributions don't match");
  }
  apply_entrywise_binary_operator<BinaryOperator>(input1.LockedMatrix(),
                                                  input2.LockedMatrix(),
                                                  output.Matrix());
}

#endif // __CUDACC__ || __HIPCC__

} // namespace gpu_lib
} // namespace lbann
