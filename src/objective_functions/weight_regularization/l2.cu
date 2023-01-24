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

#include "lbann/models/model.hpp"
#include "lbann/objective_functions/weight_regularization/l2.hpp"

#include "lbann/utils/gpu/helpers.hpp"

namespace lbann {

namespace {

template <typename TensorDataType, El::Int block_size>
__global__ void
accumulate_contribution_kernel(El::Int height,
                               El::Int width,
                               const TensorDataType* __restrict__ vals,
                               El::Int vals_ldim,
                               DataType* __restrict__ contribution)
{

  // Indices
  const El::Int tid = threadIdx.x;
  const El::Int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const El::Int nthreads = blockDim.x * gridDim.x;

  // Compute contributions for each thread
  TensorDataType private_contribution = 0;
  const auto& size = height * width;
  for (El::Int i = gid; i < size; i += nthreads) {
    const auto& row = i % height;
    const auto& col = i / height;
    const auto& val = vals[row + col * vals_ldim];
    private_contribution += val * val;
  }

  // Shared memory reduction to get contribution for each block
  /// @todo unroll loops
  __shared__ TensorDataType shared_contribution[block_size];
  shared_contribution[tid] = private_contribution;
  for (El::Int stride = block_size / 2; stride > 0; stride /= 2) {
    __syncthreads();
    if (tid < stride) {
      shared_contribution[tid] += shared_contribution[tid + stride];
    }
  }
  if (tid == 0) {
    gpu_lib::atomic_add(contribution, shared_contribution[0]);
  }
}

} // namespace

template <>
void l2_weight_regularization::accumulate_contribution<El::Device::GPU>(
  const El::Matrix<AccumulateDataType, El::Device::GPU>& vals,
  El::Matrix<AccumulateDataType, El::Device::GPU>& contribution)
{

  if (!vals.IsEmpty()) {
    auto multisync = El::MakeMultiSync(gpu::get_sync_info(contribution),
                                       gpu::get_sync_info(vals));

    const auto& size = vals.Height() * vals.Width();
    const El::Int block_size = 256;
    const auto& grid_size = (size + block_size - 1) / block_size;
    hydrogen::gpu::LaunchKernel(
      accumulate_contribution_kernel<AccumulateDataType, block_size>,
      grid_size,
      block_size,
      0,
      multisync,
      vals.Height(),
      vals.Width(),
      vals.LockedBuffer(),
      vals.LDim(),
      contribution.Buffer());
  }
}

} // namespace lbann
