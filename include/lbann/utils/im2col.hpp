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

#ifndef LBANN_UTILS_IM2COL_HPP
#define LBANN_UTILS_IM2COL_HPP

#include "lbann/base.hpp"

namespace lbann {

/// Rearrange image blocks into matrix columns
/** The 'col' matrix is generated from the 'im' tensor im by shifting
 *  a window across im. Each column of col is produced by positioning
 *  the window, extracting entries from im, and flattening.
 *  @param im               im tensor, represented as a column vector.
 *  @param col              col matrix. Height should be equal to
 *                          window size and width equal to number of
 *                          window shifts. Data should be contiguous.
 *  @param num_channels     Number of channels in im tensor.
 *  @param im_num_dims      Number of dimensions in im tensor.
 *  @param im_dims          im tensor dimensions.
 *  @param im_pads          Zero pads for im tensor.
 *  @param window_dims      Dimensions of window.
 *  @param window_strides   Window shift strides.
 */
template <typename TensorDataType>
void im2col(const CPUMatDT<TensorDataType>& im,
            CPUMatDT<TensorDataType>& col,
            int num_channels,
            int im_num_dims,
            const int* im_dims,
            const int* im_pads,
            const int* window_dims,
            const int* window_strides);

#ifdef LBANN_HAS_GPU
template <typename TensorDataType>
void im2col(const El::Matrix<TensorDataType, El::Device::GPU>& im,
            El::Matrix<TensorDataType, El::Device::GPU>& col,
            const int num_channels,
            const int im_num_dims,
            const int* im_dims,
            const int* im_pads,
            const int* window_dims,
            const int* window_strides,
            const El::SyncInfo<El::Device::GPU>& sync_info);
#endif // LBANN_HAS_GPU

/** Get the height and the width of col matrix.
 */
std::pair<size_t, size_t> get_im2col_output_size(const int num_samples,
                                                 const int num_channels,
                                                 const int im_num_dims,
                                                 const int* im_dims,
                                                 const int* im_pads,
                                                 const int* window_dims,
                                                 const int* window_strides);

/// Rearrange matrix columns into image blocks
/** This is approximately the inverse of im2col. The output tensor im
 *  is produced from the input matrix col by shifting a window across
 *  im. Each column of col is matched with the corresponding window
 *  position and corresponding entries are added to im.
 *  @param col              col matrix. Height should be equal to
 *                          window size and width equal to number of
 *                          window shifts. Data should be contiguous.
 *  @param im               im tensor, represented as a column vector.
 *  @param num_channels     Number of channels in im tensor.
 *  @param im_num_dims      Number of dimensions in im tensor.
 *  @param im_dims          im tensor dimensions.
 *  @param im_pads          Zero pads for im tensor.
 *  @param window_dims      Dimensions of window.
 *  @param window_strides   Window shift strides.
 */
template <typename TensorDataType>
void col2im(const CPUMatDT<TensorDataType>& col,
            CPUMatDT<TensorDataType>& im,
            int num_channels,
            int im_num_dims,
            const int* im_dims,
            const int* im_pads,
            const int* window_dims,
            const int* window_strides);

/// Rearrange matrix columns into image blocks
/** This is approximately the inverse of im2col. The output tensor im
 *  is produced from the input matrix col by shifting a window across
 *  im. Each column of col is matched with the corresponding window
 *  position and corresponding entries are reduced to im.
 *  @param col              col matrix. Height should be equal to
 *                          window size and width equal to number of
 *                          window shifts. Data should be contiguous.
 *  @param im               im tensor, represented as a column vector.
 *  @param num_channels     Number of channels in im tensor.
 *  @param im_num_dims      Number of dimensions in im tensor.
 *  @param im_dims          im tensor dimensions.
 *  @param im_pads          Zero pads for im tensor.
 *  @param window_dims      Dimensions of window.
 *  @param window_strides   Window shift strides.
 *  @param reduction_op     Reduction operation.
 */
template <typename TensorDataType>
void col2im(const CPUMatDT<TensorDataType>& col,
            CPUMatDT<TensorDataType>& im,
            int num_channels,
            int im_num_dims,
            const int* im_dims,
            const int* im_pads,
            const int* window_dims,
            const int* window_strides,
            std::function<TensorDataType(const TensorDataType&,
                                         const TensorDataType&)> reduction_op);

/// Rearrange 1x1 image blocks into matrix columns
/** This is an optimized implementation of im2col when the window has
 *  a size of one, there is no padding, and the window stride is
 *  one. im2col will automatically call this routine if it detects a
 *  1x1 im2col.
 */
template <typename TensorDataType>
void im2col_1x1(const TensorDataType* input_buffer,
                TensorDataType* output_buffer,
                int num_channels,
                int num_input_dims,
                const int* input_dims);

/// Rearrange 2D image blocks into matrix columns
/** This is an optimized implementation of im2col for 2D data. im2col
 *  will automatically call this routine if it detects 2D data.
 */
template <typename TensorDataType>
void im2col_2d(const TensorDataType* __restrict__ input_buffer,
               TensorDataType* __restrict__ output_buffer,
               int input_dim_x,
               int input_dim_y,
               int input_pad_x,
               int input_pad_y,
               int num_channels,
               int window_dim_x,
               int window_dim_y,
               int offset_stride_x,
               int offset_stride_y);

/// Rearrange matrix columns into 1x1 image blocks
/** This is an optimized implementation of col2im when the window has
 *  a size of one, there is no padding, and the window stride is
 *  one. col2im will automatically call this routine if it detects a
 *  1x1 col2im.
 */
template <typename TensorDataType>
void col2im_1x1(const TensorDataType* input_buffer,
                TensorDataType* output_buffer,
                const int num_channels,
                const int num_output_dims,
                const int* output_dims);

/// Rearrange matrix columns into 2D image blocks
/** This is an optimized implementation of col2im for 2D data. col2im
 *  will automatically call this routine if it detects 2D data.
 */
template <typename TensorDataType>
void col2im_2d(const TensorDataType* __restrict__ input_buffer,
               TensorDataType* __restrict__ output_buffer,
               int output_dim_x,
               int output_dim_y,
               int output_pad_x,
               int output_pad_y,
               int num_channels,
               int window_dim_x,
               int window_dim_y,
               int offset_stride_x,
               int offset_stride_y);

} // namespace lbann

#endif // LBANN_UTILS_IM2COL_HPP
