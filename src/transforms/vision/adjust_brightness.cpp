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

#include "lbann/transforms/vision/adjust_brightness.hpp"
#include "lbann/utils/dim_helpers.hpp"
#include "lbann/utils/memory.hpp"
#include "lbann/utils/opencv.hpp"

#include <transforms.pb.h>

namespace lbann {
namespace transform {

void adjust_brightness::apply(utils::type_erased_matrix& data,
                              std::vector<size_t>& dims)
{
  // Adjusting the brightness is simply scaling by a constant value
  // taking care to saturate.
  cv::Mat src = utils::get_opencv_mat(data, dims);
  if (!src.isContinuous()) {
    // This should not occur, but just in case.
    LBANN_ERROR("Do not support non-contiguous OpenCV matrices.");
  }
  uint8_t* __restrict__ src_buf = src.ptr();
  const size_t size = get_linear_size(dims);
  for (size_t i = 0; i < size; ++i) {
    src_buf[i] = cv::saturate_cast<uint8_t>(src_buf[i] * m_factor);
  }
}

std::unique_ptr<transform> build_adjust_brightness_transform_from_pbuf(
  google::protobuf::Message const& msg)
{
  auto const& params =
    dynamic_cast<lbann_data::Transform::AdjustBrightness const&>(msg);
  return std::make_unique<adjust_brightness>(params.factor());
}

} // namespace transform
} // namespace lbann
