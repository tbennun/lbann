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

#include "lbann/transforms/scale.hpp"
#include "lbann/utils/memory.hpp"
#include <transforms.pb.h>

namespace lbann {
namespace transform {

void scale::apply(utils::type_erased_matrix& data, std::vector<size_t>&)
{
  // Currently only works on DataTypes.
  // Need to decide how to handle uint8_t matrices.
  auto& mat = data.template get<DataType>();
  if (mat.Height() != mat.LDim()) {
    LBANN_ERROR("Scaling non-contiguous matrix not supported.");
  }
  // Don't use El::Scale because it spawns OpenMP threads.
  DataType* __restrict__ buf = mat.Buffer();
  const El::Int size = mat.Height() * mat.Width();
  for (El::Int i = 0; i < size; ++i) {
    buf[i] *= m_scale;
  }
}

std::unique_ptr<transform>
build_scale_transform_from_pbuf(google::protobuf::Message const& msg)
{
  auto const& params = dynamic_cast<lbann_data::Transform::Scale const&>(msg);
  return std::make_unique<scale>(params.scale());
}

} // namespace transform
} // namespace lbann
