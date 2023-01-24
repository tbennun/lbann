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

#ifndef LBANN_TRANSFORMS_SCALE_HPP_INCLUDED
#define LBANN_TRANSFORMS_SCALE_HPP_INCLUDED

#include "lbann/transforms/transform.hpp"

#include <google/protobuf/message.h>

#include <memory>

namespace lbann {
namespace transform {

/** Scale data by a constant. */
class scale : public transform
{
public:
  /** Scale all data by scale_val. */
  scale(float scale_val) : transform(), m_scale(scale_val) {}

  transform* copy() const override { return new scale(*this); }

  std::string get_type() const override { return "scale"; }

  void apply(utils::type_erased_matrix& data,
             std::vector<size_t>& dims) override;

private:
  /** Amount to scale data by. */
  float m_scale;
};

// Builder function
std::unique_ptr<transform>
build_scale_transform_from_pbuf(google::protobuf::Message const&);

} // namespace transform
} // namespace lbann

#endif // LBANN_TRANSFORMS_SCALE_HPP_INCLUDED
