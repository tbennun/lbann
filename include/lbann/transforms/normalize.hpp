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

#ifndef LBANN_TRANSFORMS_NORMALIZE_HPP_INCLUDED
#define LBANN_TRANSFORMS_NORMALIZE_HPP_INCLUDED

#include "lbann/transforms/transform.hpp"
#include "lbann/utils/exception.hpp"

#include <google/protobuf/message.h>

#include <vector>

namespace lbann {
namespace transform {

/**
 * Normalize with mean and standard deviation.
 * This is done channel-wise for images. If the input does not have channels,
 * (e.g. it is not an image), it is treated as having one "channel".
 * This is only applicable after conversion to an LBANN CPUMat.
 */
class normalize : public transform
{
public:
  /** Apply channel-wise means and standard deviations. */
  normalize(std::vector<float> means, std::vector<float> stds)
    : transform(), m_means(means), m_stds(stds)
  {
    if (m_means.size() != m_stds.size()) {
      LBANN_ERROR("Normalize mean and std have different numbers of channels.");
    }
  }

  transform* copy() const override { return new normalize(*this); }

  std::string get_type() const override { return "normalize"; }

  bool supports_non_inplace() const override { return true; }

  void apply(utils::type_erased_matrix& data,
             std::vector<size_t>& dims) override;
  void apply(utils::type_erased_matrix& data,
             CPUMat& out,
             std::vector<size_t>& dims) override;

private:
  /** Channel-wise means. */
  std::vector<float> m_means;
  /** Channel-wise standard deviations. */
  std::vector<float> m_stds;
};

// Builder function
std::unique_ptr<transform>
build_normalize_transform_from_pbuf(google::protobuf::Message const&);

} // namespace transform
} // namespace lbann

#endif // LBANN_TRANSFORMS_NORMALIZED_CENTER_CROP_HPP_INCLUDED
