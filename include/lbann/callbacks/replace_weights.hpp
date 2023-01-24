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
//
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CALLBACKS_CALLBACK_REPLACE_WEIGHTS_HPP_INCLUDED
#define LBANN_CALLBACKS_CALLBACK_REPLACE_WEIGHTS_HPP_INCLUDED

#include <utility>

#include "lbann/callbacks/callback.hpp"

namespace lbann {
namespace callback {

/**
 *  Weights/parameters replacement on k-batch end
 *  Currently support replacing weights/parameters using layer names
 *  Can easily be extended to support replacement by weights name
 *  Given two layers specified in prototext, weights are copied from source
 * layer to destination layer.
 */
class replace_weights : public callback_base
{
public:
  replace_weights(std::vector<std::string> src,
                  std::vector<std::string> dst,
                  int batch_interval = 1)
    : callback_base(batch_interval),
      m_src_layer_names(std::move(src)),
      m_dst_layer_names(std::move(dst))
  {
    if (m_src_layer_names.size() != m_dst_layer_names.size())
      LBANN_ERROR("In replace weights callback: number of src and dest layers "
                  "does not match.");
  }

  replace_weights(const replace_weights&) = default;
  replace_weights& operator=(const replace_weights&) = default;
  replace_weights* copy() const override { return new replace_weights(*this); }
  void setup(model* m) override;
  void on_batch_end(model* m) override;

  std::string name() const override { return "replace weights"; }

private:
  std::vector<std::string> m_src_layer_names, m_dst_layer_names;
  std::vector<Layer*> m_src_layers, m_dst_layers;
};

// Builder function
std::unique_ptr<callback_base>
build_replace_weights_callback_from_pbuf(const google::protobuf::Message&,
                                         std::shared_ptr<lbann_summary> const&);

} // namespace callback
} // namespace lbann

#endif // LBANN_CALLBACKS_CALLBACK_REPLACE_WEIGHTS_HPP_INCLUDED
