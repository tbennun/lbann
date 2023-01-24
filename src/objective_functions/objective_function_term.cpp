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

#include "lbann/objective_functions/objective_function_term.hpp"
#include "lbann/models/model.hpp"
#include "lbann/utils/serialize.hpp"

namespace lbann {

objective_function_term::objective_function_term(EvalType scale_factor)
  : m_scale_factor(scale_factor)
{
  if (m_scale_factor == EvalType(0)) {
    m_scale_factor = EvalType(1);
  }
}

template <class Archive>
void objective_function_term::serialize(Archive& ar)
{
  ar(CEREAL_NVP(m_scale_factor), CEREAL_NVP(m_layers), CEREAL_NVP(m_weights));
}

void objective_function_term::setup(model& m) { m_comm = m.get_comm(); }

std::vector<ViewingLayerPtr> objective_function_term::get_layer_pointers() const
{
  return m_layers;
}

void objective_function_term::set_layer_pointers(
  std::vector<ViewingLayerPtr> layers)
{
  m_layers = std::move(layers);
}

std::vector<ViewingWeightsPtr>
objective_function_term::get_weights_pointers() const
{
  return m_weights;
}

void objective_function_term::set_weights_pointers(
  std::vector<ViewingWeightsPtr> w)
{
  m_weights = std::move(w);
}

} // namespace lbann

#define LBANN_CLASS_NAME objective_function_term
#include <lbann/macros/register_class_with_cereal.hpp>
