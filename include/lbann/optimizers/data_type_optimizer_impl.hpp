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

#ifndef LBANN_OPTIMIZERS_DATA_TYPE_OPTIMIZER_IMPL_HPP_INCLUDED
#define LBANN_OPTIMIZERS_DATA_TYPE_OPTIMIZER_IMPL_HPP_INCLUDED

#include "lbann/utils/serialize.hpp"
#include "lbann/utils/timer.hpp"
#include "lbann/weights/data_type_weights.hpp"

#include "lbann/optimizers/data_type_optimizer.hpp"

namespace lbann {

template <typename TensorDataType>
data_type_optimizer<TensorDataType>::data_type_optimizer(
  TensorDataType learning_rate)
  : m_learning_rate(learning_rate)
{}

template <typename TensorDataType>
data_type_optimizer<TensorDataType>::data_type_optimizer(
  const data_type_optimizer<TensorDataType>& other)
  : BaseType(other),
    m_weights(other.m_weights),
    m_gradient(other.m_gradient ? other.m_gradient->Copy() : nullptr),
    m_gradient_v(other.m_gradient_v ? other.m_gradient_v->Copy() : nullptr),
    m_learning_rate(other.m_learning_rate)
{}

template <typename TensorDataType>
data_type_optimizer<TensorDataType>&
data_type_optimizer<TensorDataType>::operator=(
  const data_type_optimizer<TensorDataType>& other)
{
  optimizer::operator=(other);
  m_weights = other.m_weights;
  m_gradient.reset(other.m_gradient ? other.m_gradient->Copy() : nullptr);
  m_gradient_v.reset(other.m_gradient_v ? other.m_gradient_v->Copy() : nullptr);
  m_learning_rate = other.m_learning_rate;
  return *this;
}

template <typename TensorDataType>
description data_type_optimizer<TensorDataType>::get_description() const
{
  description desc = optimizer::get_description();
  desc.add("Learning rate", m_learning_rate);
  return desc;
}

template <typename TensorDataType>
auto data_type_optimizer<TensorDataType>::get_weights() -> WeightsType&
{
  // Item 3, p. 23 in "Effective C++", 3rd ed., by Scott Meyers
  return const_cast<WeightsType&>(
    static_cast<const data_type_optimizer&>(*this).get_weights());
}

template <typename TensorDataType>
auto data_type_optimizer<TensorDataType>::get_weights() const
  -> const WeightsType&
{
  if (m_weights == nullptr) {
    LBANN_ERROR("attempted to access the weights being optimized "
                "before they are set");
  }
  return *m_weights;
}

template <typename TensorDataType>
auto data_type_optimizer<TensorDataType>::get_gradient() -> AbsDistMatrixType&
{

  // Make sure gradient matrix has been setup
  if (m_gradient == nullptr) {
    LBANN_ERROR("attempted to access gradient before it is set up");
  }

  // Make sure gradient values are ready
  this->start_gradient_allreduce();
  this->finish_gradient_allreduce();

  // Gather all gradients to the master precision
  this->accumulate_all_gradient_contributions(*m_gradient);

  // Return gradient
  return *m_gradient;
}

template <typename TensorDataType>
void data_type_optimizer<TensorDataType>::setup(weights* w_in)
{
  if (auto* w = dynamic_cast<WeightsType*>(w_in))
    this->setup(w);
  else
    LBANN_ERROR("Incompatible weights type.");
}

template <typename TensorDataType>
void data_type_optimizer<TensorDataType>::setup(WeightsType* w)
{
  this->setup_base(w);
}

template <typename TensorDataType>
void data_type_optimizer<TensorDataType>::setup_base(WeightsType* w)
{
  this->set_comm(w->get_comm());
  this->clear_gradient();

  // Set weights being optimized
  if (w != nullptr) {
    set_weights(w);
  }
  if (m_weights == nullptr) {
    LBANN_ERROR("attempted to setup optimizer without weights");
  }

  // Initialize matrices
  const auto& height = m_weights->get_matrix_height();
  const auto& width = m_weights->get_matrix_width();
  const AbsDistMatrixType& values = m_weights->get_values();
  m_gradient.reset(AbsDistMatrixType::Instantiate(values.DistData()));
  m_gradient->AlignWith(values);
  m_gradient->Resize(height, width);
  m_gradient_v.reset(AbsDistMatrixType::Instantiate(values.DistData()));
  m_gradient_v->AlignWith(values);
#ifdef HYDROGEN_HAVE_CUB
  if (m_gradient_v->GetLocalDevice() == El::Device::GPU) {
    m_gradient_v->Matrix().SetMemoryMode(1); // CUB GPU memory pool
  }
#endif // HYDROGEN_HAVE_CUB
}

template <typename TensorDataType>
double data_type_optimizer<TensorDataType>::get_learning_rate() const
{
  return m_learning_rate;
}

template <typename TensorDataType>
void data_type_optimizer<TensorDataType>::set_learning_rate(
  double learning_rate)
{
  m_learning_rate = learning_rate;
}

template <typename TensorDataType>
void data_type_optimizer<TensorDataType>::step()
{
  if (m_weights == nullptr) {
    LBANN_ERROR("attempted to perform optimization step without weights");
  }
  const auto start_time = get_time();
  this->step_compute(m_weights->get_values(), this->get_gradient());
  this->inc_step_time(get_time() - start_time);
}

template <typename TensorDataType>
std::tuple<El::Int, El::Int, El::DistData>
data_type_optimizer<TensorDataType>::get_matrix_info() const
{
  auto const& w = this->get_weights();
  return {w.get_matrix_height(),
          w.get_matrix_width(),
          w.get_matrix_distribution()};
}

template <typename TensorDataType>
template <class Archive>
void data_type_optimizer<TensorDataType>::serialize(Archive& ar)
{
  ar(cereal::base_class<optimizer>(this), CEREAL_NVP(m_learning_rate));
}

} // namespace lbann

#endif // LBANN_OPTIMIZERS_DATA_TYPE_OPTIMIZER_IMPL_HPP_INCLUDED
