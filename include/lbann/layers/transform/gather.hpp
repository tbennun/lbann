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

#ifndef LBANN_LAYERS_TRANSFORM_GATHER_HPP_INCLUDED
#define LBANN_LAYERS_TRANSFORM_GATHER_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {

/** @brief Gather values from specified tensor indices
 *
 *  Expects two input tensors: an @f$ N @f$-D data tensor and a 1D
 *  index vector. For 1D data:
 *  @f[
 *    y[i] = x[\text{ind}[i]]
 *  @f]
 *  If an index is out-of-range, the corresponding output is set to
 *  zero.
 *
 *  For higher-dimensional data, the layer performs a gather along one
 *  dimension. For example, with 2D data and axis=1,
 *  @f[
 *    y[i,j] = x[i,\text{ind}[j]]
 *  @f]
 *  Currently, only 1D and 2D data is supported.
 *
 *  The size of the the output tensor along the gather dimension is
 *  equal to the size of the index vector. The remaining dimensions of
 *  the output tensor are identical to the data tensor.
 *
 *  @todo Support higher-dimensional data
 */
template <typename TensorDataType,
          data_layout Layout = data_layout::DATA_PARALLEL,
          El::Device Device = El::Device::CPU>
class gather_layer : public data_type_layer<TensorDataType>
{
  static_assert(Layout == data_layout::DATA_PARALLEL,
                "gather layer only supports data parallel layout");

public:
  gather_layer(const int axis);
  gather_layer(const gather_layer& other) = default;
  gather_layer& operator=(const gather_layer& other) = default;

  gather_layer* copy() const override;

  /** @name Serialization */
  ///@{

  template <typename ArchiveT>
  void serialize(ArchiveT& ar);

  ///@}

  std::string get_type() const override;
  data_layout get_data_layout() const override;
  El::Device get_device_allocation() const override;

protected:
  friend class cereal::access;
  gather_layer() : gather_layer(-1) {}
  void setup_dims(DataReaderMetaData& dr_metadata) override;
  void fp_compute() override;
  void bp_compute() override;

private:
  int m_gather_axis;
};

// =========================================================
// Implementation
// =========================================================

template <typename TensorDataType, data_layout Layout, El::Device Device>
gather_layer<TensorDataType, Layout, Device>::gather_layer(const int axis)
  : data_type_layer<TensorDataType>(nullptr), m_gather_axis{axis}
{
  this->m_expected_num_parent_layers = 2;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
gather_layer<TensorDataType, Layout, Device>*
gather_layer<TensorDataType, Layout, Device>::copy() const
{
  return new gather_layer(*this);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
std::string gather_layer<TensorDataType, Layout, Device>::get_type() const
{
  return "gather";
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
data_layout
gather_layer<TensorDataType, Layout, Device>::get_data_layout() const
{
  return Layout;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
El::Device
gather_layer<TensorDataType, Layout, Device>::get_device_allocation() const
{
  return Device;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void gather_layer<TensorDataType, Layout, Device>::setup_dims(
  DataReaderMetaData& dr_metadata)
{
  data_type_layer<TensorDataType>::setup_dims(dr_metadata);

  // Tensor dimensions
  const auto& input0_dims = this->get_input_dims(0);
  const auto& input1_dims = this->get_input_dims(1);
  // Only support 1D indices
  const auto is_indices_not_1D = input1_dims.size() != 1;

  // Only support 1D or 2D values
  const auto is_values_1D = input0_dims.size() == 1;
  const auto is_values_2D = input0_dims.size() == 2;

  const bool along_axis_0 = this->m_gather_axis == 0;
  if (is_values_2D) {
    if (this->m_gather_axis == -1) {
      LBANN_ERROR(this->get_type(),
                  " Layer \"",
                  this->get_name(),
                  "\" ",
                  "has 2D input, but does not set a  gather axis.",
                  "Axis must be either set to 0 or 1");
    }
  }
  if (is_values_1D) {
    this->set_output_dims(input1_dims);
  }
  else {
    //
    if (along_axis_0) {
      this->set_output_dims(std::vector<int>{input1_dims[0], input0_dims[1]});
    }
    else {
      this->set_output_dims(std::vector<int>{input0_dims[0], input1_dims[0]});
    }
  }

  auto dims_to_str = [](const std::vector<int>& dims) -> std::string {
    std::ostringstream ss;
    for (size_t i = 0; i < dims.size(); ++i) {
      ss << (i > 0 ? "x" : "") << dims[i];
    }
    return ss.str();
  };

  // Make sure input tensors have supported numbers of dimensions

  if (is_indices_not_1D || !(is_values_1D || is_values_2D)) {
    const auto& parent0 = this->get_parent_layer(0);
    const auto& parent1 = this->get_parent_layer(1);
    LBANN_ERROR(this->get_type(),
                " layer \"",
                this->get_name(),
                "\" ",
                "has input tensors with different numbers of dimensions ",
                "(",
                parent0.get_type(),
                " layer \"",
                parent0.get_name(),
                "\" ",
                "outputs ",
                dims_to_str(input0_dims),
                ", ",
                parent1.get_type(),
                " layer \"",
                parent1.get_name(),
                "\" ",
                "outputs ",
                dims_to_str(input1_dims),
                ")");
  }

  // Check that tensors are 1D
  /// @todo Support gathering from/into higher-order tensors
  if (!is_values_1D && !is_values_2D) {
    LBANN_ERROR(this->get_type(),
                " layer \"",
                this->get_name(),
                "\" ",
                "attempted to gather from a ",
                input0_dims.size(),
                "-D tensor ",
                "(",
                dims_to_str(input0_dims),
                "), "
                "but the gather layer currently only supports ",
                "gathering from a 1-D or 2-D tensor");
  }
}

#ifndef LBANN_GATHER_LAYER_INSTANTIATE
#define PROTO_DEVICE(T, Device)                                                \
  extern template class gather_layer<T, data_layout::DATA_PARALLEL, Device>
#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE
#endif // LBANN_GATHER_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYERS_TRANSFORM_GATHER_HPP_INCLUDED
