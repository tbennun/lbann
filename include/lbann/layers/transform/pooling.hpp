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

#ifndef LBANN_LAYER_POOLING_HPP_INCLUDED
#define LBANN_LAYER_POOLING_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"
#include "lbann/models/model.hpp"
#include "lbann/utils/dim_helpers.hpp"
#include "lbann/utils/dnn_enums.hpp"
#ifdef LBANN_HAS_DNN_LIB
#include "lbann/utils/dnn_lib/helpers.hpp"
#include "lbann/utils/dnn_lib/pooling.hpp"
#endif // LBANN_HAS_DNN_LIB
#include "lbann/utils/distconv.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/im2col.hpp"

#include <utility>
#include <vector>

namespace lbann {

inline pooling_mode to_pool_mode(std::string m)
{
#ifdef LBANN_DETERMINISTIC
  if (m == "max")
    return pooling_mode::MAX_DETERMINISTIC;
#else
  if (m == "max")
    return pooling_mode::MAX;
#endif // LBANN_DETERMINISTIC
  if (m == "average")
    return pooling_mode::AVERAGE_COUNT_INCLUDE_PADDING;
  if (m == "average_no_pad")
    return pooling_mode::AVERAGE_COUNT_EXCLUDE_PADDING;
  else {
    LBANN_ERROR("Invalid pooling mode requested.");
  }
}

#ifdef LBANN_HAS_DISTCONV
template <typename TensorDataType,
          data_layout T_layout = data_layout::DATA_PARALLEL,
          El::Device Dev = El::Device::CPU>
class pooling_distconv_adapter
  : public data_type_distconv_adapter<TensorDataType>
{
public:
  using TensorDevType =
    typename data_type_distconv_adapter<TensorDataType>::TensorDevType;
  pooling_distconv_adapter(Layer& layer)
    : data_type_distconv_adapter<TensorDataType>(layer)
  {}
  virtual ~pooling_distconv_adapter() = default;
  void setup_distributions(tensor_overlap_constraints& constraints) override;
  dc::Shape get_activations_local_shape(int index = 0) const override;
  void setup_layer(size_t workspace_capacity) override;
  void
  fp_compute(bool training = true); // training=true for max back-compatibility.
  void bp_compute();
  std::unique_ptr<dc::Pooling<TensorDataType>> m_pooling;
};
#endif // LBANN_HAS_DISTCONV

// Forward declaration
template <typename TensorDataType, data_layout T_layout, El::Device Dev>
class unpooling_layer;

template <typename TensorDataType,
          data_layout T_layout = data_layout::DATA_PARALLEL,
          El::Device Dev = El::Device::CPU>
class pooling_layer : public data_type_layer<TensorDataType>
{
  static_assert(T_layout == data_layout::DATA_PARALLEL,
                "pooling only supports DATA_PARALLEL");

private:
  /** Pooling mode. */
  pooling_mode m_pool_mode;

  /** Pooling window dimensions. */
  std::vector<int> m_pool_dims;
  /** Size of pooling window. */
  int m_pool_size;
  /** Pooling padding. */
  std::vector<int> m_pads;
  /** Pooling strides. */
  std::vector<int> m_strides;

  /** Input indices for max pooling.
   *  Each entry corresponds to a local entry in the activations
   *  matrix. The entry gives the index of the maximum entry within
   *  the pooling window.
   */
  std::vector<int> m_max_pool_indices;

#ifdef LBANN_HAS_DNN_LIB
  /** Pooling descriptor. */
  dnn_lib::PoolingDescriptor m_pooling_dnn_desc;
  /** Tensor DNN library descriptors. */
  dnn_lib::data_parallel_layer_tensor_manager<TensorDataType>
    m_tensors_dnn_desc;
#endif // LBANN_HAS_DNN_LIB

  friend class unpooling_layer<TensorDataType, T_layout, Dev>;

public:
  pooling_layer(lbann_comm* comm,
                int num_data_dims,
                int pool_dim,
                int pad,
                int stride,
                pooling_mode mode)
    : pooling_layer(comm,
                    num_data_dims,
                    std::vector<int>(num_data_dims, pool_dim),
                    std::vector<int>(num_data_dims, pad),
                    std::vector<int>(num_data_dims, stride),
                    mode)
  {}

  pooling_layer(lbann_comm* comm,
                int num_data_dims,
                std::vector<int> pool_dims,
                std::vector<int> pads,
                std::vector<int> strides,
                pooling_mode mode)
    : data_type_layer<TensorDataType>(comm),
      m_pool_mode(mode),
      m_pool_dims(pool_dims),
      m_pads(pads),
      m_strides(strides)
#ifdef LBANN_HAS_DNN_LIB
      ,
      m_tensors_dnn_desc(this)
#endif // LBANN_HAS_DNN_LIB
  {
    // Initialize input dimensions and pooling parameters
    m_pool_size = get_linear_size(m_pool_dims);
  }

  pooling_layer(const pooling_layer& other)
    : data_type_layer<TensorDataType>(other),
      m_pool_mode(other.m_pool_mode),
      m_pool_dims(other.m_pool_dims),
      m_pool_size(other.m_pool_size),
      m_pads(other.m_pads),
      m_strides(other.m_strides),
      m_max_pool_indices(other.m_max_pool_indices)
#ifdef LBANN_HAS_DNN_LIB
      ,
      m_pooling_dnn_desc(other.m_pooling_dnn_desc),
      m_tensors_dnn_desc(other.m_tensors_dnn_desc)
#endif // LBANN_HAS_DNN_LIB
  {
#ifdef LBANN_HAS_DNN_LIB
    m_tensors_dnn_desc.set_layer(this);
#endif // LBANN_HAS_DNN_LIB
  }

  pooling_layer& operator=(const pooling_layer& other)
  {
    data_type_layer<TensorDataType>::operator=(other);
    m_pool_mode = other.m_pool_mode;
    m_pool_dims = other.m_pool_dims;
    m_pool_size = other.m_pool_size;
    m_pads = other.m_pads;
    m_strides = other.m_strides;
    m_max_pool_indices = other.m_max_pool_indices;
#ifdef LBANN_HAS_DNN_LIB
    m_pooling_dnn_desc = other.m_pooling_dnn_desc;
    m_tensors_dnn_desc = other.m_tensors_dnn_desc;
    m_tensors_dnn_desc.set_layer(this);
#endif // LBANN_HAS_DNN_LIB
    return *this;
  }

  ~pooling_layer() override = default;

  pooling_layer* copy() const override { return new pooling_layer(*this); }

  /** @name Serialization */
  ///@{

  template <typename ArchiveT>
  void serialize(ArchiveT& ar);

  ///@}

  std::string get_type() const override { return "pooling"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }

#ifdef LBANN_HAS_ONNX
  void fill_onnx_node(onnx::GraphProto& graph) const override;
#endif // LBANN_HAS_ONNX

  description get_description() const override
  {
    auto desc = data_type_layer<TensorDataType>::get_description();
    std::stringstream ss;

    // Pool mode
    ss.str(std::string{});
    ss.clear();
    switch (m_pool_mode) {
    case pooling_mode::MAX:
      ss << "max";
      break;
    case pooling_mode::MAX_DETERMINISTIC:
      ss << "max (deterministic)";
      break;
    case pooling_mode::AVERAGE_COUNT_INCLUDE_PADDING:
      ss << "average";
      break;
    case pooling_mode::AVERAGE_COUNT_EXCLUDE_PADDING:
      ss << "average (no pad)";
      break;
    default:
      ss << "invalid";
    }
    desc.add("Pool mode", ss.str());

    // Pool dimensions
    ss.str(std::string{});
    ss.clear();
    for (size_t i = 0; i < m_pool_dims.size(); ++i) {
      ss << (i > 0 ? ", " : "") << m_pool_dims[i];
    }
    desc.add("Pool dimensions", ss.str());

    // Strides
    ss.str(std::string{});
    ss.clear();
    for (size_t i = 0; i < m_strides.size(); ++i) {
      ss << (i > 0 ? ", " : "") << m_strides[i];
    }
    desc.add("Strides", ss.str());

    // Pads
    ss.str(std::string{});
    ss.clear();
    for (size_t i = 0; i < m_pads.size(); ++i) {
      ss << (i > 0 ? ", " : "") << m_pads[i];
    }
    desc.add("Pads", ss.str());

    // Result
    return desc;
  }

protected:
  friend class cereal::access;
  pooling_layer() : pooling_layer(nullptr, 1, 1, 1, 1, pooling_mode::MAX) {}

  void setup_dims(DataReaderMetaData& dr_metadata) override
  {
    data_type_layer<TensorDataType>::setup_dims(dr_metadata);
    const auto& input_dims = this->get_input_dims();
    auto output_dims = input_dims;
    for (size_t i = 0; i < output_dims.size() - 1; ++i) {
      const int effective_dim =
        (input_dims[i + 1] + 2 * m_pads[i] - m_pool_dims[i] + 1);
      output_dims[i + 1] = (effective_dim + m_strides[i] - 1) / m_strides[i];
    }
    this->set_output_dims(output_dims);
  }

  /// Initialize GPU objects
  void setup_gpu() override
  {
    data_type_layer<TensorDataType>::setup_gpu();
#ifndef LBANN_HAS_DNN_LIB
    LBANN_ERROR("DNN library not detected");
#else

    // Set pooling descriptor
    m_pooling_dnn_desc.set(m_pool_mode,
                           dnn_lib::DNN_PROPAGATE_NAN,
                           m_pool_dims.size(),
                           m_pool_dims.data(),
                           m_pads.data(),
                           m_strides.data());

#endif // #ifndef LBANN_HAS_DNN_LIB
  }

  void fp_compute() override
  {
    if (this->using_gpus()) {
#ifdef LBANN_HAS_DISTCONV
      if (this->distconv_enabled()) {
        const auto& mode =
          this->m_model->get_execution_context().get_execution_mode();
        get_distconv_adapter().fp_compute(mode == execution_mode::training);
        return;
      }
#endif // LBANN_HAS_DISTCONV
      fp_compute_dnn();
    }
    else {
      fp_compute_im2col();
    }
  }

  void bp_compute() override
  {
    if (this->using_gpus()) {
#ifdef LBANN_HAS_DISTCONV
      if (this->distconv_enabled()) {
        get_distconv_adapter().bp_compute();
        return;
      }
#endif // LBANN_HAS_DISTCONV
      bp_compute_dnn();
    }
    else {
      bp_compute_im2col();
    }
  }

private:
  /// Pooling forward propagation with DNN library
  void fp_compute_dnn()
  {
#ifndef LBANN_HAS_DNN_LIB
    LBANN_ERROR("DNN library not detected");
#else
    // Initialize GPU workspace
    El::Matrix<TensorDataType, El::Device::GPU> workspace;
    size_t workspace_size =
      dnn_lib::get_pooling_ws_size(m_pooling_dnn_desc,
                                   m_tensors_dnn_desc.get_activations());
    workspace.Resize(workspace_size / sizeof(TensorDataType), 1);

    using ScalingType = dnn_lib::ScalingParamType<TensorDataType>;
    const auto& local_input = this->get_local_prev_activations();
    auto& local_output = this->get_local_activations();
    if (local_input.Height() > 0 && local_input.Width() > 0) {
      const auto zero = El::TypeTraits<ScalingType>::Zero();
      const auto one = El::TypeTraits<ScalingType>::One();
      dnn_lib::pooling_forward(m_pooling_dnn_desc,
                               one,
                               m_tensors_dnn_desc.get_prev_activations(),
                               local_input,
                               zero,
                               m_tensors_dnn_desc.get_activations(),
                               local_output,
                               workspace);
    }
#endif // #ifndef LBANN_HAS_DNN_LIB
  }

  /// Pooling backward propagation with DNN library
  void bp_compute_dnn()
  {
#ifndef LBANN_HAS_DNN_LIB
    LBANN_ERROR("DNN library not detected");
#else
    // Initialize GPU workspace
    El::Matrix<TensorDataType, El::Device::GPU> workspace;
    size_t workspace_size =
      dnn_lib::get_pooling_ws_size(m_pooling_dnn_desc,
                                   m_tensors_dnn_desc.get_activations());
    workspace.Resize(workspace_size / sizeof(TensorDataType), 1);

    using ScalingType = dnn_lib::ScalingParamType<TensorDataType>;
    const auto& local_input = this->get_local_prev_activations();
    const auto& local_output = this->get_local_activations();
    const auto& local_gradient_wrt_output =
      this->get_local_prev_error_signals();
    auto& local_gradient_wrt_input = this->get_local_error_signals();
    if (local_input.Height() > 0 && local_input.Width() > 0) {

      // Useful constants
      const auto one = El::TypeTraits<ScalingType>::One();
      const auto zero = El::TypeTraits<ScalingType>::Zero();

      // Perform backprop on GPU
      dnn_lib::pooling_backward(m_pooling_dnn_desc,
                                one,
                                m_tensors_dnn_desc.get_activations(),
                                local_output,
                                m_tensors_dnn_desc.get_prev_error_signals(),
                                local_gradient_wrt_output,
                                m_tensors_dnn_desc.get_prev_activations(),
                                local_input,
                                zero,
                                m_tensors_dnn_desc.get_error_signals(),
                                local_gradient_wrt_input,
                                workspace);
    }
#endif // #ifndef LBANN_HAS_DNN_LIB
  }

  /// Pooling forward propagation with im2col
  void fp_compute_im2col()
  {
    if (m_pool_mode != pooling_mode::MAX &&
        m_pool_mode != pooling_mode::MAX_DETERMINISTIC &&
        m_pool_mode != pooling_mode::AVERAGE_COUNT_INCLUDE_PADDING) {
      LBANN_ERROR("CPU pooling layer only supports max and average pooling");
    }

    // Local matrices
    const auto& local_input = this->get_local_prev_activations();
    auto& local_output = this->get_local_activations();

    // Pool parameters
    const int local_width = local_input.Width();
    const auto& input_dims = this->get_input_dims();
    const int num_channels = input_dims[0];
    const int num_per_output_channel = this->get_output_size() / num_channels;

    // Initialize max pool indices if needed
    if (m_pool_mode == pooling_mode::MAX ||
        m_pool_mode == pooling_mode::MAX_DETERMINISTIC) {
      m_max_pool_indices.assign(this->get_output_size() * local_width, 0);
    }

    // Initialize matrices
    El::Matrix<TensorDataType, Dev> im2col_mat(m_pool_size * num_channels,
                                               num_per_output_channel);
    El::Matrix<TensorDataType, Dev> input_mat;

    // Iterate through data samples
    for (int sample = 0; sample < local_width; ++sample) {

      // Construct im2col matrix from input
      El::LockedView(input_mat, local_input, El::ALL, El::IR(sample));
      im2col<TensorDataType>(input_mat,
                             im2col_mat,
                             num_channels,
                             input_dims.size() - 1,
                             &input_dims[1],
                             m_pads.data(),
                             m_pool_dims.data(),
                             m_strides.data());

      if (m_pool_mode == pooling_mode::MAX ||
          m_pool_mode == pooling_mode::MAX_DETERMINISTIC) {
        // Apply max pooling
        TensorDataType* output_buffer = local_output.Buffer(0, sample);
        int* indices_buffer =
          &m_max_pool_indices[sample * this->get_output_size()];
        LBANN_OMP_PARALLEL_FOR
        for (int channel = 0; channel < num_channels; ++channel) {
          for (int j = 0; j < num_per_output_channel; ++j) {
            TensorDataType* im2col_buffer =
              im2col_mat.Buffer(channel * m_pool_size, j);
            TensorDataType max_entry = im2col_buffer[0];
            int max_index = 0;
            for (int i = 1; i < m_pool_size; ++i) {
              const TensorDataType current_entry = im2col_buffer[i];
              if (current_entry > max_entry) {
                max_entry = current_entry;
                max_index = i;
              }
            }
            const int output_index = j + channel * num_per_output_channel;
            output_buffer[output_index] = max_entry;
            indices_buffer[output_index] = max_index;
          }
        }
      }

      if (m_pool_mode == pooling_mode::AVERAGE_COUNT_INCLUDE_PADDING) {
        // Apply average pooling
        TensorDataType* output_buffer = local_output.Buffer(0, sample);
        LBANN_OMP_PARALLEL_FOR
        for (int channel = 0; channel < num_channels; ++channel) {
          for (int j = 0; j < num_per_output_channel; ++j) {
            const TensorDataType* im2col_buffer =
              im2col_mat.LockedBuffer(channel * m_pool_size, j);
            TensorDataType output_entry =
              El::TypeTraits<TensorDataType>::Zero();
            for (int i = 0; i < m_pool_size; ++i) {
              output_entry += im2col_buffer[i];
            }
            output_entry /= m_pool_size;
            const int output_index = j + channel * num_per_output_channel;
            output_buffer[output_index] = output_entry;
          }
        }
      }
    }
  }

  /// Pooling forward propagation with im2col
  void bp_compute_im2col()
  {
    using CPUMatType = El::Matrix<TensorDataType, El::Device::CPU>;
    if (m_pool_mode != pooling_mode::MAX &&
        m_pool_mode != pooling_mode::MAX_DETERMINISTIC &&
        m_pool_mode != pooling_mode::AVERAGE_COUNT_INCLUDE_PADDING) {
      LBANN_ERROR("CPU pooling layer only supports max and average pooling");
    }

    // Local matrices
    const auto& local_gradient_wrt_output =
      this->get_local_prev_error_signals();
    auto& local_gradient_wrt_input = this->get_local_error_signals();

    // Pool parameters
    const int local_width = local_gradient_wrt_output.Width();
    const auto& input_dims = this->get_input_dims();
    const int num_channels = input_dims[0];
    const int num_per_input_channel = this->get_output_size() / num_channels;

    // Initialize matrices
    CPUMatType im2col_mat(m_pool_size * num_channels, num_per_input_channel);
    CPUMatType gradient_wrt_input_col;

    // Iterate through data samples
    for (int sample = 0; sample < local_width; ++sample) {

      // Compute gradient w.r.t. im2col matrix for max pooling
      if (m_pool_mode == pooling_mode::MAX ||
          m_pool_mode == pooling_mode::MAX_DETERMINISTIC) {

        // Clear im2col matrix
        El::Zero(im2col_mat);

        // Copy previous error signal to im2col matrix entries
        // corresponding to max
        const TensorDataType* gradient_wrt_output_buffer =
          local_gradient_wrt_output.LockedBuffer(0, sample);
        const int* indices_buffer =
          &m_max_pool_indices[sample * this->get_output_size()];
        LBANN_OMP_PARALLEL_FOR
        for (int channel = 0; channel < num_channels; ++channel) {
          for (int j = 0; j < num_per_input_channel; ++j) {
            const int input_index = j + channel * num_per_input_channel;
            const int max_index = indices_buffer[input_index];
            TensorDataType* im2col_buffer =
              im2col_mat.Buffer(channel * m_pool_size, j);
            im2col_buffer[max_index] = gradient_wrt_output_buffer[input_index];
          }
        }
      }

      // Compute gradient w.r.t. im2col matrix for average pooling
      if (m_pool_mode == pooling_mode::AVERAGE_COUNT_INCLUDE_PADDING) {
        const TensorDataType* gradient_wrt_output_buffer =
          local_gradient_wrt_output.LockedBuffer(0, sample);
        LBANN_OMP_PARALLEL_FOR
        for (int channel = 0; channel < num_channels; ++channel) {
          for (int j = 0; j < num_per_input_channel; ++j) {
            TensorDataType* im2col_buffer =
              im2col_mat.Buffer(channel * m_pool_size, j);
            const int input_index = j + channel * num_per_input_channel;
            const TensorDataType output_entry =
              gradient_wrt_output_buffer[input_index] /
              El::To<TensorDataType>(m_pool_size);
            for (int i = 0; i < m_pool_size; ++i) {
              im2col_buffer[i] = output_entry;
            }
          }
        }
      }

      // Compute error signal (i.e. gradient w.r.t. input)
      El::View(gradient_wrt_input_col,
               local_gradient_wrt_input,
               El::ALL,
               El::IR(sample));
      col2im<TensorDataType>(im2col_mat,
                             gradient_wrt_input_col,
                             num_channels,
                             input_dims.size() - 1,
                             &input_dims[1],
                             m_pads.data(),
                             m_pool_dims.data(),
                             m_strides.data());
    }
  }

#ifdef LBANN_HAS_DISTCONV
  friend class pooling_distconv_adapter<TensorDataType, T_layout, Dev>;

protected:
  bool is_distconv_supported() const override;
  void setup_distconv_adapter(const DataReaderMetaData& dr_metadata) override
  {
    this->get_distconv_adapter_ptr() =
      std::make_unique<pooling_distconv_adapter<TensorDataType, T_layout, Dev>>(
        *this);
  }
  pooling_distconv_adapter<TensorDataType, T_layout, Dev>&
  get_distconv_adapter() override;
  const pooling_distconv_adapter<TensorDataType, T_layout, Dev>&
  get_distconv_adapter() const override;
#endif // LBANN_HAS_DISTCONV
};

#ifdef LBANN_HAS_ONNX
template <typename T, data_layout L, El::Device D>
void pooling_layer<T, L, D>::fill_onnx_node(onnx::GraphProto& graph) const
{
  auto* pool = graph.add_node();

  // Get the attributes setup first
  {
    auto* kernel_shape = pool->add_attribute();
    kernel_shape->set_name("kernel_shape");
    kernel_shape->set_type(onnx::AttributeProto::INTS);
    for (auto const& k : this->m_pool_dims)
      kernel_shape->add_ints(k);
  }
  if (!this->m_strides.empty()) {
    auto* strides = pool->add_attribute();
    strides->set_name("strides");
    strides->set_type(onnx::AttributeProto::INTS);
    for (auto const& s : this->m_strides)
      strides->add_ints(s);
  }
  if (!this->m_pads.empty()) {
    auto* pads = pool->add_attribute();
    pads->set_name("pads");
    pads->set_type(onnx::AttributeProto::INTS);
    for (auto const& p : this->m_pads) {
      pads->add_ints(p);
      pads->add_ints(p);
    }
  }
  // FIXME: This is missing "dilations". However, they're only a valid
  // attribute for MaxPool, not AveragePool.

  for (auto const* parent : this->get_parent_layers()) {
    size_t idx = parent->find_child_layer_index(*this);
    pool->add_input(parent->get_name() + "_" + std::to_string(idx));
  }
  for (size_t ii = 0; ii < this->num_weights(); ii++)
    pool->add_input(this->get_weights(ii).get_name());
  for (auto const* child : this->get_child_layers()) {
    size_t idx = this->find_child_layer_index(*child);
    pool->add_output(this->get_name() + "_" + std::to_string(idx));
  }
  pool->set_name(this->get_name());

  switch (m_pool_mode) {
  case pooling_mode::MAX:
    pool->set_op_type("MaxPool");
    break;
  case pooling_mode::MAX_DETERMINISTIC:
    pool->set_op_type("MaxPool");
    break;
  case pooling_mode::AVERAGE_COUNT_INCLUDE_PADDING:
    pool->set_op_type("AveragePool");
    break;
  case pooling_mode::AVERAGE_COUNT_EXCLUDE_PADDING:
    pool->set_op_type("AveragePool");
    break;
  default:
    LBANN_ERROR("pooling_layer: no ONNX implementation for pooling mode");
  }

  pool->set_domain("");
  pool->set_doc_string(this->get_type());
}
#endif

#ifdef LBANN_HAS_DISTCONV
template <typename TensorDataType, data_layout T_layout, El::Device Dev>
pooling_distconv_adapter<TensorDataType, T_layout, Dev>&
pooling_layer<TensorDataType, T_layout, Dev>::get_distconv_adapter()
{
  return const_cast<pooling_distconv_adapter<TensorDataType, T_layout, Dev>&>(
    static_cast<const pooling_layer<TensorDataType, T_layout, Dev>&>(*this)
      .get_distconv_adapter());
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
const pooling_distconv_adapter<TensorDataType, T_layout, Dev>&
pooling_layer<TensorDataType, T_layout, Dev>::get_distconv_adapter() const
{
  return dynamic_cast<
    const pooling_distconv_adapter<TensorDataType, T_layout, Dev>&>(
    data_type_layer<TensorDataType>::get_distconv_adapter());
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
bool pooling_layer<TensorDataType, T_layout, Dev>::is_distconv_supported() const
{
  if (Dev != El::Device::GPU || T_layout != data_layout::DATA_PARALLEL) {
    return false;
  }

  bool cond = true;
  for (int i = 0; i < dc::get_num_spatial_dims(*this); i++) {
    cond &= (m_pool_dims[i] % 2 != 0) || (m_pool_dims[i] == m_strides[i]);
  }
  if (!cond) {
    dc::MPIPrintStreamDebug() << "pooling: unsupported due to window shape: "
                              << dc::util::join_xd_array(m_pool_dims);
    return false;
  }

  for (int i = 0; i < dc::get_num_spatial_dims(*this); i++) {
    bool odd = m_pool_dims[i] % 2;
    if (odd) {
      int stencil = (m_pool_dims[i] - 1) / 2;
      if (!(m_pads[i] == 0 || m_pads[i] == stencil)) {
        dc::MPIPrintStreamDebug()
          << "pooling: unsupported due to padding: " << m_pads[i];
        return false;
      }
      if (!(m_strides[i] == 1 || m_strides[i] == stencil + 1)) {
        dc::MPIPrintStreamDebug() << "pooling: unsupported due to strides";
        return false;
      }
    }
    else {
      if (m_pads[i] != 0)
        return false;
      if (m_pool_dims[i] != m_strides[i])
        return false;
    }
  }

  return true;
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void pooling_distconv_adapter<TensorDataType, T_layout, Dev>::
  setup_distributions(tensor_overlap_constraints& constraints)
{
  data_type_distconv_adapter<TensorDataType>::setup_distributions(constraints);
  const auto& l =
    dynamic_cast<const pooling_layer<TensorDataType, T_layout, Dev>&>(
      this->layer());
  dc::IntVector overlap(dc::get_num_dims(l), 0);
  const auto& ps = l.get_parallel_strategy();
  auto pool_dims = l.m_pool_dims;
  std::reverse(pool_dims.begin(), pool_dims.end());
  for (int i = 0; i < dc::get_num_spatial_dims(l); i++) {
    int splits = 0;
    switch (i) {
    case 0:
      splits = ps.width_splits;
      break;
    case 1:
      splits = ps.height_splits;
      break;
    case 2:
      splits = ps.depth_splits;
      break;
    }
    if (splits == 1)
      continue;
    int ov = 0;
    if (pool_dims[i] % 2) {
      ov = (pool_dims[i] - 1) / 2;
    }
    else {
      // no halo dependency is assumed for now
      ov = 0;
    }
    overlap[i] = ov;
  }
  auto& prev_activations_dist = this->get_prev_activations_dist();
  auto& activations_dist = this->get_activations_dist();
  auto& error_signals_dist = this->get_error_signals_dist();
  auto& prev_error_signals_dist = this->get_prev_error_signals_dist();
  prev_activations_dist.set_overlap(overlap);
  constraints.mark_updated(prev_activations_dist);
  constraints.mark_invariant(prev_activations_dist);
  // cudnnPoolingBackward requires activations and
  // prev_error_signals must have the same stride
  constraints.mark_equivalent(activations_dist, prev_error_signals_dist);
  // cudnnPoolingBackward requires prev_activations and
  // error_signals must have the same stride
  constraints.mark_equivalent(error_signals_dist, prev_activations_dist);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
dc::Shape pooling_distconv_adapter<TensorDataType, Layout, Device>::
  get_activations_local_shape(int index) const
{
  assert_eq(index, 0);
  const auto& layer =
    dynamic_cast<const pooling_layer<TensorDataType, Layout, Device>&>(
      this->layer());
  auto filter_dims = layer.m_pool_dims;
  std::reverse(std::begin(filter_dims), std::end(filter_dims));
  auto strides = layer.m_strides;
  std::reverse(std::begin(strides), std::end(strides));
  const std::vector<int> dilations(dc::get_num_spatial_dims(layer), 1);
  bool use_padding = layer.m_pads[0] != 0;
  auto output_spatial_local_shape =
    ::distconv::get_pooling_output_local_tensor_shape(
      this->get_prev_activations(),
      filter_dims,
      strides,
      use_padding,
      dilations);
  return output_spatial_local_shape;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void pooling_distconv_adapter<TensorDataType, Layout, Device>::setup_layer(
  size_t workspace_capacity)
{
  auto& l =
    dynamic_cast<pooling_layer<TensorDataType, Layout, Device>&>(this->layer());

  // Init the dc::Pooling layer
  m_pooling = std::make_unique<dc::Pooling<TensorDataType>>(
    dc::get_backend(),
    dc::get_num_dims(l),
    dc::get_halo_exchange_method());

  std::string mode;
  switch (l.m_pool_mode) {
  case pooling_mode::MAX:
    mode = "MAX";
    break;
  case pooling_mode::MAX_DETERMINISTIC:
    mode = "MAX";
    break;
  case pooling_mode::AVERAGE_COUNT_INCLUDE_PADDING:
    mode = "AVERAGE";
    break;
  case pooling_mode::AVERAGE_COUNT_EXCLUDE_PADDING:
    mode = "AVERAGE_NO_PAD";
    break;
  default:
    LBANN_ERROR("pooling_layer: no DISTCONV implementation for pooling mode");
  }

  std::vector<int> pool_dims = l.m_pool_dims;
  std::reverse(pool_dims.begin(), pool_dims.end());
  std::vector<int> pads = l.m_pads;
  std::reverse(pads.begin(), pads.end());
  std::vector<int> strides = l.m_strides;
  std::reverse(strides.begin(), strides.end());

  m_pooling->setup(this->get_prev_activations(),
                   this->get_activations(),
                   this->get_error_signals(),
                   this->get_prev_error_signals(),
                   pool_dims,
                   pads,
                   strides,
                   mode);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void pooling_distconv_adapter<TensorDataType, Layout, Device>::fp_compute(
  bool const training)
{
  m_pooling->forward(El::To<TensorDataType>(1),
                     this->get_prev_activations(),
                     El::To<TensorDataType>(0),
                     this->get_activations(),
                     training);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void pooling_distconv_adapter<TensorDataType, Layout, Device>::bp_compute()
{
  m_pooling->backward(El::To<TensorDataType>(1),
                      this->get_activations(),
                      this->get_prev_error_signals(),
                      this->get_prev_activations(),
                      El::To<TensorDataType>(0),
                      this->get_error_signals());
}
#endif // LBANN_HAS_DISTCONV

#ifndef LBANN_POOLING_LAYER_INSTANTIATE
#define PROTO_DEVICE(T, Device)                                                \
  extern template class pooling_layer<T, data_layout::DATA_PARALLEL, Device>

#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE
#endif // LBANN_POOLING_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYER_POOLING_HPP_INCLUDED
