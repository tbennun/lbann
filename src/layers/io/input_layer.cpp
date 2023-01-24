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

#include "lbann/layers/layer.hpp"
#include <type_traits>
#define LBANN_INPUT_LAYER_INSTANTIATE
#include "lbann/layers/io/input_layer.hpp"

#include "lbann/callbacks/imcomm.hpp"
#include "lbann/execution_algorithms/execution_context.hpp"
#include "lbann/execution_algorithms/sgd_execution_context.hpp"
#include "lbann/utils/profiling.hpp"
#include "lbann/utils/protobuf.hpp"
#include "lbann/utils/serialize.hpp"

template <typename T, lbann::data_layout L, El::Device D>
std::unique_ptr<lbann::Layer>
lbann::build_input_layer_from_pbuf(lbann_comm* comm,
                                   lbann_data::Layer const& proto_layer)
{
  const auto& params = proto_layer.input();
  const auto& data_field = params.data_field();
  if constexpr (L != data_layout::DATA_PARALLEL) {
    LBANN_ERROR("input layer is only supported with "
                "a data-parallel layout");
  }
  if constexpr (std::is_same_v<T, DataType> &&
                (L == data_layout::DATA_PARALLEL)) {
    return std::make_unique<
      input_layer<DataType, data_layout::DATA_PARALLEL, D>>(comm, data_field);
  }
  else {
    LBANN_ERROR("Input layers are only valid with "
                "TensorDataType == DataType and Layout == DATA_PARALLEL");
  }
  return nullptr;
}

namespace lbann {

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void input_layer<TensorDataType, T_layout, Dev>::setup_dims(
  DataReaderMetaData& dr_metadata)
{
  data_type_layer<TensorDataType>::setup_dims(dr_metadata);
  for (int i = 0; i < this->get_num_children(); ++i) {
    this->set_output_dims(get_data_dims(dr_metadata, i), i);
  }
  if (m_data_field == "") {
    LBANN_ERROR("Failed to setup input layer with empty data field");
  }
  get_trainer().get_data_coordinator().register_active_data_field(m_data_field);
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void input_layer<TensorDataType, T_layout, Dev>::setup_data(
  size_t max_mini_batch_size)
{
  data_type_layer<TensorDataType>::setup_data(max_mini_batch_size);

  // Resize output to maximum mini-batch size
  for (int i = 0; i < this->get_num_children(); ++i) {
    auto& output = this->get_activations(i);
    output.Resize(output.Height(), max_mini_batch_size);
  }
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void input_layer<TensorDataType, T_layout, Dev>::fp_setup_outputs(
  El::Int mini_batch_size)
{
  /// During model setup there is no valid execution context, but
  /// during execution there is a context
  if (this->m_model->has_valid_execution_context()) {
    auto& c = dynamic_cast<SGDExecutionContext&>(
      this->m_model->get_execution_context());
    auto mode = c.get_execution_mode();
    auto effective_mini_batch_size = mini_batch_size;
    if (!(mode == execution_mode::inference)) {
      data_coordinator& dc = get_trainer().get_data_coordinator();
      // Determine model mini-batch size and effective mini-batch size
      // Note: If inter-model communication is activated, the effective
      // mini-batch is equal to the global mini-batch size.
      /// @todo This functionality should probably be moved elsewhere
      mini_batch_size = dc.get_current_mini_batch_size(mode);

      effective_mini_batch_size = mini_batch_size;
      for (auto&& cb : this->m_model->get_callbacks()) {
        if (dynamic_cast<callback::imcomm*>(cb) != nullptr) {
          effective_mini_batch_size =
            dc.get_current_global_mini_batch_size(mode);
          break;
        }
      }
    }
    // Set mini-batch size in model
    c.set_current_mini_batch_size(mini_batch_size);
    c.set_effective_mini_batch_size(effective_mini_batch_size);
  }

  // Activation matrices are initalized in setup_data and further
  // managed in the distribute_from_local_matrix function of the
  // data_coordinator.
  // However, on the first pass through the execution algorithm it
  // is necessary to setup the size of the matrix.
  for (int i = 0; i < this->get_num_children(); ++i) {
    auto& output = this->get_activations(i);
    if (!output.Viewing()) {
      output.Empty(false);
      output.Resize(this->get_output_size(i), mini_batch_size);
    }
  }
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void input_layer<TensorDataType, T_layout, Dev>::fp_compute()
{
  if (!this->m_samples_loaded) {
    execution_mode const mode =
      this->m_model->get_execution_context().get_execution_mode();
    buffered_data_coordinator<TensorDataType>& dc =
      static_cast<buffered_data_coordinator<TensorDataType>&>(
        get_trainer().get_data_coordinator());

    dc.distribute_from_local_matrix(mode,
                                    m_data_field,
                                    this->get_activations(0));

#ifdef LBANN_HAS_DISTCONV
    if (this->distconv_enabled()) {
      get_distconv_adapter().fp_compute();
    }
#endif // LBANN_HAS_DISTCONV
  }
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void input_layer<TensorDataType, T_layout, Dev>::set_samples(
  const El::AbstractDistMatrix<TensorDataType>& samples)
{
  El::Copy(samples, this->get_activations(0));
  this->m_samples_loaded = true;
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
std::vector<int> input_layer<TensorDataType, T_layout, Dev>::get_data_dims(
  DataReaderMetaData& dr_metadata,
  int child_index) const
{
  if (child_index != 0) {
    LBANN_ERROR("get_data_dims: Invalid child index");
  }
  if (m_data_field == INPUT_DATA_TYPE_SAMPLES) {
    return dr_metadata.data_dims[data_reader_target_mode::INPUT];
  }
  else if (m_data_field == INPUT_DATA_TYPE_LABELS) {
    return dr_metadata.data_dims[data_reader_target_mode::CLASSIFICATION];
  }
  else if (m_data_field == INPUT_DATA_TYPE_RESPONSES) {
    return dr_metadata.data_dims[data_reader_target_mode::REGRESSION];
  }
  else {
    LBANN_ERROR("Unknown data_field_type value provided: " + m_data_field);
  }
  return std::vector<int>(1, 0);
}

#ifdef LBANN_HAS_ONNX
template <typename T, data_layout L, El::Device D>
void input_layer<T, L, D>::fill_onnx_node(onnx::GraphProto& graph) const
{
  auto child_layers = this->get_child_layers();
  for (auto const* child : this->get_child_layers()) {
    auto idx = this->find_child_layer_index(*child);
    auto* input = graph.add_input();
    input->set_name(this->get_name() + "_" + std::to_string(idx));
    auto* input_type = input->mutable_type();
    input_type->mutable_tensor_type()->set_elem_type(onnx::TensorProto::FLOAT);

    auto* dims = input_type->mutable_tensor_type()->mutable_shape()->add_dim();
    dims->set_dim_param("batch");
    for (auto const& dim : this->get_output_dims(idx)) {
      dims = input_type->mutable_tensor_type()->mutable_shape()->add_dim();
      dims->set_dim_value(dim);
    }
    input->set_doc_string("Input layer info");
  }
}
#endif // LBANN_HAS_ONNX

#ifdef LBANN_HAS_DISTCONV
template <typename TensorDataType, data_layout T_layout, El::Device Dev>
input_distconv_adapter<TensorDataType, T_layout, Dev>::input_distconv_adapter(
  Layer& layer,
  const data_field_type data_field,
  const bool shuffle_required)
  : data_type_distconv_adapter<TensorDataType>(layer),
    m_data_field(data_field),
    m_shuffle_required(shuffle_required)
{

  // Distconv currently only supports CosmoFlow data
  if (m_data_field != INPUT_DATA_TYPE_SAMPLES &&
      m_data_field != INPUT_DATA_TYPE_RESPONSES) {
    LBANN_ERROR("attempted to create distconv adapter for ",
                "input layer with unsupported data field (",
                m_data_field,
                ")");
  }

  // Input data is only processed when its consumer layer is also
  // enabled for distconv
  m_is_input_processed = (m_data_field == INPUT_DATA_TYPE_SAMPLES) ||
                         layer.get_child_layer().distconv_enabled();
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
typename input_distconv_adapter<TensorDataType, T_layout, Dev>::
  TensorHostShuffler&
  input_distconv_adapter<TensorDataType, T_layout, Dev>::get_shuffler(
    const TensorHost& src,
    const TensorHost& dst)
{
  size_t cur_mb_size = src.get_shape()[dc::get_sample_dim()];
  auto src_buf = m_shuffler_src_buf.get();
  auto dst_buf = m_shuffler_dst_buf.get();
  int shfl_idx = -1;
  if (cur_mb_size == get_trainer().get_max_mini_batch_size()) {
    shfl_idx = 0;
  }
  else {
    // The last remaining mini-batches for the train, validation, and
    // testing modes
    auto mode =
      this->layer().get_model()->get_execution_context().get_execution_mode();
    shfl_idx = 1 + static_cast<int>(mode);
  }
  assert_always(shfl_idx >= 0 && shfl_idx < 4);
  auto& shfl = m_shufflers[shfl_idx];
  if (shfl == nullptr) {
    shfl = std::make_unique<TensorHostShuffler>(src, dst, src_buf, dst_buf);
  }
  return *shfl;
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void input_distconv_adapter<TensorDataType, T_layout, Dev>::setup_fp_tensors()
{
  const auto sample_dist = dc::get_hydrogen_data_parallel_distribution(
    dc::get_num_dims(this->layer()));

  if (m_is_input_processed) {

    const auto shape = this->get_activations_shape(0);
    auto local_shape = shape;
    if (m_shuffle_required) {
      local_shape[dc::get_sample_dim()] = 0;
    }
    else {
      local_shape = 0;
    }

    // Use the same MPI communicator for both IO buffers. This seems
    // to work around MPI errors likely caused with the alltoallv for
    // shuffling.
    const dc::LocaleMPI loc(dc::get_mpi_comm(), false);

    auto dist = this->get_activations_dist();
    if (m_data_field == INPUT_DATA_TYPE_RESPONSES) {
      // assumes no halo for the ground-truth data
      dist.clear_overlap();
    }
    auto dist_no_halo = dist;
    dist_no_halo.clear_overlap();

    const auto original_host_tensor_dist =
      m_shuffle_required ? sample_dist : dist_no_halo;
    // Create a view to the host LBANN matrix
    m_original_host_tensor =
      std::make_unique<TensorHost>(shape,
                                   loc,
                                   original_host_tensor_dist,
                                   local_shape);

    // When shuffled, host tensor will have the same distribution as
    // the final output; otherwise, it is just a view to the host
    // LBANN matrix, so no overlap.
    auto host_tensor_dist = m_shuffle_required ? dist : dist_no_halo;
    m_host_tensor = std::make_unique<TensorHost>(shape, loc, host_tensor_dist);

    if (m_shuffle_required) {
      // TODO: This is a temporary hack. Should use
      // CUDAHostPooledAllocator, but the shuffler is
      // only specialized for BaseAllocator.
      size_t buf_size =
        m_host_tensor->get_local_real_size() * sizeof(TensorDataType);
      TensorDataType* buf = nullptr;
#if H2_HAS_CUDA
      CHECK_CUDA(cudaMallocHost(&buf, buf_size));
#elif H2_HAS_ROCM
      CHECK_ROCM(hipHostMalloc(&buf, buf_size));
#endif
      // Note buf should be deallocated.
      dc::tensor::View(*m_host_tensor, buf);
      setup_shuffler_buffers(*m_original_host_tensor, *m_host_tensor);
    }
  }

  this->setup_activations();
  this->setup_original_activations();
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
std::unique_ptr<
  typename input_distconv_adapter<TensorDataType, T_layout, Dev>::TensorDevType>
input_distconv_adapter<TensorDataType, T_layout, Dev>::setup_activations_i(
  int index) const
{
  if (!m_is_input_processed)
    return nullptr;
  if (m_data_field == INPUT_DATA_TYPE_SAMPLES) {
    return data_type_distconv_adapter<TensorDataType>::setup_activations_i(
      index);
  }
  else if (m_data_field == INPUT_DATA_TYPE_RESPONSES) {
    // Note: the default setup_activations_i can't be used because
    // the distribution might need to be changed to remove
    // overlap. This can be fixed by making each tensor hav a
    // different distribution.
    const dc::LocaleMPI loc(dc::get_mpi_comm(), false);
    auto dist = this->get_activations_dist();
    dist.clear_overlap();
    const auto shape = get_activations_shape(index);
    const auto local_shape = get_activations_local_shape(index);
    auto t = std::make_unique<TensorDevType>(shape, loc, dist, local_shape);
    assert0(t->allocate());
    t->zero(default_hydrogen_stream());
    return t;
  }
  else {
    LBANN_ERROR("unsupported data field (", m_data_field, ")");
  }
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
dc::Shape input_distconv_adapter<TensorDataType, T_layout, Dev>::
  get_activations_local_shape(int index) const
{
  // No enforced local shape as the activations tensor is always
  // copied from the El matrix.
  return dc::Shape(dc::get_num_dims(this->layer()), 0);
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
dc::Shape
input_distconv_adapter<TensorDataType, T_layout, Dev>::get_activations_shape(
  int index) const
{
  if (m_data_field == INPUT_DATA_TYPE_SAMPLES) {
    return data_type_distconv_adapter<TensorDataType>::get_activations_shape(
      index);
  }
  else if (m_data_field == INPUT_DATA_TYPE_RESPONSES) {
    // TODO: This is a temporary hack. The label tensor shape should
    // be set based on the shape set by the data reader, but the data
    // reader does not provide it. Using the shape shape as the data
    // tensor works fine for the U-Net model.
    auto shape =
      data_type_distconv_adapter<TensorDataType>::get_activations_shape(
        0); /// @todo Should this be getting shape corresponding to
            /// INPUT_DATA_TYPE_SAMPLES?
    auto label_size =
      data_type_distconv_adapter<TensorDataType>::get_activations_shape(0)
        .reduce_prod();
    const std::string env = std::getenv("DISTCONV_LABEL_NUM_CHANNELS");
    auto num_channels =
      env != "" ? std::stoi(env) : label_size / shape.reduce_prod();
    shape[-2] = num_channels;
    return shape;
  }
  else {
    LBANN_ERROR("unsupported data field (", m_data_field, ")");
  }
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void input_distconv_adapter<TensorDataType, T_layout, Dev>::
  setup_shuffler_buffers(const TensorHost& src, const TensorHost& dst)
{
  auto shuffler_src_size = TensorHostShuffler::get_buf_size(src);
  if (m_shuffler_src_buf_size < shuffler_src_size) {
    m_shuffler_src_buf_size = shuffler_src_size;
    m_shuffler_src_buf =
      std::unique_ptr<TensorDataType>(static_cast<TensorDataType*>(
        dc::util::aligned_malloc(m_shuffler_src_buf_size)));
  }
  auto shuffler_dst_size = TensorHostShuffler::get_buf_size(dst);
  if (m_shuffler_dst_buf_size < shuffler_dst_size) {
    m_shuffler_dst_buf_size = shuffler_dst_size;
    m_shuffler_dst_buf =
      std::unique_ptr<TensorDataType>(static_cast<TensorDataType*>(
        dc::util::aligned_malloc(m_shuffler_dst_buf_size)));
  }
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
bool input_distconv_adapter<TensorDataType, T_layout, Dev>::child_copy_required(
  size_t output_index) const
{
  // Not required when label is not handled.
  if (m_data_field == INPUT_DATA_TYPE_RESPONSES && !m_is_input_processed) {
    return false;
  }
  else {
    return data_type_distconv_adapter<TensorDataType>::child_copy_required(
      output_index);
  }
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
bool input_distconv_adapter<TensorDataType, T_layout, Dev>::
  child_shuffle_required(size_t output_index) const
{
  // Not required when label is not handled.
  if (m_data_field == INPUT_DATA_TYPE_RESPONSES && !m_is_input_processed) {
    return false;
  }
  else {
    return data_type_distconv_adapter<TensorDataType>::child_shuffle_required(
      output_index);
  }
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void input_distconv_adapter<TensorDataType, T_layout, Dev>::fp_compute()
{
  auto& l =
    dynamic_cast<input_layer<TensorDataType, T_layout, Dev>&>(this->layer());
  auto stream = default_hydrogen_stream();
  // Note that the mini-batch size of the data reader is not
  // actually the one for the current mini-batch as the mini-batch
  // index is already updated by fp_compute.
  const int mb_size =
    static_cast<SGDExecutionContext&>(l.get_model()->get_execution_context())
      .get_current_mini_batch_size();

  if (m_is_input_processed) {

    // TODO: This is diabled as it raises an error when the HDF5 data
    // reader with hyperslab labels is used. Remove this assertion or
    // reshape the actiavtion tensor (data_field = RESPONSES).
    // assert_eq(mb_size * dc::get_number_of_io_partitions(),
    //           l.get_activations().Width());

    auto& original_tensor = *m_original_host_tensor;
    auto& host_tensor = *m_host_tensor;
    auto& device_tensor = this->get_activations();

    // Adjust the mini-batch size
    original_tensor.set_outermost_dimension(mb_size);
    host_tensor.set_outermost_dimension(mb_size);
    device_tensor.set_outermost_dimension(mb_size);

    // Setup view
    assert0(
      dc::tensor::View(original_tensor, l.get_activations().LockedBuffer()));

    // Shuffle if necessary
    if (m_shuffle_required) {
      get_shuffler(original_tensor, host_tensor)
        .shuffle_forward(original_tensor.get_const_base_ptr(),
                         host_tensor.get_base_ptr());
    }
    else {
      // The input buffer is already partitioned
      assert0(
        dc::tensor::View(host_tensor, original_tensor.get_const_buffer()));
    }

    // After this, there is no inter-process communication, so it's
    // safe to exit if the local tensor is empty.
    if (host_tensor.get_local_size() > 0) {
      prof_region_begin("copy-to-device", prof_colors[1], false);
      assert0(dc::tensor::Copy(device_tensor, host_tensor, stream));
      prof_region_end("copy-to-device", false);
    }
  }
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
const input_distconv_adapter<TensorDataType, T_layout, Dev>&
input_layer<TensorDataType, T_layout, Dev>::get_distconv_adapter() const
{
  return dynamic_cast<
    const input_distconv_adapter<TensorDataType, T_layout, Dev>&>(
    data_type_layer<TensorDataType>::get_distconv_adapter());
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
input_distconv_adapter<TensorDataType, T_layout, Dev>&
input_layer<TensorDataType, T_layout, Dev>::get_distconv_adapter()
{
  return const_cast<input_distconv_adapter<TensorDataType, T_layout, Dev>&>(
    static_cast<const input_layer<TensorDataType, T_layout, Dev>&>(*this)
      .get_distconv_adapter());
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
bool input_layer<TensorDataType, T_layout, Dev>::keep_original_outputs(
  int index) const
{
  // The original output matrices are always needed as we copy them
  // into distconv tensors.
  return true;
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
bool input_layer<TensorDataType, T_layout, Dev>::
  keep_original_gradient_wrt_outputs(int index) const
{
  // Error signals are ignored
  return false;
}

#endif // LBANN_HAS_DISTCONV

#define PROTO_DEVICE(T, Device)                                                \
  template class input_layer<T, data_layout::DATA_PARALLEL, Device>;           \
  LBANN_LAYER_BUILDER_ETI(input, T, Device)

#include "lbann/macros/instantiate_device.hpp"

} // namespace lbann
