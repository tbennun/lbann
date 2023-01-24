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

#ifndef LBANN_BUFFERED_DATA_COORDINATOR_HPP
#define LBANN_BUFFERED_DATA_COORDINATOR_HPP

#include "lbann/data_coordinator/data_coordinator.hpp"
#include "lbann/data_coordinator/io_data_buffer.hpp"

namespace lbann {

template <typename TensorDataType>
class buffered_data_coordinator : public data_coordinator
{
public:
  /** @name Public Types */
  ///@{

  /** @brief The tensor type expected in this object. */
  using AbsDistMatrixType = El::AbstractDistMatrix<TensorDataType>;

  using io_buffer_map_t = std::map<execution_mode, std::atomic<int>>;

  /** @brief The local tensor type expected for IO in this object. */
  using IODataType = DataType;

  ///@}
public:
  typedef std::map<execution_mode, std::unique_ptr<data_buffer<IODataType>>>
    data_buffer_map_t;

public:
  buffered_data_coordinator(lbann_comm* comm) : data_coordinator(comm)
  {

    // Initialize two buffers
    m_data_buffers.resize(2);
    for (size_t i = 0; i < m_data_buffers.size(); i++) {
      for (auto m : execution_mode_iterator()) {
        if (m != execution_mode::invalid) {
          m_data_buffers[i][m] =
            std::make_unique<data_buffer<IODataType>>(comm);
        }
      }
    }

    for (auto m : execution_mode_iterator()) {
      if (m != execution_mode::invalid) {
        this->m_active_buffer[m].store(-1);
      }
    }
  }

  ~buffered_data_coordinator() {}

  // Data Coordinators copy their data readers.
  buffered_data_coordinator(const buffered_data_coordinator& other)
    : data_coordinator(other)
  {
    m_data_buffers.resize(other.m_data_buffers.size());
    for (size_t i = 0; i < other.m_data_buffers.size(); i++) {
      data_buffer_map_t& buffer_map = m_data_buffers[i];
      const data_buffer_map_t& other_buffer_map = other.m_data_buffers[i];
      for (auto& b : other_buffer_map) {
        buffer_map[b.first].reset(b.second ? b.second->copy() : nullptr);
      }
    }
  }

  buffered_data_coordinator& operator=(const buffered_data_coordinator& other)
  {
    data_coordinator::operator=(other);
    m_data_buffers.clear();
    m_data_buffers.resize(other.m_data_buffers.size());
    for (size_t i = 0; i < other.m_data_buffers.size(); i++) {
      data_buffer_map_t& buffer_map = m_data_buffers[i];
      const data_buffer_map_t& other_buffer_map = other.m_data_buffers[i];
      for (auto& b : other_buffer_map) {
        buffer_map[b.first].reset(b.second ? b.second->copy() : nullptr);
      }
    }
    return *this;
  }

  /** Archive for checkpoint and restart */
  template <class Archive>
  void serialize(Archive& ar);

  /** @brief After registering the active data field, allocate storage for each
   *  data field in the context maps within the double buffer.
   */
  void register_active_data_field(data_field_type const data_field) override;

  void fp_setup_data(data_buffer<IODataType>& buffer,
                     El::Int cur_mini_batch_size);

  void fetch_data(execution_mode mode) override;

  const data_buffer_map_t& get_active_buffer_map(execution_mode mode) const;
  data_buffer_map_t& get_active_buffer_map(execution_mode mode);

  const data_buffer<IODataType>& get_active_buffer(execution_mode mode) const;
  data_buffer<IODataType>& get_active_buffer(execution_mode mode);

  const El::Matrix<El::Int>*
  get_sample_indices_per_mb(execution_mode mode) const override;
  El::Matrix<El::Int>* get_sample_indices_per_mb(execution_mode mode) override;

  /** @brief Complete any background I/O data fetch for the execution
      mode requested */
  void collect_background_data_fetch(execution_mode mode) override;

  bool epoch_complete(execution_mode mode) override;

  const data_buffer<IODataType>&
  get_data_buffer(const data_buffer_map_t& buffer_map,
                  const execution_mode mode) const;
  data_buffer<IODataType>& get_data_buffer(data_buffer_map_t& buffer_map,
                                           const execution_mode mode);

  void distribute_from_local_matrix(execution_mode mode,
                                    data_field_type data_field,
                                    AbsDistMatrixType& input_buffer);

protected:
  int fetch_to_local_matrix(data_buffer_map_t& buffer_map,
                            const execution_mode mode);

  void fetch_data_in_background(int future_active_buffer, execution_mode mode);

  int get_active_buffer_idx(execution_mode m) const
  {
    return m_active_buffer.at(m).load();
  }

  int get_active_buffer_idx(execution_mode m)
  {
    return m_active_buffer[m].load();
  }

  void increment_active_buffer_idx(execution_mode m) { m_active_buffer[m]++; }

  bool update_data_set(generic_data_reader* data_reader, execution_mode mode);

  //************************************************************************
  //
  //************************************************************************

  // save state of IO to a checkpoint
  bool save_to_checkpoint_shared(persist& p) const override;

  // reload state of IO from a checkpoint
  bool load_from_checkpoint_shared(persist& p) override;

  bool save_to_checkpoint_distributed(persist& p) const override;

  bool load_from_checkpoint_distributed(persist& p) override;

protected:
  /** @brief After a data field has been registered with the data
   *  coordinator setup its buffers. Note this can be called after
   *  each call to register_active_data_field. */
  void setup_data_fields(int max_mini_batch_size) override;

  /**
   * Map from execution context to the index of the active data buffer
   */
  io_buffer_map_t m_active_buffer;

  /** Vector of input data buffers
   *  There are two sets of buffer maps to allow for double buffered execution
   *  Within each buffer map there is a buffer for each phase of execution.
   *  Each matrix column corresponds to a flattened mini-batch sample
   *  or label or responase.
   */
  std::vector<data_buffer_map_t> m_data_buffers;
};

} // namespace lbann

#endif // LBANN_BUFFERED_DATA_COORDINATOR_HPP
