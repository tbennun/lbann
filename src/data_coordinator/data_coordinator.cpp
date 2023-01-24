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

#include "lbann/comm_impl.hpp"
#include <lbann/data_coordinator/data_coordinator.hpp>
#include <lbann/trainers/trainer.hpp>
#include <lbann/utils/distconv.hpp>
#include <lbann/utils/serialize.hpp>

namespace lbann {

template <class Archive>
void data_coordinator::serialize(Archive& ar)
{
  ar(/*CEREAL_NVP(m_io_buffer),*/
     CEREAL_NVP(m_datasets)/*,
     CEREAL_NVP(m_active_data_fields),
     CEREAL_NVP(m_data_readers),
     CEREAL_NVP(m_data_set_processed)*/);
}

void data_coordinator::setup(
  thread_pool& io_thread_pool,
  int max_mini_batch_size,
  std::map<execution_mode, generic_data_reader*> data_readers)
{
  m_io_thread_pool = &io_thread_pool;

  m_data_readers = data_readers;

  // Initialize the data sets
  for (auto m : execution_mode_iterator()) {
    if (this->m_data_readers.count(m)) {
      this->m_datasets[m].total_samples() = m_data_readers[m]->get_num_data();
    }
  }

  /// @todo BVE FIXME the list of execution modes should not include
  // ones will null data readers.  Fix this in next PR.
  // Setup data readers
  for (auto&& dr : m_data_readers) {
    if (!dr.second)
      continue;
    dr.second->setup(m_io_thread_pool->get_num_threads(), m_io_thread_pool);
  }

  /** Calculate how many iterations are required for training, testing,
   *  and validation given a specified mini-batch size.
   */
  for (auto&& dr : m_data_readers) {
    if (!dr.second)
      continue;
    calculate_num_iterations_per_epoch(max_mini_batch_size, dr.second);
  }

  auto& arg_parser = global_argument_parser();
  if (arg_parser.get<bool>(LBANN_OPTION_USE_DATA_STORE) ||
      arg_parser.get<bool>(LBANN_OPTION_PRELOAD_DATA_STORE) ||
      arg_parser.get<bool>(LBANN_OPTION_DATA_STORE_CACHE) ||
      arg_parser.get<std::string>(LBANN_OPTION_DATA_STORE_SPILL) != "") {
    bool master = m_comm->am_world_master();
    if (master) {
      std::cout << "\nUSING DATA STORE!\n\n";
    }
    for (auto&& r : m_data_readers) {
      if (!r.second)
        continue;
      r.second->setup_data_store(max_mini_batch_size);
    }
  }
}

void data_coordinator::calculate_num_iterations_per_epoch(
  int max_mini_batch_size,
  generic_data_reader* data_reader)
{
  if (data_reader == nullptr) {
    return;
  }
  // If the data reader does not have any data bail out (e.g. unused validation
  // reader)
  if (data_reader->get_num_data() == 0) {
    return;
  }

  if (max_mini_batch_size > data_reader->get_num_data()) {
    max_mini_batch_size = data_reader->get_num_data();
  }

  /// Check to make sure that there is enough data for all of the parallel
  /// readers
  int num_parallel_readers_per_model =
    compute_max_num_parallel_readers(data_reader->get_num_data(),
                                     max_mini_batch_size,
                                     this->m_comm->get_procs_per_trainer());
  data_reader->set_num_parallel_readers(num_parallel_readers_per_model);
  if (num_parallel_readers_per_model == 0 ||
      (num_parallel_readers_per_model !=
         this->m_comm->get_procs_per_trainer() &&
       num_parallel_readers_per_model != max_mini_batch_size)) {
    throw lbann_exception(
      std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
      " :: generic_data_distribution: number of parallel readers is zero");
  }

#ifdef LBANN_HAS_DISTCONV
  if (dc::is_cosmoflow_parallel_io_enabled()) {
    // #trainers is assumed to be 1.
    assert_eq(this->m_comm->get_num_trainers(), 1);
  }
#endif

  /// Set the basic parameters for stride and offset of the data reader
  int batch_stride = max_mini_batch_size;
  int base_offset = this->m_comm->get_rank_in_trainer();
#ifdef LBANN_HAS_DISTCONV
  base_offset =
    dc::get_input_rank(*(this->m_comm)) / dc::get_number_of_io_partitions();
#endif
  /// Set mini-batch size and stride
  data_reader->set_mini_batch_size(max_mini_batch_size);
  data_reader->set_stride_to_next_mini_batch(batch_stride);
#ifdef LBANN_HAS_DISTCONV
  data_reader->set_sample_stride(num_parallel_readers_per_model /
                                 dc::get_number_of_io_partitions());
#else
  data_reader->set_sample_stride(num_parallel_readers_per_model);
#endif
  data_reader->set_iteration_stride(1);
  /// Set data reader base offset and model offset
  data_reader->set_base_offset(base_offset);
  data_reader->set_model_offset(0);
  data_reader->set_initial_position();

  /// By default each data reader will plan to process the entire data set
  int num_iterations_per_epoch =
    ceil((float)data_reader->get_num_data() / (float)max_mini_batch_size);
  int last_mini_batch_size = data_reader->get_num_data() % max_mini_batch_size;
  if (last_mini_batch_size == 0) {
    last_mini_batch_size = max_mini_batch_size;
  }
  data_reader->set_num_iterations_per_epoch(num_iterations_per_epoch);
  data_reader->set_last_mini_batch_size(last_mini_batch_size);
  data_reader->set_stride_to_last_mini_batch(
    data_reader->get_stride_to_next_mini_batch());
  data_reader->set_global_mini_batch_size(max_mini_batch_size);
  data_reader->set_global_last_mini_batch_size(last_mini_batch_size);
  return;
}

void data_coordinator::calculate_num_iterations_per_epoch(int mini_batch_size)
{
  for (auto&& dr : m_data_readers) {
    if (!dr.second)
      continue;
    calculate_num_iterations_per_epoch(mini_batch_size, dr.second);
  }
}

int data_coordinator::compute_max_num_parallel_readers(
  long data_set_size,
  int mini_batch_size,
  int requested_num_parallel_readers) const
{
  return compute_max_num_parallel_readers(data_set_size,
                                          mini_batch_size,
                                          requested_num_parallel_readers,
                                          this->m_comm);
}

int data_coordinator::compute_max_num_parallel_readers(
  long data_set_size,
  int mini_batch_size,
  int requested_num_parallel_readers,
  const lbann_comm* comm)
{
  int num_parallel_readers = requested_num_parallel_readers;

  if (comm->get_procs_per_trainer() != num_parallel_readers) {
    if (comm->am_trainer_master()) {
      std::cout << "Warning the requested number of parallel readers "
                << num_parallel_readers << " does not match the grid size "
                << comm->get_procs_per_trainer()
                << " OVERRIDING requested number of parallel readers."
                << std::endl;
    }
    num_parallel_readers = comm->get_procs_per_trainer();
  }

#if 0
  if(mini_batch_size < num_parallel_readers) {
    if (comm->am_trainer_master()) {
      std::cout << "Warning the requested number of parallel readers "
                << num_parallel_readers
                << " is larger than the requested mini-batch size "
                << mini_batch_size
                << " OVERRIDING requested number of parallel readers."
                << std::endl;
    }
    num_parallel_readers = mini_batch_size;
  }
#endif
  return num_parallel_readers;
}

size_t data_coordinator::get_num_iterations_per_epoch(execution_mode mode) const
{
  const generic_data_reader* data_reader = get_data_reader(mode);
  return (data_reader != nullptr) ? data_reader->get_num_iterations_per_epoch()
                                  : 0;
}

int data_coordinator::get_current_step_in_epoch(execution_mode mode) const
{
  const generic_data_reader* data_reader = get_data_reader(mode);
  return (data_reader != nullptr) ? data_reader->get_current_step_in_epoch()
                                  : 0;
}

int data_coordinator::get_mini_batch_size(execution_mode mode) const
{
  const generic_data_reader* data_reader = get_data_reader(mode);
  return (data_reader != nullptr) ? data_reader->get_mini_batch_size() : 0;
}

int data_coordinator::get_last_mini_batch_size(execution_mode mode) const
{
  const generic_data_reader* data_reader = get_data_reader(mode);
  return (data_reader != nullptr) ? data_reader->get_last_mini_batch_size() : 0;
}

int data_coordinator::get_current_mini_batch_size(execution_mode mode) const
{
  const generic_data_reader* data_reader = get_data_reader(mode);
  return (data_reader != nullptr) ? data_reader->get_current_mini_batch_size()
                                  : 0;
}

int data_coordinator::get_global_mini_batch_size(execution_mode mode) const
{
  const generic_data_reader* data_reader = get_data_reader(mode);
  return (data_reader != nullptr) ? data_reader->get_global_mini_batch_size()
                                  : 0;
}

int data_coordinator::get_current_global_mini_batch_size(
  execution_mode mode) const
{
  const generic_data_reader* data_reader = get_data_reader(mode);
  return (data_reader != nullptr)
           ? data_reader->get_current_global_mini_batch_size()
           : 0;
}

int data_coordinator::get_global_last_mini_batch_size(execution_mode mode) const
{
  const generic_data_reader* data_reader = get_data_reader(mode);
  return (data_reader != nullptr)
           ? data_reader->get_global_last_mini_batch_size()
           : 0;
}

int data_coordinator::get_world_master_mini_batch_adjustment(
  execution_mode mode) const
{
  const generic_data_reader* data_reader = get_data_reader(mode);
  return (data_reader != nullptr)
           ? data_reader->get_world_master_mini_batch_adjustment()
           : 0;
}

int data_coordinator::get_current_world_master_mini_batch_adjustment(
  execution_mode mode,
  int model_rank) const
{
  const generic_data_reader* data_reader = get_data_reader(mode);
  return (data_reader != nullptr)
           ? data_reader->get_current_world_master_mini_batch_adjustment(
               model_rank)
           : 0;
}

// save state of IO to a checkpoint
bool data_coordinator::save_to_checkpoint_shared(persist& p) const
{
  // save state of data readers from input layer
  data_reader_map_t::const_iterator it;
  if (p.get_cb_type() == callback_type::execution_context_only ||
      p.get_cb_type() == callback_type::full_checkpoint) {

    it = this->m_data_readers.find(execution_mode::training);
    if ((it != this->m_data_readers.end()) && it->second) {
      (it->second)->save_to_checkpoint_shared(p, execution_mode::training);
    }
    it = this->m_data_readers.find(execution_mode::testing);
    if ((it != this->m_data_readers.end()) && it->second) {
      (it->second)->save_to_checkpoint_shared(p, execution_mode::testing);
    }
    it = this->m_data_readers.find(execution_mode::validation);
    if ((it != this->m_data_readers.end()) && it->second) {
      (it->second)->save_to_checkpoint_shared(p, execution_mode::validation);
    }

    // if (this->m_comm->am_trainer_master()) {
    //   write_cereal_archive<const data_coordinator>(*this, p,
    //   execution_mode::training, "_dc.xml");
    // }
  }
  return true;
}

// reload state of IO from a checkpoint
bool data_coordinator::load_from_checkpoint_shared(persist& p)
{
  // save state of data readers from input layer
  data_reader_map_t::const_iterator it;
  if (p.get_cb_type() == callback_type::execution_context_only ||
      p.get_cb_type() == callback_type::full_checkpoint) {

    it = this->m_data_readers.find(execution_mode::training);
    if ((it != this->m_data_readers.end()) && it->second) {
      (it->second)->load_from_checkpoint_shared(p, execution_mode::training);
    }
    it = this->m_data_readers.find(execution_mode::testing);
    if ((it != this->m_data_readers.end()) && it->second) {
      (it->second)->load_from_checkpoint_shared(p, execution_mode::testing);
    }
    it = this->m_data_readers.find(execution_mode::validation);
    if ((it != this->m_data_readers.end()) && it->second) {
      (it->second)->load_from_checkpoint_shared(p, execution_mode::validation);
    }

    // std::string buf;
    // if (this->m_comm->am_trainer_master()) {
    //   read_cereal_archive<data_coordinator>(*this, p,
    //   execution_mode::training, "_dc.xml"); buf =
    //   create_cereal_archive_binary_string<data_coordinator>(*this);
    // }

    // // TODO: this assumes homogeneous processors
    // // broadcast state from rank 0
    // this->m_comm->trainer_broadcast(0, buf);

    // if (!this->m_comm->am_trainer_master()) {
    //   unpack_cereal_archive_binary_string<data_coordinator>(*this, buf);
    // }
  }

  return true;
}

bool data_coordinator::save_to_checkpoint_distributed(persist& p) const
{
  // save state of data readers from input layer
  data_reader_map_t::const_iterator it;
  if (p.get_cb_type() == callback_type::execution_context_only ||
      p.get_cb_type() == callback_type::full_checkpoint) {

    it = this->m_data_readers.find(execution_mode::training);
    if ((it != this->m_data_readers.end()) && it->second) {
      (it->second)->save_to_checkpoint_distributed(p, execution_mode::training);
    }
    it = this->m_data_readers.find(execution_mode::testing);
    if ((it != this->m_data_readers.end()) && it->second) {
      (it->second)->save_to_checkpoint_distributed(p, execution_mode::testing);
    }
    it = this->m_data_readers.find(execution_mode::validation);
    if ((it != this->m_data_readers.end()) && it->second) {
      (it->second)
        ->save_to_checkpoint_distributed(p, execution_mode::validation);
    }

    // write_cereal_archive<const data_coordinator>(*this, p,
    // execution_mode::training, "_dc.xml");
  }
  return true;
}

bool data_coordinator::load_from_checkpoint_distributed(persist& p)
{
  // save state of data readers from input layer
  data_reader_map_t::const_iterator it;
  it = this->m_data_readers.find(execution_mode::training);
  if ((it != this->m_data_readers.end()) && it->second) {
    (it->second)->load_from_checkpoint_distributed(p, execution_mode::training);
  }
  it = this->m_data_readers.find(execution_mode::testing);
  if ((it != this->m_data_readers.end()) && it->second) {
    (it->second)->load_from_checkpoint_distributed(p, execution_mode::testing);
  }
  it = this->m_data_readers.find(execution_mode::validation);
  if ((it != this->m_data_readers.end()) && it->second) {
    (it->second)
      ->load_from_checkpoint_distributed(p, execution_mode::validation);
  }

  // read_cereal_archive<data_coordinator>(*this, p, execution_mode::training,
  // "_dc.xml");
  return true;
}

} // namespace lbann

#define LBANN_CLASS_NAME data_coordinator
#include <lbann/macros/register_class_with_cereal.hpp>
