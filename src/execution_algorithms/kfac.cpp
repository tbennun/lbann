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

#include "lbann/execution_algorithms/kfac.hpp"
#include "lbann/execution_algorithms/kfac/execution_context.hpp"
#include "lbann/execution_algorithms/sgd_training_algorithm.hpp"

#include "lbann/base.hpp"
#include "lbann/callbacks/callback.hpp"
#include "lbann/execution_algorithms/kfac/kfac_block.hpp"
#include "lbann/execution_algorithms/kfac/kfac_block_bn.hpp"
#include "lbann/execution_algorithms/kfac/kfac_block_fc_conv.hpp"
#include "lbann/execution_algorithms/kfac/kfac_block_gru.hpp"
#include "lbann/execution_algorithms/kfac/kfac_util.hpp"
#include "lbann/layers/data_type_layer.hpp"
#include "lbann/layers/learning/convolution.hpp"
#include "lbann/layers/learning/fully_connected.hpp"
#include "lbann/layers/regularizers/batch_normalization.hpp"
#include "lbann/models/model.hpp"
#include "lbann/utils/memory.hpp"
#include "lbann/utils/profiling.hpp"

#include <training_algorithm.pb.h>

#include <cstddef>
#include <limits>

namespace lbann {

/// @todo Initialize properly
KFAC::KFAC(std::string name,
           std::unique_ptr<TermCriteriaType> stop,
           std::vector<double> damping_act_params,
           std::vector<double> damping_err_params,
           std::vector<double> damping_bn_act_params,
           std::vector<double> damping_bn_err_params,
           size_t damping_warmup_steps,
           double kronecker_decay,
           bool print_time,
           bool print_matrix,
           bool print_matrix_summary,
           bool use_pi,
           std::vector<size_t> update_intervals,
           size_t update_interval_steps,
           kfac::kfac_inverse_strategy inverse_strategy,
           std::vector<std::string> disable_layers,
           double learning_rate_factor,
           double learning_rate_factor_gru,
           size_t compute_interval)
  : TrainingAlgorithm{std::move(name)},
    m_stopping_criteria{std::move(stop)},
    m_damping_act_params{std::move(damping_act_params)},
    m_damping_err_params{std::move(damping_err_params)},
    m_damping_bn_act_params{std::move(damping_bn_act_params)},
    m_damping_bn_err_params{std::move(damping_bn_err_params)},
    m_damping_warmup_steps{std::move(damping_warmup_steps)},
    m_kronecker_decay{kronecker_decay},
    m_print_time{print_time},
    m_print_matrix{print_matrix},
    m_print_matrix_summary{print_matrix_summary},
    m_use_pi{use_pi},
    m_update_intervals{std::move(update_intervals)},
    m_update_interval_steps{update_interval_steps},
    m_inverse_strategy{inverse_strategy},
    m_disable_layers{std::move(disable_layers)},
    m_learning_rate_factor{learning_rate_factor},
    m_learning_rate_factor_gru{learning_rate_factor_gru},
    m_compute_interval{compute_interval}
{}

std::string KFAC::get_type() const { return "KFAC"; }

kfac::KFACExecutionContext* KFAC::do_get_new_execution_context() const
{
  return new kfac::KFACExecutionContext(0UL,
                                        m_damping_act_params[0],
                                        m_damping_err_params[0],
                                        m_damping_bn_act_params[0],
                                        m_damping_bn_err_params[0]);
}

// =============================================
// Evaluation and training
// =============================================

void KFAC::apply(ExecutionContext& context_,
                 model& model,
                 data_coordinator& dc,
                 execution_mode mode)
{
  ExeContextType& context = dynamic_cast<ExeContextType&>(context_);
  if (mode == execution_mode::training) {
    train(context, model, dc, *m_stopping_criteria);
  }
  else {
    SGDTrainingAlgorithm eval_algo(this->get_name() + "_eval",
                                   m_stopping_criteria->clone(),
                                   /*suppress_timer=*/true);
    auto& eval_context = context.get_sgd_execution_context();
    eval_algo.apply(eval_context, model, dc, mode);
  }
}

void KFAC::train(ExeContextType& kfac_context,
                 model& model,
                 data_coordinator& dc,
                 TermCriteriaType const& term)
{
  // Initialize some state so it knows we're training now.
  auto& sgd_context = kfac_context.get_sgd_execution_context();
  sgd_context.set_execution_mode(execution_mode::training);
  model.reset_mode(sgd_context, execution_mode::training);
  dc.reset_mode(sgd_context);

  // Get lbann comm
  auto& comm = *model.get_comm();

  // Reset KFAC context
  kfac_context.m_damping_act = m_damping_act_params[0];
  kfac_context.m_damping_err = m_damping_err_params[0];
  kfac_context.m_damping_bn_act = m_damping_bn_act_params[0];
  kfac_context.m_damping_bn_err = m_damping_bn_err_params[0];

  // Run callbacks.
  do_train_begin_cbs(model);

  // Start iterating
  bool is_start_of_epoch = true;
  sgd_context.start_timer();
  while (!term(sgd_context)) {

    if (is_start_of_epoch) {
      // Initialize epoch
      model.reset_mode(sgd_context, execution_mode::training);
      model.reset_epoch_statistics(execution_mode::training);
      dc.reset_mode(sgd_context);
      do_epoch_begin_cbs(model);
      is_start_of_epoch = false;
      // sync weights if we have a separate model for primary and secondary grid
      if (comm.get_KFAC_subgrid_create_two_models() and
          comm.get_grid_type() != GridType::NO_GRID)
        sync_weights_model(model, model.get_comm());
    }

    // Train a mini batch. Returns "true" if the data_coordinator
    // detects the end of an epoch.
    if (train_mini_batch(kfac_context, model, dc)) {
      // Finalize epoch
      sgd_context.inc_epoch();

      if (comm.get_KFAC_subgrid_create_two_models() or
          comm.get_grid_type() == GridType::NO_GRID or
          comm.get_grid_type() == GridType::PRIMARY_GRID) {
        model.reconcile_weight_values();
        do_epoch_end_cbs(model);

        // Evaluate on validation set
        //
        // FIXME (trb 05/04/2021): Upon further refactor, this should
        // move out of the main training cycle and become part of an
        // "evaluation policy" or something of that nature, ideally with
        // its own context that we needn't know about.
        if (dc.is_execution_mode_valid(execution_mode::validation)) {
          const execution_mode eval_mode = execution_mode::validation;
          SGDExecutionContext eval_context(eval_mode,
                                           dc.get_mini_batch_size(eval_mode));
          // FIXME (trb 05/05/2021): This hacks around a bad assumption
          // in the data store.
          // Note (tym 6/7/21): Copied from sgd_training_algorithm.cpp.
          size_t num_validation_epochs = 1UL;
          if (sgd_context.get_epoch() > 1UL) {
            eval_context.inc_epoch();
            ++num_validation_epochs;
          }
          SGDTrainingAlgorithm eval_algo(
            this->get_name() + "_eval",
            std::make_unique<EpochTerminationCriteria>(num_validation_epochs),
            /*suppress_timer=*/true);
          eval_algo.apply(eval_context, model, dc, eval_mode);

          // FIXME (trb 06/07/21): The early stopping callback is part
          // of the evaluation callbacks but it's meant to affect
          // training. This fixes a bug in which the training context
          // was meant to stop but was never properly told.
          sgd_context.set_early_stop(eval_context.get_early_stop());
        }

        // Trigger new epoch stuff next iteration (if there is one).
        is_start_of_epoch = true;
      }
    }
  }

  sgd_context.stop_timer();

  // Reset the model back to the training execution context prior to
  // end of training callbacks
  model.reset_mode(sgd_context, execution_mode::training);
  if (comm.get_KFAC_subgrid_create_two_models() or
      comm.get_grid_type() == GridType::NO_GRID or
      comm.get_grid_type() == GridType::PRIMARY_GRID)
    do_train_end_cbs(model);
}

// =============================================
// Mini-batch step
// =============================================

// Returns "true" if the data_coordinator detects the end of an epoch.
bool KFAC::train_mini_batch(ExeContextType& kfac_context,
                            model& model,
                            data_coordinator& dc)
{
  auto& sgd_context = kfac_context.get_sgd_execution_context();

  model.reset_mode(sgd_context, execution_mode::training);
  dc.reset_mode(sgd_context);
  do_batch_begin_cbs(model);

  bool finished = false;
  auto& comm = *model.get_comm();
  const bool compute_inverse =
    sgd_context.get_step() % this->m_compute_interval == 0;

  dc.fetch_data(execution_mode::training);

#if defined(LBANN_HAVE_OMP_TASKLOOP)
  LBANN_OMP_PARALLEL
  {
#pragma omp single
    {
#endif
      if (comm.get_grid_type() == GridType::PRIMARY_GRID or
          comm.get_KFAC_subgrid_create_two_models() or
          comm.get_grid_type() == GridType::NO_GRID) {
        // Forward prop step
        model.clear_gradients();
        // sync_weights_model(model, model.get_comm());
        model.forward_prop(execution_mode::training);
      }
      if (compute_inverse)
        on_forward_prop_end(kfac_context, model);

      if (comm.get_grid_type() == GridType::PRIMARY_GRID or
          comm.get_KFAC_subgrid_create_two_models() or
          comm.get_grid_type() == GridType::NO_GRID) {
        // check if the data coordinator has finished the epoch and kickoff
        // background I/O
        finished = dc.epoch_complete(execution_mode::training);

        // Result is not needed until the end of the mini-batch.
        model.get_objective_function()->start_evaluation(
          execution_mode::training,
          sgd_context.get_current_mini_batch_size());

        // Backward prop step
        model.get_objective_function()->differentiate();
        model.backward_prop();
      }
      else {
        finished = dc.epoch_complete(execution_mode::training);
      }

      if (compute_inverse)
        on_backward_prop_end(kfac_context, model);
      else if (comm.get_grid_type() == GridType::NO_GRID or
               m_has_kronecker_inverse == true) {
        if (comm.get_KFAC_subgrid_create_two_models() or
            comm.get_grid_type() == GridType::PRIMARY_GRID or
            comm.get_grid_type() == GridType::NO_GRID) {
          for (auto& block : kfac_context.m_blocks) {
            const bool is_gru =
              dynamic_cast<kfac_block_gru<Device>*>(block.get()) != nullptr;
            block->compute_preconditioned_gradients(
              &comm,
              is_gru ? m_learning_rate_factor_gru : m_learning_rate_factor,
              m_print_matrix,
              m_print_matrix_summary,
              m_print_time);
          }
        }
      }

      if (comm.get_grid_type() == GridType::PRIMARY_GRID or
          comm.get_KFAC_subgrid_create_two_models() or
          comm.get_grid_type() == GridType::NO_GRID) {
        model.get_objective_function()->compute_weight_regularization();

        // Finish evaluation.
        model.get_objective_function()->finish_evaluation(
          execution_mode::training,
          sgd_context.get_current_mini_batch_size());
        model.evaluate_metrics(execution_mode::training,
                               sgd_context.get_current_mini_batch_size());

        // Update step
        model.update_weights();
        model.update_layers();
      }
#if defined(LBANN_HAVE_OMP_TASKLOOP)
    }
  }
#endif

  if (compute_inverse) {
    kfac_context.inc_step();
  }

  sgd_context.inc_step();

  if (comm.get_KFAC_subgrid_create_two_models() or
      comm.get_grid_type() == GridType::NO_GRID or
      comm.get_grid_type() == GridType::PRIMARY_GRID)
    do_batch_end_cbs(model);
  return finished;
}

// =============================================
// Callbacks
// =============================================

void KFAC::do_train_begin_cbs(model& model)
{
  for (const auto& cb : model.get_callbacks()) {
    cb->on_train_begin(&model);
  }
}

void KFAC::do_train_end_cbs(model& model)
{
  for (const auto& cb : model.get_callbacks()) {
    cb->on_train_end(&model);
  }
}

void KFAC::do_epoch_begin_cbs(model& model)
{
  for (const auto& cb : model.get_callbacks()) {
    cb->on_epoch_begin(&model);
  }
}

void KFAC::do_epoch_end_cbs(model& model)
{
  for (const auto& cb : model.get_callbacks()) {
    cb->on_epoch_end(&model);
  }
}

void KFAC::do_batch_begin_cbs(model& model)
{
  SGDExecutionContext& c =
    static_cast<SGDExecutionContext&>(model.get_execution_context());
  for (const auto& cb : model.get_callbacks()) {
    if (c.get_step() % cb->get_batch_interval() == 0) {
      cb->on_batch_begin(&model);
    }
  }
}

void KFAC::do_batch_end_cbs(model& model)
{
  SGDExecutionContext& c =
    static_cast<SGDExecutionContext&>(model.get_execution_context());
  for (const auto& cb : model.get_callbacks()) {
    if (c.get_step() % cb->get_batch_interval() == 0) {
      cb->on_batch_end(&model);
    }
  }
}

// =============================================
// Sub-grid implementation
// =============================================

void KFAC::sync_weights_model(model& model, lbann_comm* comm)
{
  // Does not support Model parallel only Data Parallel

  const auto layers = model.get_layers();
  const El::mpi::Comm& combined_comm = comm->get_combined_grid_comm();
  // const El::mpi::Comm & trainer_comm = comm->get_KFAC_comm();
  // const int comm_size = El::mpi::Size(comm->get_KFAC_comm());
  const int comm_rank = El::mpi::Rank(comm->get_KFAC_comm());

  std::vector<int> primary_grid_ranks = comm->get_primary_grid_ranks();
  std::vector<int> secondary_grid_ranks = comm->get_secondary_grid_ranks();

  int num_process_primary_grid = (int)primary_grid_ranks.size();
  int num_process_secondary_grid = (int)secondary_grid_ranks.size();

  // Computing size of global buffer
  int global_buffer_size = 0;
  for (auto i_layer = layers.begin(); i_layer != layers.end(); i_layer++) {
    const auto& layer = *i_layer;
    const size_t num_weights = layer->num_weights();
    for (size_t idx_weight = 0; idx_weight < num_weights; ++idx_weight) {
      auto& weights = layer->get_weights(idx_weight);

      // Ignore weights without optimizer
      //  const optimizer *w_optimizer = weights.get_optimizer();
      //  if(w_optimizer == nullptr)
      //    continue;

      int height = weights.get_matrix_height();
      int width = weights.get_matrix_width();
      global_buffer_size += height * width;
    }
  }

  El::Matrix<DataType, Device> global_buffer(global_buffer_size, 1);
  El::SyncInfo<Device> sync_info = El::SyncInfoFromMatrix(global_buffer);

  size_t offset = 0;
  // copy weights to global buffers
  for (auto i_layer = layers.begin(); i_layer != layers.end(); i_layer++) {
    const auto& layer = *i_layer;
    const size_t num_weights = layer->num_weights();
    for (size_t idx_weight = 0; idx_weight < num_weights; ++idx_weight) {
      auto& weights = layer->get_weights(idx_weight);

      // Ignore weights without optimizer
      //  const optimizer *w_optimizer = weights.get_optimizer();
      //  if(w_optimizer == nullptr)
      //    continue;

      data_type_weights<DataType>* weights_dt =
        dynamic_cast<data_type_weights<DataType>*>(&weights);

      int height = weights.get_matrix_height();
      int width = weights.get_matrix_width();

      El::AbstractDistMatrix<DataType>& weight_values =
        weights_dt->get_values();
      El::DistMatrix<DataType, El::STAR, El::STAR, El::ELEMENT, Device>*
        weight_values_ = dynamic_cast<
          El::DistMatrix<DataType, El::STAR, El::STAR, El::ELEMENT, Device>*>(
          &weight_values);

      auto view = El::View(global_buffer,
                           El::IR(offset, offset + height * width),
                           El::ALL);
      offset += height * width;

      // El::Copy(weight_values.LockedMatrix(), view);
      El::copy::util::InterleaveMatrix(height,
                                       width,
                                       weight_values_->LockedBuffer(),
                                       1,
                                       height,
                                       view.Buffer(),
                                       1,
                                       height,
                                       sync_info);
    }
  }

  if (comm->get_grid_type() == GridType::PRIMARY_GRID) {
    int num_sends = (int)std::ceil((float)num_process_secondary_grid /
                                   (float)num_process_primary_grid);
    for (int num_send = 0; num_send < num_sends; num_send++) {

      if (comm_rank + num_send * num_process_primary_grid <
          num_process_secondary_grid) {
        int to_send_index = comm_rank + num_send * num_process_primary_grid;
        ::El::mpi::Send((DataType*)global_buffer.Buffer(),
                        global_buffer_size,
                        secondary_grid_ranks[to_send_index],
                        combined_comm,
                        sync_info);
      }
    }
  }
  if (comm->get_grid_type() == GridType::SECONDARY_GRID) {
    int recv_index = comm_rank % num_process_primary_grid;
    // std::cout<<"My CommRank:"<<comm_rank<<"
    // Recv:"<<primary_grid_ranks[recv_index]<<"\n";
    ::El::mpi::Recv((DataType*)global_buffer.Buffer(),
                    global_buffer_size,
                    primary_grid_ranks[recv_index],
                    combined_comm,
                    sync_info);
  }

  // copy weights from global buffers
  offset = 0;
  for (auto i_layer = layers.begin(); i_layer != layers.end(); i_layer++) {
    const auto& layer = *i_layer;
    const size_t num_weights = layer->num_weights();
    for (size_t idx_weight = 0; idx_weight < num_weights; ++idx_weight) {
      auto& weights = layer->get_weights(idx_weight);

      // Ignore weights without optimizer
      //  const optimizer *w_optimizer = weights.get_optimizer();
      //  if(w_optimizer == nullptr)
      //    continue;

      data_type_weights<DataType>* weights_dt =
        dynamic_cast<data_type_weights<DataType>*>(&weights);

      int height = weights.get_matrix_height();
      int width = weights.get_matrix_width();

      El::Matrix<DataType, Device> weight_buffer(height, width);

      El::AbstractDistMatrix<DataType>& weight_values =
        weights_dt->get_values();
      El::DistMatrix<DataType, El::STAR, El::STAR, El::ELEMENT, Device>*
        weight_values_ = dynamic_cast<
          El::DistMatrix<DataType, El::STAR, El::STAR, El::ELEMENT, Device>*>(
          &weight_values);

      auto view = El::View(global_buffer,
                           El::IR(offset, offset + height * width),
                           El::ALL);
      offset += height * width;

      El::copy::util::InterleaveMatrix(height,
                                       width,
                                       view.Buffer(),
                                       1,
                                       height,
                                       weight_values_->Buffer(),
                                       1,
                                       height,
                                       sync_info);
      // weights.set_values(weight_buffer);
    }
  }
}

template <typename DataType, El::Device Device>
void send_recv_precomputed_gradients(
  const std::vector<std::pair<size_t, El::AbstractMatrix<DataType>*>>& blocks,
  El::Matrix<DataType, Device>& global_buffer,
  const int data_size,
  lbann_comm* comm,
  const kfac::kfac_allgather_mode& mode)
{

  const int comm_rank = El::mpi::Rank(comm->get_KFAC_comm());
  const int combined_rank = El::mpi::Rank(comm->get_combined_grid_comm());

  std::vector<int> primary_grid_ranks = comm->get_primary_grid_ranks();
  std::vector<int> secondary_grid_ranks = comm->get_secondary_grid_ranks();

  int num_process_primary_grid = (int)primary_grid_ranks.size();
  int num_process_secondary_grid = (int)secondary_grid_ranks.size();

  const El::mpi::Comm& combined_comm = comm->get_combined_grid_comm();

  if (comm->get_grid_type() == GridType::SECONDARY_GRID) {
    int num_sends = (int)std::ceil((float)num_process_primary_grid /
                                   (float)num_process_secondary_grid);

    for (int num_send = 0; num_send < num_sends; num_send++) {

      if (comm_rank + num_send * num_process_secondary_grid <
          num_process_primary_grid) {

        El::Matrix<DataType, Device> global_buffer_local(data_size, 1);
        El::SyncInfo<Device> sync_info =
          El::SyncInfoFromMatrix(global_buffer_local);

        El::copy::util::InterleaveMatrix(data_size,
                                         1,
                                         global_buffer.Buffer(),
                                         1,
                                         data_size,
                                         global_buffer_local.Buffer(),
                                         1,
                                         data_size,
                                         sync_info);

        int to_send_index = comm_rank + num_send * num_process_secondary_grid;
        ::El::mpi::TaggedSend((DataType*)global_buffer_local.Buffer(),
                              data_size,
                              primary_grid_ranks[to_send_index],
                              primary_grid_ranks[to_send_index],
                              combined_comm,
                              sync_info);
      }
    }
  }
  if (comm->get_grid_type() == GridType::PRIMARY_GRID) {
    El::SyncInfo<Device> sync_info = El::SyncInfoFromMatrix(global_buffer);
    int recv_index = comm_rank % num_process_secondary_grid;
    ::El::mpi::TaggedRecv((DataType*)global_buffer.Buffer(),
                          data_size,
                          secondary_grid_ranks[recv_index],
                          combined_rank,
                          combined_comm,
                          sync_info);

    // Sort blocks so that received blocks per process become
    // contiguous.
    std::vector<std::pair<size_t, El::AbstractMatrix<DataType>*>> sorted_blocks(
      blocks.size());
    std::copy(blocks.begin(), blocks.end(), sorted_blocks.begin());
    if (mode == kfac::kfac_allgather_mode::ALLGATHER)
      std::stable_sort(
        sorted_blocks.begin(),
        sorted_blocks.end(),
        [](const std::pair<size_t, El::AbstractMatrix<DataType>*>& lhs,
           const std::pair<size_t, El::AbstractMatrix<DataType>*>& rhs) {
          return lhs.first < rhs.first;
        });

    // Copy blocks from the buffer.
    {
      size_t offset = 0;
      for (auto& block : sorted_blocks) {
        if (block.first != (size_t)comm->get_rank_in_trainer()) {
          const auto view =
            El::LockedView(global_buffer,
                           El::IR(offset, offset + block.second->Height()),
                           El::ALL);
          El::Copy(view, *block.second);
        }
        offset += block.second->Height();
      }
    }
  }
}

void KFAC::send_recv_inverse_matrices(ExeContextType& context, lbann_comm* comm)
{

  int global_buffer_inverses_size = 0;

  // calculate the size of the buffer
  for (auto& block : context.m_blocks) {
    global_buffer_inverses_size += block->get_inverse_matrices_size(comm);
  }

  El::Matrix<DataType, Device>& global_buffer_inverse =
    context.get_workspace_matrix("allgather_inverse_recv_buffer",
                                 global_buffer_inverses_size,
                                 1);

  size_t offset = 0;
  // Copy data into buffer
  if (comm->get_grid_type() == GridType::SECONDARY_GRID) {
    for (auto& block : context.m_blocks) {
      offset = block->get_inverse_matrices(global_buffer_inverse, offset);
    }
  }

  const int comm_rank = comm->get_rank_in_trainer();
  const int combined_rank = El::mpi::Rank(comm->get_combined_grid_comm());

  std::vector<int> primary_grid_ranks = comm->get_primary_grid_ranks();
  std::vector<int> secondary_grid_ranks = comm->get_secondary_grid_ranks();

  int num_process_primary_grid = (int)primary_grid_ranks.size();
  int num_process_secondary_grid = (int)secondary_grid_ranks.size();

  const El::mpi::Comm& combined_comm = comm->get_combined_grid_comm();

  if (comm->get_grid_type() == GridType::SECONDARY_GRID) {
    int num_sends = (int)std::ceil((float)num_process_primary_grid /
                                   (float)num_process_secondary_grid);

    for (int num_send = 0; num_send < num_sends; num_send++) {

      if (comm_rank + num_send * num_process_secondary_grid <
          num_process_primary_grid) {

        El::SyncInfo<Device> sync_info =
          El::SyncInfoFromMatrix(global_buffer_inverse);

        int to_send_index = comm_rank + num_send * num_process_secondary_grid;
        ::El::mpi::TaggedSend((DataType*)global_buffer_inverse.Buffer(),
                              global_buffer_inverses_size,
                              primary_grid_ranks[to_send_index],
                              primary_grid_ranks[to_send_index],
                              combined_comm,
                              sync_info);
      }
    }
  }
  if (comm->get_grid_type() == GridType::PRIMARY_GRID) {
    El::SyncInfo<Device> sync_info =
      El::SyncInfoFromMatrix(global_buffer_inverse);
    int recv_index = comm_rank % num_process_secondary_grid;
    ::El::mpi::TaggedRecv((DataType*)global_buffer_inverse.Buffer(),
                          global_buffer_inverses_size,
                          secondary_grid_ranks[recv_index],
                          combined_rank,
                          combined_comm,
                          sync_info);

    // Copy blocks from the buffer.
    {
      offset = 0;
      for (auto& block : context.m_blocks) {
        offset =
          block->set_inverse_matrices(global_buffer_inverse, offset, comm);
      }
    }
  }
}

// =============================================
// KFAC implementation
// =============================================

void KFAC::on_forward_prop_end(ExeContextType& context, model& model)
{

  auto& comm = *model.get_comm();
  const auto layers = model.get_layers();

  // List up layers to be updated
  if (context.m_blocks.size() == 0) {
    prof_region_begin("kfac-setup", prof_color, prof_sync);
    const size_t num_procs = comm.get_procs_per_trainer();
    std::unordered_map<std::string, int> proc_ranks;
    for (auto i_layer = layers.begin(); i_layer != layers.end(); i_layer++) {
      const size_t layer_id = std::distance(layers.begin(), i_layer);
      const auto& l = *i_layer;
      const auto l_fc = dynamic_cast<
        fully_connected_layer<DataType, data_layout::DATA_PARALLEL, Device>*>(
        l);
      const auto l_conv = dynamic_cast<
        convolution_layer<DataType, data_layout::DATA_PARALLEL, Device>*>(l);
      const auto l_bn =
        dynamic_cast<batch_normalization_layer<DataType,
                                               data_layout::DATA_PARALLEL,
                                               Device>*>(l);
      const auto l_gru =
        dynamic_cast<gru_layer<DataType, data_layout::DATA_PARALLEL, Device>*>(
          l);
      const bool is_fc = (l_fc != nullptr);
      const bool is_conv = (l_conv != nullptr);
      const bool is_bn = (l_bn != nullptr);
      const bool is_gru = (l_gru != nullptr);
      if (!(is_fc || is_conv || is_bn || is_gru))
        continue;

      if (std::find(m_disable_layers.begin(),
                    m_disable_layers.end(),
                    l->get_name()) != m_disable_layers.end()) {
        if (comm.am_trainer_master())
          std::cout << "K-fac: " << l->get_name()
                    << " is ignored to optimize with K-FAC." << std::endl;
        continue;
      }

      prof_region_begin(("kfac-setup/" + l->get_name()).c_str(),
                        prof_color,
                        prof_sync);

      // Ignore layers without optimizers
      const auto& weights = l->get_weights(0);
      const optimizer* w_optimizer = weights.get_optimizer();
      if (w_optimizer == nullptr)
        continue;

      std::string proc_rank_key = "all";
      if (m_inverse_strategy == kfac::kfac_inverse_strategy::EACH)
        proc_rank_key = l->get_type();
      if (proc_ranks.find(proc_rank_key) == proc_ranks.end())
        proc_ranks[proc_rank_key] = 0;
      int& proc_rank = proc_ranks[proc_rank_key];

      // Check layer property
      if ((l->get_num_parents() != 1 || l->get_num_children() != 1) &&
          !is_gru) {
        std::stringstream err;
        err << "K-FAC expects layers who have exact one parent and child."
            << " layer: " << l->get_name()
            << ", #parent: " << l->get_num_parents()
            << ", #child: " << l->get_num_children();
        LBANN_ERROR(err.str());
      }

      std::shared_ptr<kfac_block<Device>> block;
      if (is_fc || is_conv) {
        block = std::make_shared<kfac_block_fc_conv<Device>>(l,
                                                             &context,
                                                             layer_id,
                                                             proc_rank,
                                                             is_conv);
      }
      else if (is_bn) {
        block = std::make_shared<kfac_block_bn<Device>>(l,
                                                        &context,
                                                        layer_id,
                                                        proc_rank);
      }
      else if (is_gru) {
        block = std::make_shared<kfac_block_gru<Device>>(l,
                                                         &context,
                                                         layer_id,
                                                         proc_rank);
      }

      context.m_blocks.push_back(std::move(block));
      if (m_inverse_strategy != kfac::kfac_inverse_strategy::ROOT)
        proc_rank = (proc_rank + 1) % num_procs;

      prof_region_end(("kfac-setup/" + l->get_name()).c_str(), prof_sync);
    }

    if (comm.am_trainer_master()) {
      for (const auto& block : context.m_blocks)
        std::cout << "K-FAC setup: " << block->get_info() << std::endl;
    }

    prof_region_end("kfac-setup", prof_sync);
  }

  for (auto& block : context.m_blocks)
    block->on_forward_prop_end(&comm);

  if (context.get_step() > 1 and
      (comm.get_grid_type() == GridType::PRIMARY_GRID or
       comm.get_grid_type() == GridType::SECONDARY_GRID)) {
    send_recv_inverse_matrices(context, &comm);
    m_has_kronecker_inverse = true;
  }
}

void KFAC::on_backward_prop_end(ExeContextType& context, model& model)
{

  // Get some configs
  auto& comm = *model.get_comm();
  const auto& sgd_context = context.get_sgd_execution_context();
  const size_t num_steps = sgd_context.get_step();
  const auto layers = model.get_layers();
  const bool is_first_step = (!m_has_kronecker_inverse);

  // Update the damping value
  // using a modified Tikhonov damping tequnique from
  // http://arxiv.org/abs/1811.12019
  const auto get_next_damping = [](const double damping_prev,
                                   const std::vector<double> damping_params,
                                   const double damping_warmup_steps) {
    if (damping_params.size() == 1)
      return damping_params[0];
    const DataType alpha =
      2.0 * log10(damping_params[0] / damping_params[1]) / damping_warmup_steps;
    return (1.0 - alpha) * damping_prev + alpha * damping_params[1];
  };
  context.m_damping_act = get_next_damping(context.m_damping_act,
                                           m_damping_act_params,
                                           m_damping_warmup_steps);
  context.m_damping_err = get_next_damping(context.m_damping_err,
                                           m_damping_err_params,
                                           m_damping_warmup_steps);
  context.m_damping_bn_act = get_next_damping(context.m_damping_bn_act,
                                              m_damping_bn_act_params,
                                              m_damping_warmup_steps);
  context.m_damping_bn_err = get_next_damping(context.m_damping_bn_err,
                                              m_damping_bn_err_params,
                                              m_damping_warmup_steps);

  // Update the udpate interval
  if (m_update_intervals.size() == 1)
    context.m_update_interval = m_update_intervals[0];
  else {
    context.m_update_interval =
      m_update_intervals[0] +
      ((double)m_update_intervals[1] - m_update_intervals[0]) *
        std::min((double)num_steps / m_update_interval_steps, 1.0);
  }

  // List up layers to be updated
  if (context.m_blocks.size() == 0) {
    LBANN_ERROR("K-FAC blocks have not been setup");
  }
  for (auto& block : context.m_blocks) {
    // Exchange activations and errors
    block->initialize_activations_and_errors(&comm, 1, 1, 0);
  }

  if (comm.get_grid_type() == GridType::SECONDARY_GRID or
      comm.get_grid_type() == GridType::NO_GRID) {
    prof_region_begin("kfac-step", prof_color, prof_sync);

    // Step 1: Ensure that each process has averaged Kronecker factors
    // for the model-parallel part.
    // const bool is_first_step = (!m_has_kronecker_inverse);
    const bool is_kronecker_update_required =
      ((num_steps % context.m_update_interval) == 0 ||
       !m_has_kronecker_inverse);
    if (is_kronecker_update_required) {
      prof_region_begin("kfac-update", prof_color, prof_sync);

      prof_region_begin("kfac-update/local", prof_color, prof_sync);
      for (auto& block : context.m_blocks) {
        prof_region_begin(("kfac-update/local/" + block->get_name()).c_str(),
                          prof_color,
                          prof_sync);

        block->compute_local_kronecker_factors(&comm,
                                               m_print_matrix,
                                               m_print_matrix_summary);
        prof_region_end(("kfac-update/local/" + block->get_name()).c_str(),
                        prof_sync);
      }
      prof_region_end("kfac-update/local", prof_sync);

#ifdef LBANN_NVPROF
      prof_region_begin("kfac-update/local-barrier", prof_color, prof_sync);
      CHECK_CUDA(cudaDeviceSynchronize());
      comm.trainer_barrier();
      prof_region_end("kfac-update/local-barrier", prof_sync);
#endif // LBANN_NVPROF

      // List-up buffers to synchronize.
      std::vector<std::pair<size_t, El::AbstractMatrix<DataType>*>> buffers;
      size_t global_buffer_size = 0;
      for (auto& block : context.m_blocks)
        for (auto L : block->get_local_kronecker_buffers()) {
          const size_t rank = block->get_inverse_proc_rank();
          buffers.emplace_back(rank, L);
          assert(L->Width() == 1);
          global_buffer_size += L->Height();
        }

      // Perform reduce-scatter.
      prof_region_begin("kfac-update/reduce-scatter", prof_color, prof_sync);
      const auto reduce_scatter_mode =
        kfac::kfac_reduce_scatter_mode::ALLREDUCE;
      El::Matrix<DataType, Device>& global_buffer =
        context.get_workspace_matrix(
          "reduce_scatter_send_buffer",
          kfac::is_reduce_scatter_buffer_required(reduce_scatter_mode)
            ? global_buffer_size
            : 0,
          1);
      kfac::reduce_scatter_blocks(buffers,
                                  global_buffer,
                                  &comm,
                                  reduce_scatter_mode);
      prof_region_end("kfac-update/reduce-scatter", prof_sync);

#ifdef LBANN_NVPROF
      prof_region_begin("kfac-update/reduce-scatter-barrier",
                        prof_color,
                        prof_sync);
      CHECK_CUDA(cudaDeviceSynchronize());
      comm.trainer_barrier();
      prof_region_end("kfac-update/reduce-scatter-barrier", prof_sync);
#endif // LBANN_NVPROF

      prof_region_begin("kfac-update/average", prof_color, prof_sync);
      for (auto& block : context.m_blocks) {
        prof_region_begin(("kfac-update/average/" + block->get_name()).c_str(),
                          prof_color,
                          prof_sync);
        block->update_kronecker_average(&comm,
                                        m_kronecker_decay,
                                        m_print_matrix,
                                        m_print_matrix_summary);
        prof_region_end(("kfac-update/average/" + block->get_name()).c_str(),
                        prof_sync);
      }
      prof_region_end("kfac-update/average", prof_sync);

      prof_region_end("kfac-update", prof_sync);
    }

    // Step 2: Model-parallel inverse computation
    prof_region_begin("kfac-inverse", prof_color, prof_sync);
    for (auto& block : context.m_blocks) {
      if (!is_kronecker_update_required ||
          (size_t)comm.get_rank_in_trainer() != block->get_inverse_proc_rank())
        continue;

      prof_region_begin(("kfac-inverse/" + block->get_name()).c_str(),
                        prof_color,
                        prof_sync);
      // TODO: Add kfac_block::is_bn?
      const bool is_bn =
        dynamic_cast<kfac_block_bn<Device>*>(block.get()) != nullptr;
      const bool is_gru =
        dynamic_cast<kfac_block_gru<Device>*>(block.get()) != nullptr;
      block->update_kronecker_inverse(
        &comm,
        m_use_pi,
        is_bn ? context.m_damping_bn_act : context.m_damping_act,
        is_bn ? context.m_damping_bn_err : context.m_damping_err,
        is_gru ? m_learning_rate_factor_gru : m_learning_rate_factor,
        m_print_matrix,
        m_print_matrix_summary,
        m_print_time);
      prof_region_end(("kfac-inverse/" + block->get_name()).c_str(), prof_sync);
    }

    // allgather inverse matrices
    if (is_first_step and false) {
      kfac::allgather_inverse_matrices_sizes(context.m_blocks,
                                             m_inverse_matrices_size,
                                             &comm);
      int block_number = 0;
      for (auto& block : context.m_blocks) {
        block->resize_inverse_matrices_size(m_inverse_matrices_size,
                                            block_number);
        block_number++;
      }
    }

    int global_buffer_inverses_size = 0;

    for (auto& block : context.m_blocks) {
      global_buffer_inverses_size += block->get_inverse_matrices_size(&comm);
    }

    El::Matrix<DataType, Device>& global_buffer_inverse =
      context.get_workspace_matrix("allgather_inverse_recv_buffer",
                                   global_buffer_inverses_size,
                                   1);
    kfac::allgather_inverse_matrices(context.m_blocks,
                                     global_buffer_inverse,
                                     &comm);

    m_has_kronecker_inverse = true;
    prof_region_end("kfac-inverse", prof_sync);

#ifdef LBANN_NVPROF
    prof_region_begin("kfac-inverse-barrier", prof_color, prof_sync);
    CHECK_CUDA(cudaDeviceSynchronize());
    comm.trainer_barrier();
    prof_region_end("kfac-inverse-barrier", prof_sync);
#endif // LBANN_NVPROF
  }
  else {
    // Primary grid does nothing
    // m_has_kronecker_inverse = true;
  }

  if (comm.get_grid_type() == GridType::SECONDARY_GRID or
      comm.get_grid_type() == GridType::PRIMARY_GRID) {
    // send_recv_inverse_matrices(
    //     context,
    //     model.get_comm());
  }

  if (comm.get_KFAC_subgrid_create_two_models() or
      comm.get_grid_type() == GridType::PRIMARY_GRID or
      comm.get_grid_type() == GridType::NO_GRID) {
    if (m_has_kronecker_inverse == true) {
      for (auto& block : context.m_blocks) {
        // std::cout<<"Rank:"<<El::mpi::Rank(comm.get_combined_grid_comm())<<"KInside
        // for\n";
        const bool is_gru =
          dynamic_cast<kfac_block_gru<Device>*>(block.get()) != nullptr;
        block->compute_preconditioned_gradients(
          &comm,
          is_gru ? m_learning_rate_factor_gru : m_learning_rate_factor,
          m_print_matrix,
          m_print_matrix_summary,
          m_print_time);
      }
    }
  }

  if (is_first_step) {
    for (auto& block : context.m_blocks) {
      for (auto& info : block->get_internal_matrix_info()) {
        std::ostringstream oss;
        oss << "K-FAC matrix allocation (rank=" << comm.get_rank_in_trainer()
            << "): " << block->get_name() << " " << std::get<0>(info) << " ("
            << std::get<1>(info) << "x" << std::get<2>(info) << ")"
            << std::endl;
        std::cout << oss.str();
      }
    }
  }
}

} // namespace lbann

template <>
std::unique_ptr<lbann::KFAC>
lbann::make<lbann::KFAC>(google::protobuf::Message const& msg_in)
{
  using AlgoType = lbann::KFAC;
  auto const& params =
    dynamic_cast<lbann_data::TrainingAlgorithm const&>(msg_in);

  lbann_data::KFAC kfac_params;
  LBANN_ASSERT(params.parameters().UnpackTo(&kfac_params));

  // SGD parameters
  auto const& sgd_params = kfac_params.sgd();
  auto const& stopping_criteria = sgd_params.stopping_criteria();
  std::unique_ptr<SGDTerminationCriteria> stopping;
  switch (stopping_criteria.criterion_case()) {
  case lbann_data::SGD::TerminationCriteria::kMaxBatches:
    stopping = std::make_unique<BatchTerminationCriteria>(
      stopping_criteria.max_batches());
    break;
  case lbann_data::SGD::TerminationCriteria::kMaxEpochs:
    stopping = std::make_unique<EpochTerminationCriteria>(
      stopping_criteria.max_epochs());
    break;
  case lbann_data::SGD::TerminationCriteria::kMaxSeconds:
    stopping = std::make_unique<SecondsTerminationCriteria>(
      stopping_criteria.max_seconds());
    // LBANN_ERROR("Time-based training not yet supported in SGD.");
    break;
  default:
    LBANN_ERROR("No stopping criteria specified.");
  }

  const auto parse_damping_params = [](const std::string str) {
    if (str == "")
      return std::vector<double>({AlgoType::damping_0_default});
    else {
      const auto ret = parse_list<double>(str);
      if (ret.size() > 2)
        LBANN_ERROR("The length of damping vectors should be 1 or 2.");
      return ret;
    }
  };

  const auto parse_update_intervals = [](const std::string str) {
    if (str == "")
      return std::vector<size_t>({1});
    else {
      const auto ret = parse_list<size_t>(str);
      if (ret.size() > 2)
        LBANN_ERROR("The length of update interval vectors should be 1 or 2.");
      return ret;
    }
  };

  const std::vector<double> damping_act_params =
    parse_damping_params(kfac_params.damping_act());
  const std::vector<double> damping_err_params =
    parse_damping_params(kfac_params.damping_err());
  const std::vector<double> damping_bn_act_params =
    parse_damping_params(kfac_params.damping_bn_act());
  const std::vector<double> damping_bn_err_params =
    parse_damping_params(kfac_params.damping_bn_err());
  size_t damping_warmup_steps = kfac_params.damping_warmup_steps();
  if (damping_warmup_steps == 0)
    damping_warmup_steps = AlgoType::damping_warmup_steps_default;
  double kronecker_decay = kfac_params.kronecker_decay();
  if (kronecker_decay == 0.0)
    kronecker_decay = AlgoType::kronecker_decay_default;
  const bool print_time = kfac_params.print_time();
  const bool print_matrix = kfac_params.print_matrix();
  const bool print_matrix_summary = kfac_params.print_matrix_summary();
  const bool use_pi = kfac_params.use_pi();
  const std::vector<size_t> update_intervals =
    parse_update_intervals(kfac_params.update_intervals());
  const size_t update_interval_steps = kfac_params.update_interval_steps();
  const size_t compute_interval = El::Max(kfac_params.compute_interval(), 1);

  const std::string inverse_strategy_str = kfac_params.inverse_strategy();
  kfac::kfac_inverse_strategy inverse_strategy;
  if (inverse_strategy_str == "" || inverse_strategy_str == "all")
    inverse_strategy = kfac::kfac_inverse_strategy::ALL;
  else if (inverse_strategy_str == "each")
    inverse_strategy = kfac::kfac_inverse_strategy::EACH;
  else if (inverse_strategy_str == "root")
    inverse_strategy = kfac::kfac_inverse_strategy::ROOT;
  else {
    std::stringstream err;
    err << "Invalid inverse strategy type: " << inverse_strategy_str;
    LBANN_ERROR(err.str());
  }

  const std::vector<std::string> disable_layers =
    parse_list<std::string>(kfac_params.disable_layers());

  double learning_rate_factor = kfac_params.learning_rate_factor();
  double learning_rate_factor_gru = kfac_params.learning_rate_factor_gru();
  if (learning_rate_factor == 0.0)
    learning_rate_factor = 1.0;
  if (learning_rate_factor_gru == 0.0)
    learning_rate_factor_gru = learning_rate_factor;

  return std::make_unique<AlgoType>(params.name(),
                                    std::move(stopping),
                                    std::move(damping_act_params),
                                    std::move(damping_err_params),
                                    std::move(damping_bn_act_params),
                                    std::move(damping_bn_err_params),
                                    damping_warmup_steps,
                                    kronecker_decay,
                                    print_time,
                                    print_matrix,
                                    print_matrix_summary,
                                    use_pi,
                                    std::move(update_intervals),
                                    update_interval_steps,
                                    inverse_strategy,
                                    std::move(disable_layers),
                                    learning_rate_factor,
                                    learning_rate_factor_gru,
                                    compute_interval);
}
