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

#ifndef LBANN_CALLBACKS_CALLBACK_LTFB_HPP_INCLUDED
#define LBANN_CALLBACKS_CALLBACK_LTFB_HPP_INCLUDED

#include "lbann/callbacks/callback.hpp"
#include <memory>
#include <set>
#include <vector>

namespace lbann {
namespace callback {

// Forward declaration
class LTFBCommunicationAlgorithm;

/** @brief Tournament training.
 *
 *  This is intended to support research into the LTFB algorithm. An
 *  outline:
 *    - Divide the computational resources into multiple "trainers"
 *      that can operate in parallel.
 *    - Setup a model on each trainer and begin training independently.
 *    - Periodically launch tournaments to select "good" models. More
 *      specifically, trainers partner up and exchange their models.
 *      Each trainer evaluates a metric for its local and partner
 *      models, using its validation data set. The model with the better
 *      score is retained and the other one is discarded.
 *
 *  There are many algorithmic variations to be explored:
 *    - How is data is divvied up amongst the trainers. Is it strictly
 *      partitioned, partially shared, or completely replicated?
 *    - What model components are exchanged? Just the trainable weights,
 *      or a subset of the weights? Hyperparameters?
 *    - Can this be used to explore model architectures?
 *
 */
class ltfb : public callback_base
{
public:
  /** @brief Construct the LTFB callback
   *  @param batch_interval Number of training mini-batch steps between
   *                        tournaments.
   *  @param metric_name    Metric for tournament evaluation.
   *  @param comm_algo      Inter-trainer communication scheme.
   *  @param exchange_hyperparameters Whether to exchange hyperparameters
   *                                  with model information.
   */
  ltfb(El::Int batch_interval,
       std::string metric_name,
       std::unique_ptr<LTFBCommunicationAlgorithm> comm_algo,
       bool exchange_hyperparameters = false);
  ltfb(const ltfb& other);
  ltfb& operator=(const ltfb& other);
  ltfb* copy() const override { return new ltfb(*this); }
  std::string name() const override { return "LTFB"; }

  void on_train_begin(model* m) override;
  void on_batch_begin(model* m) override;

private:
  /** @brief Metric for tournament evaluation. */
  std::string m_metric_name;

  /** @brief Communication algorithm for exchanging models */
  std::unique_ptr<LTFBCommunicationAlgorithm> comm_algo_;

  /** @brief Whether low-scoring or high-scoring models survive a
   *  tournament. */
  bool m_low_score_wins;
};

// Builder function
std::unique_ptr<callback_base>
build_ltfb_callback_from_pbuf(const google::protobuf::Message&,
                              std::shared_ptr<lbann_summary> const&);

} // namespace callback
} // namespace lbann

#endif // LBANN_CALLBACKS_CALLBACK_LTFB_HPP_INCLUDED
