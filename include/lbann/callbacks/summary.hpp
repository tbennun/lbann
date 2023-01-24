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
// summary .hpp .cpp - Callback hooks to summarize to Tensorboard
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CALLBACKS_CALLBACK_SUMMARY_HPP_INCLUDED
#define LBANN_CALLBACKS_CALLBACK_SUMMARY_HPP_INCLUDED

#include "lbann/callbacks/callback.hpp"
#include "lbann/utils/summary.hpp"

namespace lbann {
namespace callback {

/**
 * Summarize information to Tensorboard using LBANN's summary interface.
 */
class summary : public callback_base
{
public:
  /**
   * @param summarizer The summary object to write to; this callback takes
   * ownership of it.
   * @param batch_interval The frequency with which to summarize
   * @param mat_interval FIXME
   * @todo Document mat_interval parameter.
   */
  summary(const std::shared_ptr<lbann_summary>& summarizer,
          int batch_interval = 1,
          int mat_interval = 25);
  summary(const summary&) = default;
  summary& operator=(const summary&) = default;
  summary* copy() const override { return new summary(*this); }
  void on_train_begin(model* m) override;
  void on_batch_end(model* m) override;
  void on_epoch_end(model* m) override;
  void on_test_end(model* m) override;
  std::string name() const override { return "summary"; }

protected:
  /** Write out histograms from the model's layers. */
  void save_histograms(model* m);

private:
  /** @brief lbann_summary */
  std::shared_ptr<lbann_summary> m_summarizer = nullptr;

  /** Interval for doing matrix summarization. */
  int m_mat_interval;
};

// Builder function
std::unique_ptr<callback_base>
build_summary_callback_from_pbuf(const google::protobuf::Message&,
                                 std::shared_ptr<lbann_summary> const&);

} // namespace callback
} // namespace lbann

#endif // LBANN_CALLBACKS_CALLBACK_SUMMARY_HPP_INCLUDED
