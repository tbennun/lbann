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
// dump_gradients .hpp .cpp - Callbacks to dump gradients
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CALLBACKS_CALLBACK_DUMP_GRADIENTS_HPP_INCLUDED
#define LBANN_CALLBACKS_CALLBACK_DUMP_GRADIENTS_HPP_INCLUDED

#include <utility>

#include "lbann/callbacks/callback.hpp"

namespace lbann {
namespace callback {

/**
 * @brief Dump gradient matrices to files.
 * @details This will dump each hidden layer's gradient matrix after
 * each minibatch.  The matrices are written to files using
 * Elemental's simple ASCII format. This is not meant for
 * checkpointing, but for exporting gradient matrices for analysis
 * that isn't easily done in LBANN.  Note this dumps matrices during
 * each mini-batch. This will be slow and produce a lot of output.
 */
class dump_gradients : public callback_base
{
public:
  using callback_base::on_backward_prop_end;

  /**
   * @param basename The basename for writing files.
   * @param batch_interval The frequency at which to dump the gradients
   */
  dump_gradients(std::string basename, int batch_interval = 1)
    : callback_base(batch_interval), m_basename(std::move(basename))
  {}
  dump_gradients(const dump_gradients&) = default;
  dump_gradients& operator=(const dump_gradients&) = default;
  dump_gradients* copy() const override { return new dump_gradients(*this); }
  void on_backward_prop_end(model* m) override;
  std::string name() const override { return "dump gradients"; }

  /** @name Serialization */
  ///@{

  /** @brief Store state to archive for checkpoint and restart */
  template <class Archive>
  void serialize(Archive& ar);

  ///@}

private:
  friend class cereal::access;
  dump_gradients();

  /** @brief Basename for writing files. */
  std::string m_basename;
};

// Builder function
std::unique_ptr<callback_base>
build_dump_gradients_callback_from_pbuf(const google::protobuf::Message&,
                                        std::shared_ptr<lbann_summary> const&);

} // namespace callback
} // namespace lbann

#endif // LBANN_CALLBACKS_CALLBACK_DUMP_GRADIENTS_HPP_INCLUDED
