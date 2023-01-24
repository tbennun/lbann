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
// export_onnx .hpp .cpp - Exports trained model to onnx format
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CALLBACKS_EXPORT_ONNX_HPP_INCLUDED
#define LBANN_CALLBACKS_EXPORT_ONNX_HPP_INCLUDED

#include "lbann/callbacks/callback.hpp"
#include <lbann/base.hpp>

#include <onnx/onnx_pb.h>

#include <google/protobuf/message.h>

#include <iostream>
#include <memory>
#include <vector>

namespace lbann {
namespace callback {

/** @class export_onnx
 *  @brief Callback to export a trained model to onnx format
 */
class export_onnx : public callback_base
{

public:
  /** @brief export_onnx Constructor.
   *  @param output_filename Output filename (default = lbann.onnx)
   *  @param debug_string_filename Name of file to which debug string is
   *  printed. If not set, the debug string is not output.
   */
  export_onnx(std::string output_filename, std::string debug_string_filename)
    : callback_base(/*batch_interval=*/1),
      m_output_filename{output_filename.size() ? std::move(output_filename)
                                               : std::string("lbann.onnx")},
      m_debug_string_filename{std::move(debug_string_filename)}
  {}

  /** @brief Copy interface */
  export_onnx* copy() const override { return new export_onnx(*this); }

  /** @brief Return name of callback */
  std::string name() const override { return "export_onnx"; }

  /* @brief gather graph/layer info */
  void on_train_end(model* m) override;

private:
  /* @brief name of output file. Default = lbann.onnx */
  std::string m_output_filename;

  /* @brief option to print onnx debug file. Default = none */
  std::string m_debug_string_filename;

}; // class export_onnx

std::unique_ptr<callback_base>
build_export_onnx_callback_from_pbuf(const google::protobuf::Message& proto_msg,
                                     const std::shared_ptr<lbann_summary>&);

} // namespace callback
} // namespace lbann

#endif // LBANN_CALLBACKS_EXPORT_ONNX_HPP_INCLUDED
