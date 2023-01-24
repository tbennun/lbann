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

#include "lbann/proto/factories.hpp"

#include "lbann/optimizers/optimizer.hpp"

#include "lbann/optimizers/adagrad.hpp"
#include "lbann/optimizers/adam.hpp"
#include "lbann/optimizers/hypergradient_adam.hpp"
#include "lbann/optimizers/rmsprop.hpp"
#include "lbann/optimizers/sgd.hpp"

#include "lbann/utils/factory.hpp"
#include "lbann/utils/protobuf.hpp"

#include <optimizers.pb.h>

namespace {

template <typename T>
std::unique_ptr<lbann::optimizer>
build_no_optimizer_from_pbuf(google::protobuf::Message const& msg)
{
  return nullptr;
}

using factory_type = lbann::generic_factory<
  lbann::optimizer,
  std::string,
  lbann::generate_builder_type<lbann::optimizer,
                               google::protobuf::Message const&>,
  lbann::default_key_error_policy>;

// Manage a global factory
template <typename T>
struct factory_manager
{
  factory_type factory_;

  factory_manager() { register_default_builders(); }

private:
  void register_default_builders()
  {
    factory_.register_builder("NoOptimizer", build_no_optimizer_from_pbuf<T>);
    factory_.register_builder("AdaGrad",
                              lbann::build_adagrad_optimizer_from_pbuf<T>);
    factory_.register_builder("Adam", lbann::build_adam_optimizer_from_pbuf<T>);
    factory_.register_builder(
      "HypergradientAdam",
      lbann::build_hypergradient_adam_optimizer_from_pbuf<T>);
    factory_.register_builder("RMSprop",
                              lbann::build_rmsprop_optimizer_from_pbuf<T>);
    factory_.register_builder("SGD", lbann::build_sgd_optimizer_from_pbuf<T>);
  }
};

template <typename T>
factory_type const& get_optimizer_factory() noexcept
{
  static factory_manager<T> factory_mgr_;
  return factory_mgr_.factory_;
}

} // namespace

template <typename TensorDataType>
std::unique_ptr<lbann::optimizer>
lbann::proto::construct_optimizer(const lbann_data::Optimizer& proto_opt)
{
  auto const& factory = get_optimizer_factory<TensorDataType>();
  auto const& msg = protobuf::get_oneof_message(proto_opt, "optimizer_type");
  return factory.create_object(msg.GetDescriptor()->name(), msg);
}

#define PROTO(T)                                                               \
  template std::unique_ptr<lbann::optimizer>                                   \
  lbann::proto::construct_optimizer<T>(lbann_data::Optimizer const&)
#define LBANN_INSTANTIATE_CPU_HALF
#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"
