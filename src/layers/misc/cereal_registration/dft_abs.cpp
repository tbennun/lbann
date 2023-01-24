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
#include "lbann/utils/serialize.hpp"
#include <lbann/layers/misc/dft_abs.hpp>

namespace lbann {

template <typename TensorDataType, El::Device Device>
template <typename ArchiveT>
void dft_abs_layer<TensorDataType, Device>::serialize(ArchiveT& ar)
{
  using DataTypeLayer = data_type_layer<TensorDataType>;
  ar(::cereal::make_nvp("DataTypeLayer",
                        ::cereal::base_class<DataTypeLayer>(this)));
}

} // namespace lbann

// Manually register the DFT ABS layer since it has many permutations
// of supported data and device types
#include <lbann/macros/common_cereal_registration.hpp>
#define LBANN_COMMA ,
#define PROTO_DEVICE(TYPE, DEVICE)                                             \
  LBANN_ADD_ALL_SERIALIZE_ETI(                                                 \
    ::lbann::dft_abs_layer<TYPE LBANN_COMMA DEVICE>);                          \
  CEREAL_REGISTER_TYPE_WITH_NAME(                                              \
    ::lbann::dft_abs_layer<TYPE LBANN_COMMA DEVICE>,                           \
    "dft_abs_layer (" #TYPE "," #DEVICE ")");

#ifdef LBANN_HAS_FFTW
#ifdef LBANN_HAS_FFTW_FLOAT
PROTO_DEVICE(float, El::Device::CPU)
#endif // LBANN_HAS_FFTW_FLOAT
#ifdef LBANN_HAS_FFTW_DOUBLE
PROTO_DEVICE(double, El::Device::CPU)
#endif // LBANN_HAS_FFTW_DOUBLE
#ifdef LBANN_HAS_GPU
PROTO_DEVICE(float, El::Device::GPU)
PROTO_DEVICE(double, El::Device::GPU)
#endif // LBANN_HAS_GPU
#endif // LBANN_HAS_FFTW

LBANN_REGISTER_DYNAMIC_INIT(dft_abs_layer);
