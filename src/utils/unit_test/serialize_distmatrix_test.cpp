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
#include "Catch2BasicSupport.hpp"
#include <lbann/base.hpp>

#include <h2/patterns/multimethods/SwitchDispatcher.hpp>
#include <lbann/utils/serialize.hpp>

#include "MPITestHelpers.hpp"

// Enumerate all DistMatrix types. Start by getting all the
// distributions.
template <typename T, El::Device D>
using DistMatrixTypesWithDevice = h2::meta::TL<
  // There is currently a known bug where copying
  // (CIRC,CIRC,CPU)->(CIRC,CIRC,GPU) results in an infinite recursion
  // in Hydrogen. Since we don't actually use this in our code, ignore
  // this case for now.
  //
  // El::DistMatrix<T, El::CIRC, El::CIRC, El::ELEMENT, D>,
  El::DistMatrix<T, El::MC, El::MR, El::ELEMENT, D>,
  El::DistMatrix<T, El::MC, El::STAR, El::ELEMENT, D>,
  El::DistMatrix<T, El::MD, El::STAR, El::ELEMENT, D>,
  El::DistMatrix<T, El::MR, El::MC, El::ELEMENT, D>,
  El::DistMatrix<T, El::MR, El::STAR, El::ELEMENT, D>,
  El::DistMatrix<T, El::STAR, El::MC, El::ELEMENT, D>,
  El::DistMatrix<T, El::STAR, El::MD, El::ELEMENT, D>,
  El::DistMatrix<T, El::STAR, El::MR, El::ELEMENT, D>,
  El::DistMatrix<T, El::STAR, El::STAR, El::ELEMENT, D>,
  El::DistMatrix<T, El::STAR, El::VC, El::ELEMENT, D>,
  El::DistMatrix<T, El::STAR, El::VR, El::ELEMENT, D>,
  El::DistMatrix<T, El::VC, El::STAR, El::ELEMENT, D>,
  El::DistMatrix<T, El::VR, El::STAR, El::ELEMENT, D>>;

template <typename T>
using DistMatrixTypes =
#if defined LBANN_HAS_GPU
  h2::meta::tlist::Append<DistMatrixTypesWithDevice<T, El::Device::CPU>,
                          DistMatrixTypesWithDevice<T, El::Device::GPU>>;
#else
  DistMatrixTypesWithDevice<T, El::Device::CPU>;
#endif // defined LBANN_HAS_GPU

// Finally, enumerate all data types.
using AllDistMatrixTypes =
  h2::meta::tlist::Append<DistMatrixTypes<float>, DistMatrixTypes<double>>;

TEMPLATE_LIST_TEST_CASE("DistMatrix serialization",
                        "[serialize][utils][distmatrix][mpi]",
                        AllDistMatrixTypes)
{
  using DistMatType = TestType;

  // Setup the grid stack
  auto& comm = ::unit_test::utilities::current_world_comm();
  lbann::utils::grid_manager mgr(comm.get_trainer_grid());

  std::stringstream ss;
  DistMatType mat(12, 16, lbann::utils::get_current_grid()),
    mat_restore(lbann::utils::get_current_grid());

#ifdef LBANN_HAS_CEREAL_BINARY_ARCHIVES
  SECTION("Binary archive")
  {
    El::MakeUniform(mat);

    {
      cereal::BinaryOutputArchive oarchive(ss);
      REQUIRE_NOTHROW(oarchive(mat));
    }
    {
      cereal::BinaryInputArchive iarchive(ss);
      REQUIRE_NOTHROW(iarchive(mat_restore));
    }

    REQUIRE(mat.Height() == mat_restore.Height());
    REQUIRE(mat.Width() == mat_restore.Width());
    for (El::Int col = 0; col < mat.LocalWidth(); ++col) {
      for (El::Int row = 0; row < mat.LocalHeight(); ++row) {
        INFO("(Row,Col) = (" << row << "," << col << ")");
        CHECK(mat.GetLocal(row, col) == mat_restore.GetLocal(row, col));
      }
    }
  }
#endif // LBANN_HAS_CEREAL_BINARY_ARCHIVES

#ifdef LBANN_HAS_CEREAL_XML_ARCHIVES
  SECTION("XML archive")
  {
    {
      cereal::XMLOutputArchive oarchive(ss);
      REQUIRE_NOTHROW(oarchive(mat));
    }
    {
      cereal::XMLInputArchive iarchive(ss);
      REQUIRE_NOTHROW(iarchive(mat_restore));
    }

    CHECK(mat.Height() == mat_restore.Height());
    CHECK(mat.Width() == mat_restore.Width());
  }
#endif // LBANN_HAS_CEREAL_BINARY_ARCHIVES
}

// Just a bit of sugar to make the output clearer when testing for
// null pointers.
using check_valid_ptr = bool;

TEMPLATE_LIST_TEST_CASE("DistMatrix serialization with smart pointers",
                        "[serialize][utils][distmatrix][mpi][smartptr]",
                        AllDistMatrixTypes)
{
  using DistMatType = TestType;
  using AbsDistMatType = typename TestType::absType;

  // Setup the grid stack
  auto& comm = ::unit_test::utilities::current_world_comm();
  lbann::utils::grid_manager mgr(comm.get_trainer_grid());

  std::stringstream ss;
  std::unique_ptr<AbsDistMatType> mat, mat_restore;
  mat = std::make_unique<DistMatType>(12, 16, lbann::utils::get_current_grid());

#ifdef LBANN_HAS_CEREAL_BINARY_ARCHIVES
  SECTION("Binary archive")
  {
    El::MakeUniform(*mat);

    {
      cereal::BinaryOutputArchive oarchive(ss);
      REQUIRE_NOTHROW(oarchive(mat));
    }
    {
      cereal::BinaryInputArchive iarchive(ss);
      REQUIRE_NOTHROW(iarchive(mat_restore));
    }

    REQUIRE(mat->Height() == mat_restore->Height());
    REQUIRE(mat->Width() == mat_restore->Width());
    for (El::Int col = 0; col < mat->LocalWidth(); ++col) {
      for (El::Int row = 0; row < mat->LocalHeight(); ++row) {
        INFO("(Row,Col) = (" << row << "," << col << ")");
        CHECK(mat->GetLocal(row, col) == mat_restore->GetLocal(row, col));
      }
    }
  }
#endif // LBANN_HAS_CEREAL_BINARY_ARCHIVES

#ifdef LBANN_HAS_CEREAL_XML_ARCHIVES
  SECTION("XML archive")
  {
    {
      cereal::XMLOutputArchive oarchive(ss);
      REQUIRE_NOTHROW(oarchive(mat));
    }
    {
      cereal::XMLInputArchive iarchive(ss);
      REQUIRE_NOTHROW(iarchive(mat_restore));
    }

    REQUIRE((check_valid_ptr)mat_restore);
    CHECK(mat->Height() == mat_restore->Height());
    CHECK(mat->Width() == mat_restore->Width());
  }
#endif // LBANN_HAS_CEREAL_BINARY_ARCHIVES
}
