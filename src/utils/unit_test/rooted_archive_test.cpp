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

#include "MPITestHelpers.hpp"

// #include <lbann/utils/exception.hpp>
#include <h2/patterns/multimethods/SwitchDispatcher.hpp>
#include <lbann/utils/serialize.hpp>

// Use-cases:
// 1. Checkpoint
// 2. Shipping and receiving
//
// 3. Save weights to PyTorch or the like (see save_weights
//    callback). Maybe we could even ship some python code to
//    automatically restore weights from an LBANN archive into a
//    PyTorch weights tensor. A potential pitfall that I see here is
//    that NVPs are not serialized to binary with the weights, so
//    finding weights matrices in an arbitrary binary archive is
//    unlikely to be possible.

struct foo
{
  foo() : foo(0, 0.0) {}
  foo(int x, double y) : x_(x), y_(y) {}
  template <typename ArchiveT>
  void serialize(ArchiveT& ar)
  {
    ar(CEREAL_NVP(x_), CEREAL_NVP(y_));
  }
  template <typename ArchiveT>
  void serialize(lbann::RootedOutputArchiveAdaptor<ArchiveT>& ar)
  {
    ar(CEREAL_NVP(x_), CEREAL_NVP(y_));
  }
  template <typename ArchiveT>
  void serialize(lbann::RootedInputArchiveAdaptor<ArchiveT>& ar)
  {
    ar(CEREAL_NVP(x_));
    ar(CEREAL_NVP(y_));
  }

  int x_;
  double y_;
};

TEST_CASE("Rooted archive adaptor", "[mpi][cereal][archive]")
{
  auto& comm = ::unit_test::utilities::current_world_comm();
  auto const& g = comm.get_trainer_grid();
  std::stringstream ss;

  int x = 13, y = -1;
  float p = 4.12f, q = -1.f;
  foo f(x, 2.), f_restore(0, 0.0);

#ifdef LBANN_HAS_CEREAL_BINARY_ARCHIVES
  SECTION("Binary archives")
  {
    {
      lbann::RootedBinaryOutputArchive ar(ss, g);
      REQUIRE_NOTHROW(ar(x, p));
      REQUIRE_NOTHROW(ar(::cereal::make_nvp("myfoo", f)));
    }
    {
      lbann::RootedBinaryInputArchive ar(ss, g);
      REQUIRE_NOTHROW(ar(y, q));
      REQUIRE_NOTHROW(ar(::cereal::make_nvp("myfoo", f_restore)));
    }
    CHECK(x == y);
    CHECK(p == q);

    CHECK(f.x_ == f_restore.x_);
    CHECK(f.y_ == f_restore.y_);
  }
#endif // LBANN_HAS_CEREAL_BINARY_ARCHIVES
#ifdef LBANN_HAS_CEREAL_XML_ARCHIVES
  SECTION("XML archives")
  {
    {
      lbann::RootedXMLOutputArchive ar(ss, g);
      REQUIRE_NOTHROW(ar(x, ::cereal::make_nvp("myfloat", p)));
      REQUIRE_NOTHROW(ar(::cereal::make_nvp("myfoo", f)));
    }
    {
      lbann::RootedXMLInputArchive ar(ss, g);
      REQUIRE_NOTHROW(ar(y, ::cereal::make_nvp("myfloat", q)));
      REQUIRE_NOTHROW(ar(::cereal::make_nvp("myfoo", f_restore)));
    }
    CHECK(x == y);
    CHECK(p == q);

    CHECK(f.x_ == f_restore.x_);
    CHECK(f.y_ == f_restore.y_);
  }
#endif // LBANN_HAS_CEREAL_XML_ARCHIVES
}

// Matrices

// namespace cereal
// {

// template <typename T, El::Device D>
// void save(RootedOutputArchiveAdaptor<cereal::XMLOutputArchive>& archive,
//           El::Matrix<T, D> const& mat)
// {
// #if 0
//   LBANN_ASSERT(!mat.Viewing());
//   // Forward to the underlying archive on Root.
//   if (archive.am_root())
//     archive.write_to_archive(mat);
// #endif // 0
// }

// template <typename T, El::Device D>
// void load(RootedInputArchiveAdaptor<cereal::XMLInputArchive>& archive,
//           El::Matrix<T, D>& mat)
// {
// #if 0
//   LBANN_ASSERT(!mat.Viewing());
//   // Restore from the underlying archive on Root.
//   if (archive.am_root())
//     archive.read_from_archive(mat);

//   // First broadcast the size information.
//   El::Int height = mat.Height(), width = mat.Width();
//   El::mpi::Broadcast(height, archive.root(), archive.grid().Comm(),
//                      El::SyncInfoFromMatrix(mat));
//   El::mpi::Broadcast(width, archive.root(), archive.grid().Comm(),
//                      El::SyncInfoFromMatrix(mat));
//   if (!archive.am_root)
//     mat.Resize(height, width);

//   // Finally the matrix data.
//   El::Broadcast(mat, archive.grid().Comm(), archive.root());
// #endif // 0
// }

// }// namespace cereal

using MatrixTypes = h2::meta::TL<El::Matrix<float, El::Device::CPU>
#ifdef LBANN_HAS_GPU
                                 ,
                                 El::Matrix<float, El::Device::GPU>
#endif
                                 >;

TEMPLATE_LIST_TEST_CASE("Rooted archive adaptor and matrices",
                        "[mpi][cereal][archive][tdd]",
                        MatrixTypes)
{
  using MatrixType = TestType;
  auto& comm = ::unit_test::utilities::current_world_comm();
  auto const& g = comm.get_trainer_grid();
  std::stringstream ss;
  MatrixType mat(13, 17), mat_restore;

#ifdef LBANN_HAS_CEREAL_BINARY_ARCHIVES
  SECTION("Binary archive")
  {
    {
      lbann::RootedBinaryOutputArchive oarchive(ss, g);
      CHECK_NOTHROW(oarchive(mat));
    }

    {
      lbann::RootedBinaryInputArchive iarchive(ss, g);
      CHECK_NOTHROW(iarchive(mat_restore));
    }

    CHECK(mat.Height() == mat_restore.Height());
    CHECK(mat.Width() == mat_restore.Width());
  }
#endif // LBANN_HAS_CEREAL_BINARY_ARCHIVES

#ifdef LBANN_HAS_CEREAL_XML_ARCHIVES
  SECTION("XML archive")
  {
    {
      lbann::RootedXMLOutputArchive oarchive(ss, g);
      CHECK_NOTHROW(oarchive(mat));
    }

    {
      lbann::RootedXMLInputArchive iarchive(ss, g);
      CHECK_NOTHROW(iarchive(mat_restore));
    }

    CHECK(mat.Height() == mat_restore.Height());
    CHECK(mat.Width() == mat_restore.Width());
  }
#endif // LBANN_HAS_CEREAL_XML_ARCHIVES
}
