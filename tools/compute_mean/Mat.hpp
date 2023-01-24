#ifndef _TOOLS_MAT_HPP_
#define _TOOLS_MAT_HPP_
#include "lbann/utils/exception.hpp"
#include "lbann_config.hpp"
#include <cstddef> // size_t
#include <memory>
#include <string>
#include <vector>

namespace El {
using Int = long long;

class IR
{
protected:
  Int m_beg;
  Int m_end;

public:
  IR() { Init(); }
  IR(const IR&) = default;
  IR(const Int s, const Int e) { Set(s, e); }
  IR& operator=(const IR&) = default;

  void Init() { Set(0, -1); }
  bool IsInitialized() const { return (m_end >= m_beg); }

  void Set(const Int s, const Int e)
  {
    if (s < 0) {
      throw lbann::lbann_exception("Invalid m_beginning of a range: " +
                                   std::to_string(s));
    }
    m_beg = s;
    m_end = e;
  }

  Int Beg() const { return m_beg; }
  Int End() const { return m_end; }

  IR operator+(const Int d) const { return IR(m_beg + d, m_end + d); }
};

template <typename T>
class ElMatLike
{
protected:
  Int m_width;
  Int m_height;
  IR m_row_range;
  IR m_col_range;

  std::shared_ptr<std::vector<T>> m_buf;

  Int Offset(const Int r, const Int c) const;

public:
  ElMatLike() : m_width(0), m_height(0) {}
  ElMatLike(const ElMatLike&) = default;
  ElMatLike& operator=(const ElMatLike&) = default;
  ElMatLike operator()(const IR& rr, const IR& cr) const;
  T& operator()(const Int r, const Int c) const;

  Int Width() const
  {
    return m_col_range.IsInitialized() ? (m_col_range.End() - m_col_range.Beg())
                                       : m_width;
  }

  Int Height() const
  {
    return m_row_range.IsInitialized() ? (m_row_range.End() - m_row_range.Beg())
                                       : m_height;
  }

  Int LDim() const { return m_height; }

  void Resize(const Int h, const Int w);

  T* Buffer(const Int i = 0, const Int j = 0) const;
  const T* LockedBuffer(const Int i = 0, const Int j = 0) const;

  void Set(const Int r, const Int c, const T d);
  T Get(const Int r, const Int c) const;

  ElMatLike<T>& Copy(const ElMatLike<T>& src);
};

template <typename T>
inline Int ElMatLike<T>::Offset(const Int r, const Int c) const
{
  if (!m_buf || !((r < Height()) && (c < Width()) && (0 <= r) && (0 <= c))) {
    throw lbann::lbann_exception("invalid point : (" + std::to_string(r) + ',' +
                                 std::to_string(c) + ')');
  }

  const Int rv = r + m_row_range.Beg();
  const Int cv = c + m_col_range.Beg();

  return (LDim() * cv + rv);
}

template <typename T>
inline ElMatLike<T> ElMatLike<T>::operator()(const IR& rr, const IR& cr) const
{
  ElMatLike view = *this;
  if ((rr.IsInitialized() && (rr.End() > Height())) ||
      (cr.IsInitialized() && (cr.End() > Width()))) {
    using std::to_string;
    throw lbann::lbann_exception(
      "Invalid range: " + ((rr.End() > Height())
                             ? ("(rows End " + to_string(rr.End()) +
                                " < Height " + to_string(Height()) + ")")
                             : ("(cols End " + to_string(cr.End()) +
                                " < Width " + to_string(Width()) + ")")));
  }

  Int r_beg = m_row_range.IsInitialized() ? m_row_range.Beg() : 0;
  Int c_beg = m_col_range.IsInitialized() ? m_col_range.Beg() : 0;

  view.m_row_range = rr + r_beg;
  view.m_col_range = cr + c_beg;
  return view;
}

template <typename T>
inline T& ElMatLike<T>::operator()(const Int r, const Int c) const
{
  const Int offset = Offset(r, c);
  return (*m_buf)[offset];
}

template <typename T>
inline void ElMatLike<T>::Resize(const Int h, const Int w)
{
  if ((w < 0) || (h < 0)) {
    m_width = 0;
    m_height = 0;
  }
  else {
    m_width = w;
    m_height = h;
  }
  if (!m_buf) {
    m_buf = std::make_shared<std::vector<T>>(m_width * m_height);
  }
  else {
    m_buf->resize(static_cast<size_t>(m_width * m_height));
  }

  m_row_range.Init();
  m_col_range.Init();
}

template <typename T>
inline T* ElMatLike<T>::Buffer(const Int i, const Int j) const
{
  if (!m_buf || (m_buf->size() == 0u)) {
    return nullptr;
  }
  const Int offset = Offset(i, j);
  return &((*m_buf)[offset]);
}

template <typename T>
inline const T* ElMatLike<T>::LockedBuffer(const Int i, const Int j) const
{
  if (!m_buf || (m_buf->size() == 0u)) {
    return nullptr;
  }
  const Int offset = Offset(i, j);
  return &((*m_buf)[offset]);
}

template <typename T>
inline void ElMatLike<T>::Set(const Int r, const Int c, const T d)
{
  const Int offset = Offset(r, c);
  (*m_buf)[offset] = d;
}

template <typename T>
inline T ElMatLike<T>::Get(const Int r, const Int c) const
{
  const Int offset = Offset(r, c);
  return (*m_buf)[offset];
}

template <typename T>
inline ElMatLike<T>& ElMatLike<T>::Copy(const ElMatLike<T>& src)
{
  m_buf = nullptr;
  Resize(src.m_height, src.m_width);
  m_row_range = src.m_row_range;
  m_col_range = src.m_col_range;
  return (*this);
}

template <typename T>
inline void
View(ElMatLike<T>& V, const ElMatLike<T>& X, const IR& r, const IR& c)
{
  V = X(r, c);
}

template <typename T>
inline void Copy(const ElMatLike<T>& S, ElMatLike<T>& D)
{
  D.Copy(S);
}

template <typename T>
using Matrix = ElMatLike<T>;

} // namespace El

using Mat = El::ElMatLike<lbann::DataType>;
using CPUMat = Mat;

#endif // _TOOLS_MAT_HPP_
