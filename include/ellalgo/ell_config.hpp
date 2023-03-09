#pragma once

#include <cstddef>
// #include <optional> // for optional
#include <utility> // for pair

/**
 * @brief Options
 *
 */
struct Options {
  size_t max_iter;
  double tol;
};

/**
 * @brief Cut Status
 *
 */
enum CutStatus {
  Success,
  NoSoln,
  NoEffect,
  SmallEnough,
};

/**
 * @brief CInfo
 *
 */
struct CInfo {
  bool feasible;
  size_t num_iters;
  CutStatus status;
};

template <typename T> using ArrayType = typename T::ArrayType;
template <typename T> using CutChoices = typename T::CutChoices;
template <typename T> using Cut = std::pair<ArrayType<T>, CutChoices<T>>;
template <typename T> using RetQ = std::tuple<Cut<T>, bool, ArrayType<T>, bool>;

#if __cpp_concepts >= 201907L
#include "ell_concepts.hpp"
#define REQUIRES(x) requires x
#else
#define REQUIRES(x) /* nothing */
#endif
