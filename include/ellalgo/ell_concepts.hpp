#pragma once

#include <concepts>
#include <cstddef>
#include <utility> // for pair

// trait UpdateByCutChoices<SS> {
//     typename ArrayType;  // double for 1D; ndarray::Arr1 for general
//     auto update_by(SS & ss, const Self::ArrayType&
//     grad)->std::pair<CutStatus, double>;
// };

/**
 * @brief Oracle for feasibility problems (assume convexity)
 *
 * @tparam Oracle
 */
template <class Oracle>
concept OracleFeas =
    requires(Oracle omega, const ArrayType<Oracle> &x) {
      typename Oracle::ArrayType;  // double for 1D; ndarray::Arr1 for general
      typename Oracle::CutChoices; // double for single cut; (double,
                                   // Option<double) for parallel cut
      // { omega.assess_feas(x) } ->
      // std::convertible_to<std::optional<Cut<Oracle>>>;
      { omega.assess_feas(x) };
    };

/**
 * @brief Oracle for optimization problems (assume convexity)
 *
 * @tparam Oracle
 */
template <class Oracle>
concept OracleOptim =
    requires(Oracle omega, const ArrayType<Oracle> &x, double &t) {
      typename Oracle::ArrayType;  // double for 1D; ndarray::Arr1 for general
      typename Oracle::CutChoices; // double for single cut; (double,
                                   // Option<double) for parallel cut
      {
        omega.assess_optim(x, t)
        } -> std::convertible_to<std::pair<Cut<Oracle>, bool>>;
    };

/**
 * @brief Oracle for quantized optimization problems (assume convexity)
 *
 * @tparam Oracle
 */
template <class Oracle>
concept OracleQ =
    requires(Oracle omega, const ArrayType<Oracle> &x, double &t, bool retry) {
      typename Oracle::ArrayType;  // double for 1D; ndarray::Arr1 for general
      typename Oracle::CutChoices; // double for single cut; (double,
                                   // Option<double) for parallel cut
      { omega.assess_q(x, t, retry) } -> std::convertible_to<RetQ<Oracle>>;
    };

/**
 * @brief Oracle for binary search (assume monotonicity)
 *
 * @tparam Oracle
 */
template <class Oracle>
concept OracleBS = requires(Oracle omega, double &t) {
                     { omega.assess_bs(t) } -> std::convertible_to<bool>;
                   };

/**
 * @brief Search space
 *
 * @tparam Space
 * @tparam T
 */
template <class Space, typename T>
concept SearchSpace =
    requires(Space ss, const std::pair<ArrayType<Space>, T> &cut) {
      typename Space::ArrayType; // double for 1D; ndarray::Arr1 for general
      { ss.xc() } -> std::convertible_to<ArrayType<Space>>;
      { ss.update(cut) } -> std::convertible_to<std::pair<CutStatus, double>>;
    };
