#pragma once

#include <concepts>
#include <cstddef>
#include <optional>  // for optional
#include <utility>   // for pair

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

// trait UpdateByCutChoices<SS> {
//     typename ArrayType;  // double for 1D; ndarray::Arr1 for general
//     auto update_by(SS & ss, const Self::ArrayType& grad)->std::pair<CutStatus, double>;
// };

/**
 * @brief Oracle for feasibility problems (assume convexity)
 * 
 * @tparam Oracle 
 */
template <class Oracle>
concept OracleFeas = requires(Oracle omega, const ArrayType<Oracle>& x) {
    typename Oracle::ArrayType;   // double for 1D; ndarray::Arr1 for general
    typename Oracle::CutChoices;  // double for single cut; (double, Option<double) for parallel cut
    { omega.assess_feas(x) } -> std::convertible_to<std::optional<Cut<Oracle>>>;
};

/**
 * @brief Oracle for optimization problems (assume convexity)
 * 
 * @tparam Oracle 
 */
template <class Oracle>
concept OracleOptim = requires(Oracle omega, const ArrayType<Oracle>& x, double& t) {
    typename Oracle::ArrayType;   // double for 1D; ndarray::Arr1 for general
    typename Oracle::CutChoices;  // double for single cut; (double, Option<double) for parallel cut
    { omega.assess_optim(x, t) } -> std::convertible_to<std::pair<Cut<Oracle>, bool>>;
};

/**
 * @brief Oracle for quantized optimization problems (assume convexity)
 * 
 * @tparam Oracle 
 */
template <class Oracle>
concept OracleQ = requires(Oracle omega, const ArrayType<Oracle>& x, double& t, bool retry) {
    typename Oracle::ArrayType;   // double for 1D; ndarray::Arr1 for general
    typename Oracle::CutChoices;  // double for single cut; (double, Option<double) for parallel cut
    { omega.assess_q(x, t, retry) } -> std::convertible_to<RetQ<Oracle>>;
};

/**
 * @brief Oracle for binary search (assume monotonicity)
 * 
 * @tparam Oracle 
 */
template <class Oracle>
concept OracleBS = requires(Oracle omega, double& t) {
    { omega.assess_bs(t) } -> std::convertible_to<bool>;
};

/**
 * @brief Search space
 * 
 * @tparam Space 
 * @tparam T 
 */
template <class Space, typename T>
concept SearchSpace = requires(Space ss, const std::pair<ArrayType<Space>, T>& cut) {
    typename Space::ArrayType;  // double for 1D; ndarray::Arr1 for general
    { ss.xc() } -> std::convertible_to<ArrayType<Space>>;
    { ss.update(cut) } -> std::convertible_to<std::pair<CutStatus, double>>;
};
