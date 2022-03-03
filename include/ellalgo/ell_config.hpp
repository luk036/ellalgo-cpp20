#pragma once

#include <utility>  // for pair
#include <cstddef>
#include <concepts>

// using CInfo = (bool, size_t, CutStatus);

struct Options {
    size_t max_iter;
    double tol;
};

/*
// #[derive(Debug, Clone, Copy)]
enum CutChoices {
    Single(double),
    Parallelstd::pair<double, std::optional<double>>,
};
*/
// #[derive(Debug, PartialEq, Eq)]
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

// trait UpdateByCutChoices<SS> {
//     typename ArrayType;  // double for 1D; ndarray::Arr1 for general
//     auto update_by(SS & ss, const Self::ArrayType& grad)->std::pair<CutStatus, double>;
// };

/// Oracle for feasibility problems
template <class Oracle>
concept OracleFeas = requires(Oracle omega, const typename Oracle::ArrayType& x) {
    typename Oracle::ArrayType;   // double for 1D; ndarray::Arr1 for general
    typename Oracle::CutChoices;  // double for single cut; (double, Option<double) for parallel cut
    { omega.assess_feas(x) }
        -> std::convertible_to<std::pair<std::pair<typename Oracle::ArrayType, typename Oracle::CutChoices>, bool>>;
};

/// Oracle for optimization problems
template <class Oracle>
concept OracleOptim = requires(Oracle omega, const typename Oracle::ArrayType& x, double& t) {
    typename Oracle::ArrayType;   // double for 1D; ndarray::Arr1 for general
    typename Oracle::CutChoices;  // double for single cut; (double, Option<double) for parallel cut
    { omega.assess_optim(x, t) } 
        -> std::convertible_to<std::pair<std::pair<typename Oracle::ArrayType, typename Oracle::CutChoices>, bool>>;
};

/// Oracle for quantized optimization problems
template <class Oracle>
concept OracleQ = requires(Oracle omega, const typename Oracle::ArrayType& x, double& t, bool retry) {
    typename Oracle::ArrayType;   // double for 1D; ndarray::Arr1 for general
    typename Oracle::CutChoices;  // double for single cut; (double, Option<double) for parallel cut
    { omega.assess_q(x, t, retry) }
        -> std::convertible_to<std::tuple<std::pair<typename Oracle::ArrayType, typename Oracle::CutChoices>, bool, typename Oracle::ArrayType, bool>>;
};

/// Oracle for binary search
template <class Oracle>
concept OracleBS = requires(Oracle omega, double& t) {
    { omega.assess_bs(t) } -> std::convertible_to<bool>;
};

template <class Space, typename T>
concept SearchSpace = requires(Space ss, const std::pair<typename Space::ArrayType, T>& cut) {
    typename Space::ArrayType;  // double for 1D; ndarray::Arr1 for general
    { ss.xc() } -> std::convertible_to<typename Space::ArrayType>;
    { ss.update(cut) }
        -> std::convertible_to<std::pair<CutStatus, double>>;
};
