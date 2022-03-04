#pragma once

#include <optional>            // for std::optional
#include <xtensor/xarray.hpp>  // for ndarray

#include "ell_calc.hpp"    // for EllCalc
#include "ell_config.hpp"  // for CutStatus, SearchSpace, UpdateByCutChoices

using Arr1 = xt::xarray<double, xt::layout_type::row_major>;
using Arr2 = xt::xarray<double, xt::layout_type::row_major>;

/**
 * @brief Ellipsoid Search Space
 *
 *  EllStable = {x | (x - xc)^T mq^-1 (x - xc) \le \kappa}
 *
 * Keep $mq$ symmetric but no promise of positive definite
 */
// #[derive(Debug, Clone)]
class EllStable {
    using Self = EllStable;
    using Parallel = std::pair<double, std::optional<double>>;

    size_t n;
    double kappa;
    Arr2 mq;
    Arr1 xc_;
    EllCalc helper;

  public:
    using ArrayType = Arr1;

    bool no_defer_trick;

    EllStable(double kappa, Arr2 mq, Arr1 xc);

    EllStable(Arr1 val, Arr1 xc);

    EllStable(double alpha, Arr1 xc);

    auto update_single(const Arr1& grad, const double& beta) -> std::pair<CutStatus, double>;

    auto update_parallel(const Arr1& grad, const Parallel& beta) -> std::pair<CutStatus, double>;

    /**
     * @brief copy the whole array anyway
     *
     * @return Arr1
     */
    auto xc() const -> Self::ArrayType { return this->xc_; }

    template <typename T> auto update(const std::pair<Self::ArrayType, T>& cut)
        -> std::pair<CutStatus, double> {
        const auto [grad, beta] = cut;
        if constexpr (std::is_same_v<T, double>) {
            return this->update_single(grad, beta);
        } else if constexpr (std::is_same_v<T, Parallel>) {
            return this->update_parallel(grad, beta);
        } else {
            // static_assert(false, "Not supported type");
            return {CutStatus::NoSoln, 0.0};
        }
    }
};
