#pragma once

#include <xtensor/xarray.hpp>  // for ndarray
#include <optional> // for std::optional
#include "ell_calc.hpp"    // for EllCalc
#include "ell_config.hpp"  // for CutStatus, SearchSpace, UpdateByCutChoices

using Arr1 = xt::xarray<double, xt::layout_type::row_major>;
using Arr2 = xt::xarray<double, xt::layout_type::row_major>;

/**
 * @brief Ellipsoid Search Space
 *
 *  Ell = {x | (x - xc)^T mq^-1 (x - xc) \le \kappa}
 *
 * Keep $mq$ symmetric but no promise of positive definite
 */
// #[derive(Debug, Clone)]
class Ell {
    using Self = Ell;
    using Parallel = std::pair<double, std::optional<double>>;

    Arr2 mq;
    Arr1 xc;
    double kappa;
    size_t n;
    EllCalc helper;

  public:
    bool no_defer_trick;

    using ArrayType = Arr1;

    Ell(double kappa, Arr2 mq, Arr1 xc);

    Ell(Arr1 val, Arr1 xc);

    auto update_single(const Arr1& grad, const double& beta) -> std::pair<CutStatus, double>;

    auto update_parallel(const Arr1& grad, const Parallel& beta) -> std::pair<CutStatus, double>;

    /**
     * @brief copy the whole array anyway
     *
     * @return Arr1
     */
    auto xc() const -> Self::ArrayType { return this->xc; }

    template <typename T> auto update(const std::pair<Self::ArrayType, T>& cut)
        -> std::pair<CutStatus, double> {
        const auto(grad, beta) = cut;
        if constexpr (std::is_same_t<T, double>) {
            return this->update_single(grad, beta);
        } else if constexpr (std::is_same_t<T, Parallel>) {
            return this->update_parallel(grad, beta);
        } else {
            static_assert(false, "Not supported type");
            return {CutStatus::NoSoln, 0.0};
        }
    }
};
