#include <cmath>                        // for sqrt
#include <ellalgo/ell_config.hpp>       // for CUTStatus, CUTStatus::success
#include <ellalgo/ell.hpp>              // for ell, ell::Arr
#include <ellalgo/ell_calc.hpp>         // for ell, ell::Arr
#include <ellalgo/ell_assert.hpp>       // for ELL_UNLIKELY
#include <tuple>                        // for tuple
#include <xtensor/xarray.hpp>           // for xarray_container
#include <xtensor/xcontainer.hpp>       // for xcontainer
#include <xtensor/xlayout.hpp>          // for layout_type, layout_type::row...
#include <xtensor/xoperation.hpp>       // for xfunction_type_t, operator-
#include <xtensor/xsemantic.hpp>        // for xsemantic_base
#include <xtensor/xtensor_forward.hpp>  // for xarray

/**
 * @brief Construct a new Ell object
 *
 * @tparam V
 * @tparam U
 * @param kappa
 * @param mq
 * @param x
 */
auto Ell::new_with_matrix(double kappa, Arr2 mq, Arr1 xc) -> Ell {
    const auto n = xc.size();
    const auto helper = EllCalc::new(n as double);

    Ell {
        kappa,
        mq,
        xc,
        n,
        helper,
        false no_defer_trick;
    }
}

/**
 * @brief Construct a new Ell object
 *
 * @param[in] val
 * @param[in] x
 */
auto Ell::new(Arr1 val, Arr1 xc) -> Ell {
    Ell::new_with_matrix(1.0, Array2::from_diag(val), xc)
}

/**
 * @brief Update ellipsoid core function using the cut
 *
 *  $grad^T * (x - xc) + beta <= 0$
 *
 * @tparam T
 * @param[in] cut
 * @return (i32, double)
 */
auto Ell::update_single(Arr1& grad, const double& beta) -> std::pair<CutStatus, double> {
    // const auto (grad, beta) = cut;
    auto mq_g = Array1::zeros(this->n); // initial x0
    auto omega = 0.0;
    for (auto i : range(this->n)) {
        for (auto j : range(this->n)) {
            mq_g[i] += this->mq[{i, j}] * grad[j];
        }
        omega += mq_g[i] * grad[i];
    }

    this->helper.tsq = this->kappa * omega;
    const auto status = this->helper.calc_dc(*beta);
    if (status != CutStatus::Success) {
        return {status, this->helper.tsq};
    }

    this->xc -= &((this->helper.rho / omega) * &mq_g); // n

    const auto r = this->helper.sigma / omega;
    for (auto i : range(this->n)) {
        const auto r_mq_g = r * mq_g[i];
        for (auto j : range(i)) {
            this->mq[{i, j}] -= r_mq_g * mq_g[j];
            this->mq[{j, i}] = this->mq[{i, j}];
        }
        this->mq[{i, i}] -= r_mq_g * mq_g[i];
    }

    this->kappa *= this->helper.delta;

    if (this->no_defer_trick) {
        this->mq *= this->kappa;
        this->kappa = 1.0;
    }
    return {status, this->helper.tsq};
}

/**
 * @brief Update ellipsoid core function using the cut
 *
 *  $grad^T * (x - xc) + beta <= 0$
 *
 * @tparam T
 * @param[in] cut
 * @return (i32, double)
 */
auto Ell::update_parallel(Arr1& grad, const std::pair<double, std::optional<double>>& beta) -> std::pair<CutStatus, double> {
    // const auto (grad, beta) = cut;
    auto mq_g = Arr2::zeros(this->n); // initial x0
    auto omega = 0.0;
    for (auto i : range(this->n)) {
        for (auto j : range(this->n)) {
            mq_g[i] += this->mq[{i, j}] * grad[j];
        }
        omega += mq_g[i] * grad[i];
    }

    this->helper.tsq = this->kappa * omega;

    const auto (b0, b1_opt) = *beta;
    const auto status = if (const auto Some(b1) = b1_opt) {
        this->helper.calc_ll_core(b0, b1)
    } else {
        this->helper.calc_dc(b0)
    };
    if (status != CutStatus::Success) {
        return {status, this->helper.tsq};
    }

    this->xc -= &((this->helper.rho / omega) * &mq_g); // n

    // n*(n+1)/2 + n
    // this->mq -= (this->sigma / omega) * xt::linalg::outer(mq_g, mq_g);

    const auto r = this->helper.sigma / omega;
    for (auto i : range(this->n)) {
        const auto r_mq_g = r * mq_g[i];
        for (auto j : range(i)) {
            this->mq[{i, j}] -= r_mq_g * mq_g[j];
            this->mq[{j, i}] = this->mq[{i, j}];
        }
        this->mq[{i, i}] -= r_mq_g * mq_g[i];
    }

    this->kappa *= this->helper.delta;

    if (this->no_defer_trick) {
        this->mq *= this->kappa;
        this->kappa = 1.0;
    }
    return {status, this->helper.tsq};
}
