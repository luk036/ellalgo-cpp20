#include <cmath>                        // for sqrt
#include <ellalgo/ell_config.hpp>       // for CutStatus, CutStatus::success
#include <ellalgo/ell_stable.hpp>              // for ell, ell::Arr
#include <ellalgo/ell_calc.hpp>         // for ell, ell::Arr
#include <utility>                      // for pair
#include <xtensor/xarray.hpp>           // for xarray_container
#include <xtensor/xcontainer.hpp>       // for xcontainer
#include <xtensor/xlayout.hpp>          // for layout_type, layout_type::row...
#include <xtensor/xoperation.hpp>       // for xfunction_type_t, operator-
#include <xtensor/xsemantic.hpp>        // for xsemantic_base
#include <xtensor/xtensor_forward.hpp>  // for xarray
#include <py2cpp/range.hpp>  // for range
#include <optional> // for optional

using Arr1 = xt::xarray<double, xt::layout_type::row_major>;
using Arr2 = xt::xarray<double, xt::layout_type::row_major>;

/**
 * @brief Construct a new EllStable object
 *
 * @tparam V
 * @tparam U
 * @param kappa
 * @param mq
 * @param x
 */
EllStable::EllStable(double kappa, Arr2 mq, Arr1 xc) :
    n{xc.size()},
    kappa{kappa},
    mq{std::move(mq)},
    xc_{std::move(xc)},
    helper(double(n)),
    no_defer_trick{false}
    {}

/**
 * @brief Construct a new EllStable object
 *
 * @param[in] val
 * @param[in] x
 */
EllStable::EllStable(Arr1 val, Arr1 xc) :
    EllStable{1.0, xt::diag(val), std::move(xc)}
    {}

/**
 * @brief Construct a new EllStable object
 *
 * @param[in] val
 * @param[in] x
 */
EllStable::EllStable(double alpha, Arr1 xc) :
    EllStable{alpha, xt::eye(xc.size()), std::move(xc)}
    {}

/**
 * @brief Update ellipsoid core function using the cut
 *
 *  $grad^T * (x - xc) + beta <= 0$
 *
 * @tparam T
 * @param[in] cut
 * @return (i32, double)
 */
auto EllStable::update_single(const Arr1& grad, const double& beta) -> std::pair<CutStatus, double> {
    auto inv_ml_g = grad; // initial x0
    for (auto i : py::range(this->n)) {
        for (auto j : py::range(i)) {
            this->mq[{i, j}] = this->mq[{j, i}] * inv_ml_g[j];
            // keep for rank-one update
            inv_ml_g[i] -= this->mq[{i, j}];
        }
    }
    // calculate inv(D)*inv(L)*grad: n
    auto inv_md_inv_ml_g = inv_ml_g; // initially
    for (auto i : py::range(this->n)) {
        inv_md_inv_ml_g[i] *= this->mq[{i, i}];
    }
    // calculate omega: n
    auto g_mq_g = inv_md_inv_ml_g; // initially
    auto omega = 0.0; // initially
    for (auto i : py::range(this->n)) {
        g_mq_g[i] *= inv_ml_g[i];
        omega += g_mq_g[i];
    }
    this->helper.tsq = this->kappa * omega;
    const auto status = this->helper.calc_dc(beta);
    if (status != CutStatus::Success) {
        return {status, this->helper.tsq};
    }
    // calculate mq*grad = inv(L')*inv(D)*inv(L)*grad (n-1 )*n/2
    auto mq_g = inv_md_inv_ml_g; // initially
    // for (auto i : py::range((1, this->n).rev())) {
    for (auto i = this->n - 1; i != 0; --i) {
        // backward subsituition
        for (auto j : py::range(i, this->n)) {
            mq_g[i - 1] -= this->mq[{i, j}] * mq_g[j]; // ???
        }
    }
    // calculate xc: n
    this->xc_ -= (this->helper.rho / omega) * mq_g; // n
    // rank-one 3*n + (n-1 update)*n/2
    // const auto r = this->sigma / omega;
    const auto mu = this->helper.sigma / (1.0 - this->helper.sigma);
    auto oldt = omega / mu; // initially
    const auto m = this->n - 1;
    for (auto j : py::range(m)) {
        // p=sqrt(k)*vv[j];
        // const auto p = inv_ml_g[j];
        // const auto mup = mu * p;
        const auto t = oldt + g_mq_g[j];
        // this->mq[{j, j}] /= t; // update invD
        const auto beta2 = inv_md_inv_ml_g[j] / t;
        this->mq[{j, j}] *= oldt / t; // update invD
        for (auto l : py::range((j + 1), this->n)) {
            // v(l) -= p * this->mq(j, l);
            this->mq(j, l) += beta2 * this->mq[{l, j}];
        }
        oldt = t;
    }
    // const auto p = inv_ml_g(n1);
    // const auto mup = mu * p;
    const auto t = oldt + g_mq_g[m];
    this->mq[{m, m}] *= oldt / t; // update invD
    this->kappa *= this->helper.delta;
    // if this->no_defer_trick
    // {
    //     this->mq *= this->kappa;
    //     this->kappa = 1.;
    // }
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
auto EllStable::update_parallel(const Arr1& grad, const std::pair<double, std::optional<double>>& beta) -> std::pair<CutStatus, double> {
    // const auto [grad, beta] = cut;
    auto inv_ml_g = grad; // initial x0
    for (auto i : py::range(this->n)) {
        for (auto j : py::range(i)) {
            this->mq[{i, j}] = this->mq[{j, i}] * inv_ml_g[j];
            // keep for rank-one update
            inv_ml_g[i] -= this->mq[{i, j}];
        }
    }
    // calculate inv(D)*inv(L)*grad: n
    auto inv_md_inv_ml_g = inv_ml_g; // initially
    for (auto i : py::range(this->n)) {
        inv_md_inv_ml_g[i] *= this->mq[{i, i}];
    }
    // calculate omega: n
    auto g_mq_g = inv_md_inv_ml_g; // initially
    auto omega = 0.0; // initially
    for (auto i : py::range(this->n)) {
        g_mq_g[i] *= inv_ml_g[i];
        omega += g_mq_g[i];
    }

    this->helper.tsq = this->kappa * omega;

    const auto [b0, b1_opt] = beta;
    const auto status = b1_opt ?
        this->helper.calc_ll_core(b0, *b1_opt)
    :
        this->helper.calc_dc(b0);
    if (status != CutStatus::Success) {
        return {status, this->helper.tsq};
    }

    // calculate mq*grad = inv(L')*inv(D)*inv(L)*grad (n-1 )*n/2
    auto mq_g = inv_md_inv_ml_g; // initially
    // for (auto i : py::range((1, this->n).rev())) {
    for (auto i = this->n - 1; i != 0; --i) {
        // backward subsituition
        for (auto j : py::range(i, this->n)) {
            mq_g[i - 1] -= this->mq[{i, j}] * mq_g[j]; // ???
        }
    }
    this->xc_ -= (this->helper.rho / omega) * mq_g; // n

    // n*(n+1)/2 + n
    // this->mq -= (this->sigma / omega) * xt::linalg::outer(mq_g, mq_g);
    const auto mu = this->helper.sigma / (1.0 - this->helper.sigma);
    auto oldt = omega / mu; // initially
    const auto m = this->n - 1;
    for (auto j : py::range(m)) {
        // p=sqrt(k)*vv[j];
        // const auto p = inv_ml_g[j];
        // const auto mup = mu * p;
        const auto t = oldt + g_mq_g[j];
        // this->mq[{j, j}] /= t; // update invD
        const auto beta2 = inv_md_inv_ml_g[j] / t;
        this->mq[{j, j}] *= oldt / t; // update invD
        for (auto l : py::range((j + 1), this->n)) {
            // v(l) -= p * this->mq(j, l);
            this->mq(j, l) += beta2 * this->mq[{l, j}];
        }
        oldt = t;
    }
    // const auto p = inv_ml_g(n1);
    // const auto mup = mu * p;
    const auto t = oldt + g_mq_g[m];
    this->mq[{m, m}] *= oldt / t; // update invD
    this->kappa *= this->helper.delta;

    return {status, this->helper.tsq};
}
