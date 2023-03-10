#include <cmath>                       // for sqrt
#include <ellalgo/ell.hpp>             // for ell, ell::Arr
#include <ellalgo/ell_calc.hpp>        // for ell, ell::Arr
#include <ellalgo/ell_config.hpp>      // for CutStatus, CutStatus::success
#include <optional>                    // for optional
#include <utility>                     // for pair
#include <xtensor/xarray.hpp>          // for xarray_container
#include <xtensor/xcontainer.hpp>      // for xcontainer
#include <xtensor/xlayout.hpp>         // for layout_type, layout_type::row...
#include <xtensor/xoperation.hpp>      // for xfunction_type_t, operator-
#include <xtensor/xsemantic.hpp>       // for xsemantic_base
#include <xtensor/xtensor_forward.hpp> // for xarray

using Arr1 = xt::xarray<double, xt::layout_type::row_major>;
using Arr2 = xt::xarray<double, xt::layout_type::row_major>;

/**
 * @brief Construct a new Ell object
 *
 * @tparam V
 * @tparam U
 * @param kappa
 * @param mq
 * @param x
 */
Ell::Ell(double kappa, Arr2 mq, Arr1 xc)
    : n{xc.size()}, kappa{kappa}, mq{std::move(mq)}, xc_{std::move(xc)},
      helper(double(n)), no_defer_trick{false} {}

/**
 * @brief Construct a new Ell object
 *
 * @param[in] val
 * @param[in] x
 */
Ell::Ell(Arr1 val, Arr1 xc) : Ell{1.0, xt::diag(val), std::move(xc)} {}

/**
 * @brief Construct a new Ell object
 *
 * @param[in] val
 * @param[in] x
 */
Ell::Ell(double alpha, Arr1 xc)
    : Ell{alpha, xt::eye(xc.size()), std::move(xc)} {}

/**
 * @brief Update ellipsoid core function using the cut
 *
 *  $grad^T * (x - xc) + beta <= 0$
 *
 * @tparam T
 * @param[in] cut
 * @return (i32, double)
 */
auto Ell::update_single(const Arr1 &grad, const double &beta)
    -> std::pair<CutStatus, double> {
  // const auto [grad, beta] = cut;
  auto mq_g = Arr1{xt::zeros<double>({this->n})}; // initial x0
  auto omega = 0.0;
  for (auto i = 0U; i != this->n; ++i) {
    for (auto j = 0U; j != this->n; ++j) {
      mq_g[i] += this->mq[{i, j}] * grad[j];
    }
    omega += mq_g[i] * grad[i];
  }

  this->helper.tsq = this->kappa * omega;
  const auto status = this->helper.calc_dc(beta);
  if (status != CutStatus::Success) {
    return {status, this->helper.tsq};
  }

  this->xc_ -= (this->helper.rho / omega) * mq_g; // n

  const auto r = this->helper.sigma / omega;
  for (auto i = 0U; i != this->n; ++i) {
    const auto r_mq_g = r * mq_g[i];
    for (auto j = 0U; j != i; ++j) {
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
auto Ell::update_parallel(const Arr1 &grad,
                          const std::pair<double, double> &beta)
    -> std::pair<CutStatus, double> {
  // const auto [grad, beta] = cut;
  auto mq_g = Arr1{xt::zeros<double>({this->n})}; // initial x0
  auto omega = 0.0;
  for (auto i = 0U; i != this->n; ++i) {
    for (auto j = 0U; j != this->n; ++j) {
      mq_g[i] += this->mq[{i, j}] * grad[j];
    }
    omega += mq_g[i] * grad[i];
  }

  this->helper.tsq = this->kappa * omega;
  // const auto [b0, b1_opt] = beta;
  // const auto status = b1_opt ? this->helper.calc_ll_core(b0, *b1_opt)
  //                            : this->helper.calc_dc(b0);
  const auto b0 = beta.first;
  const auto b1 = beta.second;
  const auto status = this->helper.calc_ll_core(b0, b1);
  if (status != CutStatus::Success) {
    return {status, this->helper.tsq};
  }

  this->xc_ -= (this->helper.rho / omega) * mq_g; // n

  // n*(n+1)/2 + n
  // this->mq -= (this->sigma / omega) * xt::linalg::outer(mq_g, mq_g);

  const auto r = this->helper.sigma / omega;
  for (auto i = 0U; i != this->n; ++i) {
    const auto r_mq_g = r * mq_g[i];
    for (auto j = 0U; j != i; ++j) {
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
