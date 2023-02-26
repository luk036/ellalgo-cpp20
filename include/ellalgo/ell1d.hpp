#pragma once

#include <ellalgo/ell_config.hpp> // for CutStatus, CutStatus::success

/**
 * @brief Ellipsoid Method for special 1D case
 *
 *  Ell = {x | (x - xc)^T mq^-1 (x - xc) \le \kappa}
 *
 * Keep $mq$ symmetric but no promise of positive definite
 */
// #[derive(Debug, Clone)]
class Ell1D {
  double r;
  double xc_;

public:
  using ArrayType = double;

  /**
   * @brief Construct a new Ell1D object
   *
   * @param[in] l
   * @param[in] u
   */
  Ell1D(double l, double u) : r{(u - l) / 2.0}, xc_{l + r} {}

  /**
   * @brief Set the xc object
   *
   * @param[in] xc
   */
  auto set_xc(double xc) { this->xc_ = xc; }

  /**
   * @brief Update ellipsoid using the single cut
   *
   * @param[in] grad
   * @param[in] b0
   * @return std::pair<CutStatus, double>
   */
  auto update_single(const double &grad, const double &b0)
      -> std::pair<CutStatus, double> {
    const auto g = grad;
    const auto beta = b0;
    const auto temp = this->r * g;
    const auto tau = (g < 0.0) ? -temp : temp;
    const auto tsq = tau * tau;

    if (beta == 0.0) {
      this->r /= 2.0;
      this->xc_ += (g > 0.0) ? -this->r : this->r;
      return {CutStatus::Success, tsq};
    }
    if (beta > tau) {
      return {CutStatus::NoSoln, tsq}; // no sol'n
    }
    if (beta < -tau) {
      return {CutStatus::NoEffect, tsq}; // no effect
    }

    const auto bound = this->xc_ - beta / g;
    const auto u = (g > 0.0) ? bound : this->xc_ + this->r;
    const auto l = (g > 0.0) ? this->xc_ - this->r : bound;

    this->r = (u - l) / 2.0;
    this->xc_ = l + this->r;
    return {CutStatus::Success, tsq};
  }

  /**
   * @brief
   *
   * @return double
   */
  auto xc() const -> double { return this->xc_; }

  /**
   * @brief
   *
   * @param[in] cut
   * @return std::pair<CutStatus, double>
   */
  auto update(const std::pair<ArrayType, double> &cut)
      -> std::pair<CutStatus, double> {
    const auto [grad, beta] = cut;
    return this->update_single(grad, beta);
  }
};

// TODO: Support Parallel Cut
// impl UpdateByCutChoices<Ell1D> for std::pair<double, std::optional<double>> {
//     using ArrayType = Arr1;
//     auto update_by(mut Ell1const D& ell, Self::ArrayType) -> (const
//     CutStatus& grad, double) {
//         const auto beta = self;
//         ell.update_parallel(grad, beta)
//     }
// }
