#include <array>
#include <cmath> // for std::sqrt()
#include <ellalgo/cutting_plane.hpp>
#include <ellalgo/ell1d.hpp>
#include <ellalgo/ell_calc.hpp>

/**
 * @brief
 *
 * @param[in] b0
 * @param[in] b1
 * @return i32
 */
auto EllCalc::calc_ll_core(double b0, double b1) -> CutStatus {
  const auto b1sqn = b1 * (b1 / this->tsq);
  const auto t1n = 1.0 - b1sqn;
  if (t1n <= 0.0 || !this->use_parallel_cut) {
    return this->calc_dc(b0);
  }
  const auto bdiff = b1 - b0;
  if (bdiff < 0.0) {
    return CutStatus::NoSoln; // no sol'n
  }
  if (b0 == 0.0) {
    // central cut
    this->calc_ll_cc(b1, b1sqn);
    return CutStatus::Success;
  }
  const auto b0b1n = b0 * (b1 / this->tsq);
  if (this->n_float * b0b1n < -1.0) {
    return CutStatus::NoEffect; // no effect
  }
  const auto t0n = 1.0 - b0 * (b0 / this->tsq);
  const auto bsum = b0 + b1;
  const auto bsumn = bsum / this->tsq;
  const auto bav = bsum / 2.0;
  const auto tempn = this->half_n * bsumn * bdiff;
  const auto xi = std::sqrt(t0n * t1n + tempn * tempn);
  this->sigma = this->c3 + (1.0 + b0b1n - xi) / (bsumn * bav) / this->n_plus_1;
  this->rho = this->sigma * bav;
  this->delta = this->c1 * ((t0n + t1n) / 2.0 + xi / this->n_float);
  return CutStatus::Success;
}

/**
 * @brief
 *
 * @param[in] b1
 * @param[in] b1sq
 * @return void
 */
auto EllCalc::calc_ll_cc(double b1, double b1sqn) -> void {
  const auto temp = this->half_n * b1sqn;
  const auto xi = std::sqrt(1.0 - b1sqn + temp * temp);
  this->sigma = this->c3 + this->c2 * (1.0 - xi) / b1sqn;
  this->rho = this->sigma * b1 / 2.0;
  this->delta = this->c1 * (1.0 - b1sqn / 2.0 + xi / this->n_float);
}

/**
 * @brief Deep Cut
 *
 * @param[in] beta
 * @return i32
 */
auto EllCalc::calc_dc(double beta) -> CutStatus {
  const auto tau = std::sqrt(this->tsq);

  const auto bdiff = tau - beta;
  if (bdiff < 0.0) {
    return CutStatus::NoSoln; // no sol'n
  }
  if (beta == 0.0) {
    this->calc_cc(tau);
    return CutStatus::Success;
  }
  const auto gamma = tau + this->n_float * beta;
  if (gamma < 0.0) {
    return CutStatus::NoEffect; // no effect
  }
  // this->mu = (bdiff / gamma) * this->half_n_minus_1;
  this->rho = gamma / this->n_plus_1;
  this->sigma = 2.0 * this->rho / (tau + beta);
  this->delta = this->c1 * (1.0 - beta * (beta / this->tsq));
  return CutStatus::Success;
}

/**
 * @brief Central Cut
 *
 * @param[in] tau
 * @return i32
 */
auto EllCalc::calc_cc(double tau) -> void {
  // this->mu = this->half_n_minus_1;
  this->sigma = this->c2;
  this->rho = tau / this->n_plus_1;
  this->delta = this->c1;
}
