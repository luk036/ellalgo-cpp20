#pragma once

#include <array>

#include "ell_config.hpp"

/**
 * @brief Ellipsoid Search Space
 *
 *  EllCalc = {x | (x - xc)^T mq^-1 (x - xc) \le \kappa}
 *
 * Keep $mq$ symmetric but no promise of positive definite
 */
// #[derive(Debug, Clone)]
class EllCalc {
  private:
    double n_float;
    double n_plus_1;
    double half_n;
    double n_sq;
    double c1;
    double c2;
    double c3;

  public:
    double rho;
    double sigma;
    double delta;
    double tsq;
    bool use_parallel_cut;

    /**
     * @brief Construct a new EllCalc object
     *
     * @tparam V
     * @tparam U
     * @param kappa
     * @param mq
     * @param x
     */
    EllCalc(double n_float)
        : n_plus_1{n_float + 1.0},
          half_n{n_float / 2.0},
          n_sq{n_float * n_float},
          c1{n_sq / (n_sq - 1.0)},
          c2{2.0 / n_plus_1},
          c3{n_float / n_plus_1},
          rho{0.0},
          sigma{0.0},
          delta{0.0},
          tsq{0.0},
          use_parallel_cut{true} {}

    // auto update_cut(double) -> CutStatus { this->calc_dc(beta beta) }

    /**
     * @brief
     *
     * @param[in] b0
     * @param[in] b1
     * @return i32
     */
    auto calc_ll_core(double b0, double b1) -> CutStatus;

    /**
     * @brief
     *
     * @param[in] b1
     * @param[in] b1sq
     * @return void
     */
    auto calc_ll_cc(double b1, double b1sqn) -> void;

    /**
     * @brief Deep Cut
     *
     * @param[in] beta
     * @return i32
     */
    auto calc_dc(double beta) -> CutStatus;

    /**
     * @brief Central Cut
     *
     * @param[in] tau
     * @return i32
     */
    auto calc_cc(double tau) -> void;

    /**
     * @brief Get the results object
     *
     * @return std::array<double, 4>
     */
    auto get_results() const -> std::array<double, 4> {
        return {this->rho, this->sigma, this->delta, this->tsq};
    }
};

// trait UpdateByCutChoices {
//     auto update_by(self, EllCalc ell) -> CutStatus;
// }
