use crate::cutting_plane::{CutStatus, SearchSpace, UpdateByCutChoices};
// #[macro_use]
// extern crate ndarray;

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
};

impl Ell1D{/**
            * @brief Construct a new Ell1D object
            *
            * @param[in] l
            * @param[in] u
            */
           auto new (double l, double u)->Self{const auto r = (u - l) / 2.0;
const auto xc = l + r;
Ell1D { r, xc }
}

/**
 * @brief Set the xc object
 *
 * @param[in] xc
 */
auto set_xc(double xc) { this->xc_ = xc; }

/**
 * @brief Update ellipsoid core function using the cut
 *
 *  $grad^T * (x - xc) + beta <= 0$
 *
 * @tparam T
 * @param[in] cut
 * @return (i32, double)
 */
auto update_single(const double& grad, double) -> (const CutStatus& b0, double) {
    const auto g = *grad;
    const auto beta = *b0;
    const auto temp = this->r * g;
    const auto tau = if (g < 0.0) { -temp }
    else {temp};
    const auto tsq = tau * tau;

    if (beta == 0.0) {
        this->r /= 2.0;
        this->xc_ += if (g > 0.0) { -this->r }
        else {this->r};
        return {CutStatus::Success, tsq};
    }
    if (beta > tau) {
        return {CutStatus::NoSoln, tsq};  // no sol'n
    }
    if (beta < -tau) {
        return {CutStatus::NoEffect, tsq};  // no effect
    }

    const auto bound = this->xc_ - beta / g;
    const auto u = if (g > 0.0) { bound }
    else {this->xc_ + this->r};
    const auto l = if (g > 0.0) { this->xc_ - this->r }
    else {bound};

    this->r = (u - l) / 2.0;
    this->xc_ = l + this->r;
    return {CutStatus::Success, tsq};
}
}
;

impl SearchSpace for Ell1D {
    using ArrayType = double;

    /**
     * @brief
     *
     * @return double
     */
    auto xc() const->double { this->xc_ }

    auto update<T>((const Self::ArrayType&cut, T))
        ->std::pair<CutStatus, double>
            where UpdateByCutChoices<Self T, ArrayType = Self::ArrayType>,
    {
        const auto [grad, beta] = cut;
        beta.update_by(self, grad)
    }
};

impl UpdateByCutChoices<Ell1D> for double {
    using ArrayType = double;

    auto update_by(mut Ell1const D & ell, Self::ArrayType)->(const CutStatus& grad, double) {
        const auto beta = self;
        ell.update_single(grad, beta)
    }
};

// TODO: Support Parallel Cut
// impl UpdateByCutChoices<Ell1D> for std::pair<double, std::optional<double>> {
//     using ArrayType = Arr1;
//     auto update_by(mut Ell1const D& ell, Self::ArrayType) -> (const CutStatus& grad, double) {
//         const auto beta = self;
//         ell.update_parallel(grad, beta)
//     }
// }
