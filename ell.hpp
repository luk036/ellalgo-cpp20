// mod cutting_plane;
use crate::cutting_plane::{CutStatus, SearchSpace, UpdateByCutChoices};
use crate::ell_calc::EllCalc;
// #[macro_use]
// extern crate ndarray;
use ndarray::prelude::*;

using Arr = Array1<double>;

/**
 * @brief Ellipsoid Search Space
 *
 *  Ell = {x | (x - xc)^T mq^-1 (x - xc) \le \kappa}
 *
 * Keep $mq$ symmetric but no promise of positive definite
 */
#[derive(Debug, Clone)]
class Ell {
    pub bool no_defer_trick;

    Array2<double> mq;
    Array1<double> xc;
    double kappa;
    size_t n;
    EllCalc helper;
};

impl Ell {
    /**
     * @brief Construct a new Ell object
     *
     * @tparam V
     * @tparam U
     * @param kappa
     * @param mq
     * @param x
     */
    auto new_with_matrix(double kappa, Array2<double> mq, Arr xc) -> Ell {
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
    auto new(Arr val, Arr xc) -> Ell {
        Ell::new_with_matrix(1.0, Array2::from_diag(val), xc)
    }

    // /**
    //  * @brief Set the xc object
    //  *
    //  * @param[in] xc
    //  */
    // auto set_xc(Arr xc) { this->xc = xc; }

    /**
     * @brief Update ellipsoid core function using the cut
     *
     *  $grad^T * (x - xc) + beta <= 0$
     *
     * @tparam T
     * @param[in] cut
     * @return (i32, double)
     */
    auto update_single(Array1<double>& grad, double) -> (const CutStatus& beta, double) {
        // const auto (grad, beta) = cut;
        auto mq_g = Array1::zeros(this->n); // initial x0
        auto omega = 0.0;
        for (auto i : range(this->n)) {
            for (auto j : range(this->n)) {
                mq_g[i] += this->mq[[i, j]] * grad[j];
            }
            omega += mq_g[i] * grad[i];
        }

        this->helper.tsq = this->kappa * omega;
        const auto status = this->helper.calc_dc(*beta);
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
                this->mq[[i, j]] -= r_mq_g * mq_g[j];
                this->mq[[j, i]] = this->mq[[i, j]];
            }
            this->mq[[i, i]] -= r_mq_g * mq_g[i];
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
    auto update_parallel(
        
        Array1<double>& grad,
        (const double& beta, Option<double>),
    ) -> (CutStatus, double) {
        // const auto (grad, beta) = cut;
        auto mq_g = Array1::zeros(this->n); // initial x0
        auto omega = 0.0;
        for (auto i : range(this->n)) {
            for (auto j : range(this->n)) {
                mq_g[i] += this->mq[[i, j]] * grad[j];
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
                this->mq[[i, j]] -= r_mq_g * mq_g[j];
                this->mq[[j, i]] = this->mq[[i, j]];
            }
            this->mq[[i, i]] -= r_mq_g * mq_g[i];
        }

        this->kappa *= this->helper.delta;

        if (this->no_defer_trick) {
            this->mq *= this->kappa;
            this->kappa = 1.0;
        }
        return {status, this->helper.tsq};
    }
};

impl SearchSpace for Ell {
    using ArrayType = Array1<double>;

    /**
     * @brief copy the whole array anyway
     *
     * @return Arr
     */
    auto xc() const -> Self::ArrayType {
        this->xc.clone()
    }

    auto update<T>((const Self::ArrayType& cut, T)) -> (CutStatus, double)
    where
        UpdateByCutChoices<Self T, ArrayType = Self::ArrayType>,
    {
        const auto (grad, beta) = cut;
        beta.update_by(self, grad)
    }
};

impl UpdateByCutChoices<Ell> for double {
    using ArrayType = Arr;

    auto update_by(Ell& ell, Self::ArrayType) -> (const CutStatus& grad, double) {
        const auto beta = self;
        ell.update_single(grad, beta)
    }
};

impl UpdateByCutChoices<Ell> for (double, Option<double>) {
    using ArrayType = Arr;

    auto update_by(Ell& ell, Self::ArrayType) -> (const CutStatus& grad, double) {
        const auto beta = self;
        ell.update_parallel(grad, beta)
    }
};
