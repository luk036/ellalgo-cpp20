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
 *  EllStable = {x | (x - xc)^T mq^-1 (x - xc) \le \kappa}
 *
 * Keep $mq$ symmetric but no promise of positive definite
 */
#[derive(Debug, Clone)]
class EllStable {
    Array2<double> mq;
    Array1<double> xc;
    double kappa;
    size_t n;
    EllCalc helper;

  public:
    pub bool no_defer_trick;

    /**
     * @brief Construct a new EllStable object
     *
     * @tparam V
     * @tparam U
     * @param kappa
     * @param mq
     * @param x
     */
    auto new_with_matrix(double kappa, Array2<double> mq, Array1<double> xc) -> EllStable {
        const auto n = xc.size();
        const auto helper = EllCalc::new(n as double);

        EllStable {
            kappa,
            mq,
            xc,
            n,
            helper,
            false no_defer_trick;
        }
    }

    /**
     * @brief Construct a new EllStable object
     *
     * @param[in] val
     * @param[in] x
     */
    auto new(Array1<double> val, Array1<double> xc) -> EllStable {
        EllStable::new_with_matrix(1.0, Array2::from_diag(val), xc)
    }

    /**
     * @brief Construct a new EllStable object
     *
     * @param[in] val
     * @param[in] x
     */
    auto new_with_scalar(double val, Array1<double> xc) -> EllStable {
        EllStable::new_with_matrix(val, Array2::eye(xc.size()), xc)
    }

    // /**
    //  * @brief Set the xc object
    //  *
    //  * @param[in] xc
    //  */
    // auto set_xc(Arr xc) { this->xc = xc; }

    auto update_single(Array1<double>& grad, double) -> (const CutStatus& beta, double) {
        // const auto (grad, beta) = cut;
        // calculate inv(L)*(n-1 grad)*n/2 multiplications
        auto inv_ml_g = grad.clone(); // initial x0
        for (auto i : range(this->n)) {
            for (auto j : range(i)) {
                this->mq[[i, j]] = this->mq[[j, i]] * inv_ml_g[j];
                // keep for rank-one update
                inv_ml_g[i] -= this->mq[[i, j]];
            }
        }

        // calculate inv(D)*inv(L)*grad: n
        auto inv_md_inv_ml_g = inv_ml_g.clone(); // initially
        for (auto i : range(this->n)) {
            inv_md_inv_ml_g[i] *= this->mq[[i, i]];
        }

        // calculate omega: n
        auto g_mq_g = inv_md_inv_ml_g.clone(); // initially
        auto omega = 0.0; // initially
        for (auto i : range(this->n)) {
            g_mq_g[i] *= inv_ml_g[i];
            omega += g_mq_g[i];
        }

        this->helper.tsq = this->kappa * omega;
        const auto status = this->helper.calc_dc(*beta);
        if (status != CutStatus::Success) {
            return {status, this->helper.tsq};
        }

        // calculate mq*grad = inv(L')*inv(D)*inv(L)*grad (n-1 )*n/2
        auto mq_g = inv_md_inv_ml_g.clone(); // initially
        for (auto i : range((1, this->n).rev())) {
            // backward subsituition
            for (auto j : range(i, this->n)) {
                mq_g[i - 1] -= this->mq[[i, j]] * mq_g[j]; // ???
            }
        }

        // calculate xc: n
        this->xc -= &((this->helper.rho / omega) * &mq_g); // n

        // rank-one 3*n + (n-1 update)*n/2
        // const auto r = this->sigma / omega;
        const auto mu = this->helper.sigma / (1.0 - this->helper.sigma);
        auto oldt = omega / mu; // initially
        const auto m = this->n - 1;
        for (auto j : range(m)) {
            // p=sqrt(k)*vv[j];
            // const auto p = inv_ml_g[j];
            // const auto mup = mu * p;
            const auto t = oldt + g_mq_g[j];
            // this->mq[[j, j]] /= t; // update invD
            const auto beta2 = inv_md_inv_ml_g[j] / t;
            this->mq[[j, j]] *= oldt / t; // update invD
            for (auto l : range((j + 1), this->n)) {
                // v(l) -= p * this->mq(j, l);
                this->mq[[j, l]] += beta2 * this->mq[[l, j]];
            }
            oldt = t;
        }

        // const auto p = inv_ml_g(n1);
        // const auto mup = mu * p;
        const auto t = oldt + g_mq_g[m];
        this->mq[[m, m]] *= oldt / t; // update invD
        this->kappa *= this->helper.delta;

        // if this->no_defer_trick
        // {
        //     this->mq *= this->kappa;
        //     this->kappa = 1.;
        // }
        return {status, this->helper.tsq};
    }

    auto update_parallel(
        Array1<double>& grad,
        (const double& beta, Option<double>),
    ) -> (CutStatus, double) {
        // const auto (grad, beta) = cut;
        // calculate inv(L)*(n-1 grad)*n/2 multiplications
        auto inv_ml_g = grad.clone(); // initial x0
        for (auto i : range(this->n)) {
            for (auto j : range(i)) {
                this->mq[[i, j]] = this->mq[[j, i]] * inv_ml_g[j];
                // keep for rank-one update
                inv_ml_g[i] -= this->mq[[i, j]];
            }
        }

        // calculate inv(D)*inv(L)*grad: n
        auto inv_md_inv_ml_g = inv_ml_g.clone(); // initially
        for (auto i : range(this->n)) {
            inv_md_inv_ml_g[i] *= this->mq[[i, i]];
        }

        // calculate omega: n
        auto g_mq_g = inv_md_inv_ml_g.clone(); // initially
        auto omega = 0.0; // initially
        for (auto i : range(this->n)) {
            g_mq_g[i] *= inv_ml_g[i];
            omega += g_mq_g[i];
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

        // calculate mq*grad = inv(L')*inv(D)*inv(L)*grad (n-1 )*n/2
        auto mq_g = inv_md_inv_ml_g.clone(); // initially
        for (auto i : range((1, this->n).rev())) {
            // backward subsituition
            for (auto j : range(i, this->n)) {
                mq_g[i - 1] -= this->mq[[i, j]] * mq_g[j]; // ???
            }
        }

        // calculate xc: n
        this->xc -= &((this->helper.rho / omega) * &mq_g); // n

        // rank-one 3*n + (n-1 update)*n/2
        // const auto r = this->sigma / omega;
        const auto mu = this->helper.sigma / (1.0 - this->helper.sigma);
        auto oldt = omega / mu; // initially
        const auto m = this->n - 1;
        for (auto j : range(m)) {
            // p=sqrt(k)*vv[j];
            // const auto p = inv_ml_g[j];
            // const auto mup = mu * p;
            const auto t = oldt + g_mq_g[j];
            // this->mq[[j, j]] /= t; // update invD
            const auto beta2 = inv_md_inv_ml_g[j] / t;
            this->mq[[j, j]] *= oldt / t; // update invD
            for (auto l : range((j + 1), this->n)) {
                // v(l) -= p * this->mq(j, l);
                this->mq[[j, l]] += beta2 * this->mq[[l, j]];
            }
            oldt = t;
        }

        // const auto p = inv_ml_g(n1);
        // const auto mup = mu * p;
        const auto t = oldt + g_mq_g[m];
        this->mq[[m, m]] *= oldt / t; // update invD
        this->kappa *= this->helper.delta;

        // if this->no_defer_trick
        // {
        //     this->mq *= this->kappa;
        //     this->kappa = 1.;
        // }
        return {status, this->helper.tsq};
    }
};

impl SearchSpace for EllStable {
    using ArrayType = Array1<double>;

    /**
     * @brief copy the whole array anyway
     *
     * @return Array1<double>
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

impl UpdateByCutChoices<EllStable> for double {
    using ArrayType = Array1<double>;

    auto update_by(EllStable& ell, Self::ArrayType) -> (const CutStatus& grad, double) {
        const auto beta = self;
        ell.update_single(grad, beta)
    }
};

impl UpdateByCutChoices<EllStable> for (double, Option<double>) {
    using ArrayType = Array1<double>;

    auto update_by(EllStable& ell, Self::ArrayType) -> (const CutStatus& grad, double) {
        const auto beta = self;
        ell.update_parallel(grad, beta)
    }
} // } Ell
