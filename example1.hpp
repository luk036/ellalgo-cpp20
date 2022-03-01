use super::cutting_plane::OracleOptim;
use ndarray::prelude::*;

using Arr = Array1<double>;

#[derive(Debug)]
struct MyOracle {}

impl OracleOptim for MyOracle {
    using ArrayType = Arr;
    using CutChoices = double; // single cut

    /**
     * @brief
     *
     * @param[in] z
     * @param[in,out] t
     * @return std::tuple<Cut, double>
     */
    auto assess_optim(const Arr& z, double& t) -> ((Arr, double), bool) {
        const auto x = z[0];
        const auto y = z[1];

        // constraint 1: x + y <= 3
        const auto fj = x + y - 3.0;
        if (fj > 0.0) {
            return ((array![1.0, 1.0], fj), false);
        }
        // constraint 2: x - y >= 1
        const auto fj2 = -x + y + 1.0;
        if (fj2 > 0.0) {
            return ((array![-1.0, 1.0], fj2), false);
        }
        // objective: maximize x + y
        const auto f0 = x + y;
        const auto fj3 = *t - f0;
        if (fj3 < 0.0) {
            *t = f0;
            return ((array![-1.0, -1.0], 0.0), true);
        }
        ((array![-1.0, -1.0], fj3), false)
    }
};

mod tests {
    use super::*;
    use crate::cutting_plane::{cutting_plane_optim, CutStatus, Options};
    use crate::ell::Ell;
    use ndarray::array;
    // use super::ell_stable::EllStable;

    #[test]
    auto test_feasible() {
        auto ell = Ell::new(array![10.0, 10.0], array![0.0, 0.0]);
        auto oracle = MyOracle {};
        auto t = -1.0e100; // std::numeric_limits<double>::min()
        const auto options = Options {
            max_iter: 2000,
            tol: 1e-10,
        };
        const auto (x_opt, _niter, _status) = cutting_plane_optim(oracle, ell, t, options);
        if (const auto Some(x) = x_opt) {
            assert(x[0] >= 0.0);
        } else {
            assert(false); // not feasible
        }
    }

    #[test]
    auto test_infeasible1() {
        auto ell = Ell::new(array![10.0, 10.0], array![100.0, 100.0]); // wrong initial guess
                                                                          // or ellipsoid is too small
        auto oracle = MyOracle {};
        auto t = -1.0e100; // std::numeric_limits<double>::min()
        const auto options = Options {
            max_iter: 2000,
            tol: 1e-12,
        };
        const auto (x_opt, _niter, status) = cutting_plane_optim(oracle, ell, t, options);
        if (const auto Some(_x) = x_opt) {
            assert(false);
        } else {
            assert_eq!(status, CutStatus::NoSoln); // no sol'n
        }
    }

    #[test]
    auto test_infeasible2() {
        auto ell = Ell::new(array![10.0, 10.0], array![0.0, 0.0]);
        auto oracle = MyOracle {};
        // wrong initial guess
        const auto options = Options {
            max_iter: 2000,
            tol: 1e-12,
        };
        const auto (x_opt, _niter, status) =
            cutting_plane_optim(oracle, ell, 100.0, options);
        if (const auto Some(_x) = x_opt) {
            assert(false);
        } else {
            assert_eq!(status, CutStatus::NoSoln); // no sol'n
        }
    }
};
