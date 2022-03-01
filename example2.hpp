use super::cutting_plane::OracleFeas;
use ndarray::prelude::*;

using Arr = Array1<double>;

#[derive(Debug)]
struct MyOracle {
    using ArrayType = Arr;
    using CutChoices = double;

    /**
     * @brief
     *
     * @param[in] z
     * @return std::optional<Cut>
     */
    auto assess_feas(const Arr& z) -> Option<(Arr, double)> {
        const auto x = z[0];
        const auto y = z[1];

        // constraint 1: x + y <= 3
        const auto fj = x + y - 3.0;
        if (fj > 0.0) {
            return Some((array![1.0, 1.0], fj));
        }
        // constraint 2: x - y >= 1
        const auto fj2 = -x + y + 1.0;
        if (fj2 > 0.0) {
            return Some((array![-1.0, 1.0], fj2));
        }
        return None;
    }
};

mod tests {
    use super::*;
    use crate::cutting_plane::{cutting_plane_feas, Options};
    use crate::ell::Ell;
    use ndarray::array;

    // use super::ell_stable::EllStable;

    #[test]
    auto test_example2() {
        auto ell = Ell::new(array![10.0, 10.0], array![0.0, 0.0]);
        auto oracle = MyOracle {};
        const auto options = Options {
            max_iter: 2000,
            tol: 1e-12,
        };
        const auto (feasible, _niter, _status) = cutting_plane_feas(oracle, ell, options);
        assert(feasible);
    }
};
