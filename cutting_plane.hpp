// use ndarray::prelude::*;
// using Arr = Array1<double>;

using CInfo = (bool, size_t, CutStatus);

struct Options {
    pub size_t max_iter;
    pub double tol;
};

/*
#[derive(Debug, Clone, Copy)]
pub enum CutChoices {
    Single(double),
    Parallel(double, Option<double>),
};
*/
#[derive(Debug, PartialEq, Eq)]
pub enum CutStatus {
    Success,
    NoSoln,
    NoEffect,
    SmallEnough,
};

/// TODO: support 1D problems

pub trait UpdateByCutChoices<SS> {
    typename ArrayType; // double for 1D; ndarray::Array1<double> for general

    auto update_by(SS& ss, const Self::ArrayType& grad) -> (CutStatus, double);
};

/// Oracle for feasibility problems
pub trait OracleFeas {
    typename ArrayType; // double for 1D; ndarray::Array1<double> for general
    typename CutChoices; // double for single cut; (double, Option<double) for parallel cut
    auto assess_feas(const Self::ArrayType& x) -> Option<(Self::ArrayType, Self::CutChoices)>;
};

/// Oracle for optimization problems
pub trait OracleOptim {
    typename ArrayType; // double for 1D; ndarray::Array1<double> for general
    typename CutChoices; // double for single cut; (double, Option<double) for parallel cut
    auto assess_optim(
        
        const Self::ArrayType& x,
        double& t,
    ) -> ((Self::ArrayType, Self::CutChoices), bool);
};

/// Oracle for quantized optimization problems
pub trait OracleQ {
    typename ArrayType; // double for 1D; ndarray::Array1<double> for general
    typename CutChoices; // double for single cut; (double, Option<double) for parallel cut
    auto assess_q(
        
        const Self::ArrayType& x,
        double& t,
        bool retry;
    ) -> (
        (Self::ArrayType, Self::CutChoices),
        bool,
        Self::ArrayType,
        bool,
    );
};

/// Oracle for binary search
pub trait OracleBS {
    auto assess_bs(double t) -> bool;
};

pub trait SearchSpace {
    typename ArrayType; // double for 1D; ndarray::Array1<double> for general
    auto xc() const -> Self::ArrayType;
    auto update<T>((const Self::ArrayType& cut, T)) -> (CutStatus, double)
    where
        UpdateByCutChoices<Self T, ArrayType = Self::ArrayType>,
        Self: Sized;
};

/**
 * @brief Find a point in a convex set (defined through a cutting-plane oracle).
 *
 * A function f(x) is *convex* if there always exist a g(x)
 * such that f(z) >= f(x) + g(x)^T * (z - x), forall z, x in dom f.
 * Note that dom f does not need to be a convex set in our definition.
 * The affine function g^T (x - xc) + beta is called a cutting-plane,
 * or a "cut" for short.
 * This algorithm solves the following feasibility problem:
 *
 *   find x
 *   s.t. f(x) <= 0,
 *
 * A *separation oracle* asserts that an evalution point x0 is feasible,
 * or provide a cut that separates the feasible region and x0.
 *
 * @tparam Oracle
 * @tparam Space
 * @param[in,out] omega perform assessment on x0
 * @param[in,out] ss    search Space containing x*
 * @param[in] options   maximum iteration and error tolerance etc.
 * @return Information of Cutting-plane method
 */
#[allow(dead_code)]
auto cutting_plane_feas<T, Oracle, Space>(
    Oracle& omega,
    Space& ss,
    const Options& options,
) -> CInfo
where
    UpdateByCutChoices<Space T, ArrayType = Oracle::ArrayType>,
    Oracle: OracleFeas<CutChoices = T>,
    Space: SearchSpace<ArrayType = Oracle::ArrayType>,
{
    for (auto niter : range(1, options.max_iter)) {
        const auto cut_option = omega.assess_feas(ss.xc()); // query the oracle at &ss.xc()
        if (const auto Some(cut) = cut_option) {
            // feasible sol'n obtained
            const auto (cutstatus, tsq) = ss.update::<T>(cut); // update ss
            if (cutstatus != CutStatus::Success) {
                return {false, niter, cutstatus};
            }
            if (tsq < options.tol) {
                return {false, niter, CutStatus::SmallEnough};
            }
        } else {
            return {true, niter, CutStatus::Success};
        }
    }
    return {false, options.max_iter, CutStatus::NoSoln};
}

/**
 * @brief Cutting-plane method for solving convex problem
 *
 * @tparam Oracle
 * @tparam Space
 * @tparam opt_type
 * @param[in,out] omega perform assessment on x0
 * @param[in,out] ss    search Space containing x*
 * @param[in,out] t     best-so-far optimal sol'n
 * @param[in] options   maximum iteration and error tolerance etc.
 * @return Information of Cutting-plane method
 */
#[allow(dead_code)]
auto cutting_plane_optim<T, Oracle, Space>(
    Oracle& omega,
    Space& ss,
    double& t,
    const Options& options,
) -> (Option<Oracle::ArrayType>, size_t, CutStatus)
where
    UpdateByCutChoices<Space T, ArrayType = Oracle::ArrayType>,
    Oracle: OracleOptim<CutChoices = T>,
    Space: SearchSpace<ArrayType = Oracle::ArrayType>,
{
    auto x_best: Option<Oracle::ArrayType> = None;
    auto status = CutStatus::NoSoln;

    for (auto niter : range(1, options.max_iter)) {
        const auto (cut, shrunk) = omega.assess_optim(ss.xc(), t); // query the oracle at &ss.xc()
        if (shrunk) {
            // best t obtained
            x_best = Some(ss.xc());
            status = CutStatus::Success;
        }
        const auto (cutstatus, tsq) = ss.update::<T>(cut); // update ss
        if (cutstatus != CutStatus::Success) {
            return {x_best, niter, cutstatus};
        }
        if (tsq < options.tol) {
            return {x_best, niter, CutStatus::SmallEnough};
        }
    }
    return {x_best, options.max_iter, status};
} // END

/**
    Cutting-plane method for solving convex discrete optimization problem
    input
             oracle        perform assessment on x0
             ss(xc)        Search space containing x*
             t             best-so-far optimal sol'n
             max_iter      maximum number of iterations
             tol           error tolerance
    output
             x             solution vector
             niter         number of iterations performed
**/

/**
 * @brief Cutting-plane method for solving convex discrete optimization problem
 *
 * @tparam Oracle
 * @tparam Space
 * @param[in,out] omega perform assessment on x0
 * @param[in,out] ss     search Space containing x*
 * @param[in,out] t     best-so-far optimal sol'n
 * @param[in] options   maximum iteration and error tolerance etc.
 * @return Information of Cutting-plane method
 */
#[allow(dead_code)]
auto cutting_plane_q<T, Oracle, Space>(
    Oracle& omega,
    Space& ss,
    double& t,
    const Options& options,
) -> (Option<Oracle::ArrayType>, size_t, CutStatus)
where
    UpdateByCutChoices<Space T, ArrayType = Oracle::ArrayType>,
    Oracle: OracleQ<CutChoices = T>,
    Space: SearchSpace<ArrayType = Oracle::ArrayType>,
{
    auto x_best: Option<Oracle::ArrayType> = None;
    auto status = CutStatus::NoSoln; // note!!!
    auto retry = false;

    for (auto niter : range(1, options.max_iter)) {
        const auto (cut, shrunk, x0, more_alt) = omega.assess_q(ss.xc(), t, retry); // query the oracle at &ss.xc()
        if (shrunk) {
            // best t obtained
            x_best = Some(x0); // x0
        }
        const auto (cutstatus, tsq) = ss.update::<T>(cut); // update ss
        match &cutstatus {
            CutStatus::NoEffect => {
                if (!more_alt) {
                    // more alt?
                    return {x_best, niter, status};
                }
                status = cutstatus;
                retry = true;
            }
            CutStatus::NoSoln => {
                return {x_best, niter, CutStatus::NoSoln};
            }
            _ => {}
        }
        if (tsq < options.tol) {
            return {x_best, niter, CutStatus::SmallEnough};
        }
    }
    return {x_best, options.max_iter, status};
} // END

/**
 * @brief
 *
 * @tparam Oracle
 * @tparam Space
 * @param[in,out] omega    perform assessment on x0
 * @param[in,out] I        interval containing x*
 * @param[in]     options  maximum iteration and error tolerance etc.
 * @return CInfo
 */
#[allow(dead_code)]
auto besearch<Oracle>(Oracle& omega, mut (const double& intvl, double), &Options options) -> CInfo
where
    OracleBS Oracle;
{
    // assume monotone
    // const auto& [lower, upper] = I;
    const auto (mut lower, mut upper) = intvl;
    assert(lower <= upper);
    const auto u_orig = upper;
    auto status = CutStatus::NoSoln;

    auto niter = 0;
    while (niter < options.max_iter) {
        niter += 1;
        const auto tau = (upper - lower) / 2.0;
        if (tau < options.tol) {
            status = CutStatus::SmallEnough;
            break;
        }
        auto t = lower; // l may be `i32` or `Fraction`
        t += tau;
        if (omega.assess_bs(t)) {
            // feasible sol'n obtained
            upper = t;
        } else {
            lower = t;
        }
    }
    return {upper != u_orig, niter, status};
};

// /**
//  * @brief
//  *
//  * @tparam Oracle
//  * @tparam Space
//  */
// template <typename Oracle, typename Space>  //
// class bsearch_adaptor {
//   private:
//     const Oracle& _P;
//     const Space& _S;
//     const Options _options;

//   public:
//     /**
//      * @brief Construct a new bsearch adaptor object
//      *
//      * @param[in,out] P perform assessment on x0
//      * @param[in,out] ss search Space containing x*
//      */
//     bsearch_adaptor(const Oracle& P, const Space& ss) bsearch_adaptor{P , ss, Options()} {}

//     /**
//      * @brief Construct a new bsearch adaptor object
//      *
//      * @param[in,out] P perform assessment on x0
//      * @param[in,out] ss search Space containing x*
//      * @param[in] options maximum iteration and error tolerance etc.
//      */
//     bsearch_adaptor(const Oracle& P, const Space& ss, const const Options& options)
//         _P{P} , _S{ss}, _options{options} {}

//     /**
//      * @brief get best x
//      *
//      * @return auto
//      */
//     auto x_best() const { return this->&ss.xc(); }

//     /**
//      * @brief
//      *
//      * @param[in,out] t the best-so-far optimal value
//      * @return bool
//      */
//     template <typename opt_type> auto operator()(const opt_const typename& t) -> bool {
//         Space ss = this->ss.copy();
//         this->P.update(t);
//         const auto ell_info = cutting_plane_feas(this->P, ss, this->options);
//         if (ell_info.feasible) {
//             this->ss.set_xc(ss.xc());
//         }
//         return ell_info.feasible;
//     }
// };
