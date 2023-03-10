#pragma once

// #include <optional> // for optional
#include <utility> // for pair

#include "ell_config.hpp"
#include <tuple>

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
 * @return CInfo        Information of Cutting-plane method
 */
template <typename Oracle, typename Space>
#if __cpp_concepts >= 201907L
requires(OracleFeas<Oracle> &&SearchSpace<Space, CutChoices<Oracle>>)
#endif
    auto cutting_plane_feas(Oracle &omega, Space &ss, const Options &options)
        -> CInfo {
  for (auto niter = 0U; niter < options.max_iter; ++niter) {
    const auto cut = omega.assess_feas(ss.xc()); // query the oracle at &ss.xc()
    if (!cut) {
      // feasible sol'n obtained
      const auto result = ss.update(*cut); // update ss
      const auto cutstatus = result.first;
      const auto tsq = result.second;
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
 * @param[in,out] omega perform assessment on x0
 * @param[in,out] ss    search Space containing x*
 * @param[in,out] t     best-so-far optimal sol'n
 * @param[in] options   maximum iteration and error tolerance etc.
 * @return Information of Cutting-plane method
 */
template <typename Oracle, typename Space>
#if __cpp_concepts >= 201907L
requires(OracleOptim<Oracle> &&SearchSpace<Space, CutChoices<Oracle>>)
#endif
    auto cutting_plane_optim(Oracle &omega, Space &ss, double &t,
                             const Options &options)
        -> std::tuple<ArrayType<Oracle>, size_t, CutStatus> {
  ArrayType<Oracle> x_best{};
  auto status = CutStatus::NoSoln;

  for (auto niter = 0U; niter < options.max_iter; ++niter) {
    const auto assess =
        omega.assess_optim(ss.xc(), t); // query the oracle at &ss.xc()
    const auto cut = assess.first;
    const auto shrunk = assess.second;
    if (shrunk) {
      // best t obtained
      x_best = ss.xc(); // TODO
      status = CutStatus::Success;
    }
    const auto result = ss.update(cut); // update ss
    const auto cutstatus = result.first;
    const auto tsq = result.second;
    if (cutstatus != CutStatus::Success) {
      return std::make_tuple(std::move(x_best), niter, cutstatus);
    }
    if (tsq < options.tol) {
      return std::make_tuple(std::move(x_best), niter, CutStatus::SmallEnough);
    }
  }
  return std::make_tuple(std::move(x_best), options.max_iter, status);
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
// #[allow(dead_code)]
template <typename Oracle, typename Space>
#if __cpp_concepts >= 201907L
requires(OracleQ<Oracle> &&SearchSpace<Space, CutChoices<Oracle>>)
#endif
    auto cutting_plane_q(Oracle &omega, Space &ss, double &t,
                         const Options &options)
        -> std::tuple<ArrayType<Oracle>, size_t, CutStatus> {
  auto x_best = ArrayType<Oracle>{};
  auto status = CutStatus::NoSoln; // note!!!
  auto retry = false;

  for (auto niter = 0U; niter < options.max_iter; ++niter) {
    const auto assess =
        omega.assess_q(ss.xc(), t, retry); // query the oracle at &ss.xc()
    const auto &cut = std::get<0>(assess);
    const auto shrunk = std::get<1>(assess);
    const auto &x0 = std::get<2>(assess);
    const auto more_alt = std::get<3>(assess);
    if (shrunk) {
      // best t obtained
      x_best = x0; // x0
    }
    const auto result = ss.update(cut); // update ss
    const auto cutstatus = result.first;
    const auto tsq = result.second;
    if (cutstatus == CutStatus::NoEffect) {
      if (!more_alt) {
        // more alt?
        return {std::move(x_best), niter, status};
      }
      status = cutstatus;
      retry = true;
    } else if (cutstatus == CutStatus::NoSoln) {
      return {std::move(x_best), niter, CutStatus::NoSoln};
    }
    if (tsq < options.tol) {
      return {std::move(x_best), niter, CutStatus::SmallEnough};
    }
  }
  return {std::move(x_best), options.max_iter, status};
} // END

/**
 * @brief
 *
 * @tparam Oracle
 * @tparam Space
 * @param[in,out] omega    perform assessment on x0
 * @param[in,out] intvl    interval containing x*
 * @param[in]     options  maximum iteration and error tolerance etc.
 * @return CInfo
 */
// #[allow(dead_code)]
template <typename T, typename Oracle>
#if __cpp_concepts >= 201907L
requires OracleBS<Oracle>
#endif
auto bsearch(Oracle &omega, std::pair<T, T> &intvl, const Options &options)
    -> CInfo {
  auto &lower = intvl.first;
  auto &upper = intvl.second;
  assert(lower <= upper);
  const auto u_orig = upper;

  for (auto niter = 0U; niter < options.max_iter; ++niter) {
    const auto tau = (upper - lower) / 2; // T may be an integer
    if (tau < options.tol) {
      return {upper != u_orig, niter, CutStatus::SmallEnough};
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
  return {upper != u_orig, options.max_iter, CutStatus::NoSoln};
}

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
//     bsearch_adaptor(const Oracle& P, const Space& ss) bsearch_adaptor{P , ss,
//     Options()} {}

//     /**
//      * @brief Construct a new bsearch adaptor object
//      *
//      * @param[in,out] P perform assessment on x0
//      * @param[in,out] ss search Space containing x*
//      * @param[in] options maximum iteration and error tolerance etc.
//      */
//     bsearch_adaptor(const Oracle& P, const Space& ss, const const Options&
//     options)
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
//     template <typename opt_type> auto operator()(const opt_const typename& t)
//     -> bool {
//         Space ss = this->ss.copy();
//         this->P.update(t);
//         const auto ell_info = cutting_plane_feas(this->P, ss, this->options);
//         if (ell_info.feasible) {
//             this->ss.set_xc(ss.xc());
//         }
//         return ell_info.feasible;
//     }
// };
