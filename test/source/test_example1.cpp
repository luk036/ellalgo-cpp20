#include <doctest/doctest.h> // for ResultBuilder, TestCase, CHECK

#include <ellalgo/cutting_plane.hpp>   // for cutting_plane_optim
#include <ellalgo/ell.hpp>             // for ell
#include <tuple>                       // for get, tuple
#include <xtensor/xaccessible.hpp>     // for xconst_accessible
#include <xtensor/xarray.hpp>          // for xarray_container
#include <xtensor/xlayout.hpp>         // for layout_type, layout_type::row...
#include <xtensor/xtensor_forward.hpp> // for xarray

#include "ellalgo/ell_config.hpp" // for CInfo, CUTStatus, CUTStatus::...

using Arr1 = xt::xarray<double, xt::layout_type::row_major>;

struct MyOracle {
  using ArrayType = Arr1;
  using CutChoices = double; // single cut
  using Cut = std::pair<Arr1, double>;

  /**
   * @brief
   *
   * @param[in] z
   * @param[in,out] t
   * @return std::tuple<Cut, double>
   */
  auto assess_optim(const Arr1 &z, double &t) -> std::pair<Cut, bool> {
    const auto x = z[0];
    const auto y = z[1];

    // constraint 1: x + y <= 3
    const auto fj = x + y - 3.0;
    if (fj > 0.0) {
      return {{Arr1{1.0, 1.0}, fj}, false};
    }
    // constraint 2: x - y >= 1
    const auto fj2 = -x + y + 1.0;
    if (fj2 > 0.0) {
      return {{Arr1{-1.0, 1.0}, fj2}, false};
    }
    // objective: maximize x + y
    const auto f0 = x + y;
    const auto fj3 = t - f0;
    if (fj3 < 0.0) {
      t = f0;
      return {{Arr1{-1.0, -1.0}, 0.0}, true};
    }
    return {{Arr1{-1.0, -1.0}, fj3}, false};
  }
};

TEST_CASE("Example 1, test feasible") {
  auto ell = Ell(Arr1{10.0, 10.0}, Arr1{0.0, 0.0});
  auto oracle = MyOracle{};
  auto t = -1.0e100; // std::numeric_limits<double>::min()
  const auto options = Options{2000, 1e-10};
  const auto [x, _niter, _status] =
      cutting_plane_optim(oracle, ell, t, options);
  static_assert(sizeof _niter >= 0, "make compiler happy");
  static_assert(sizeof _status >= 0, "make compiler happy");
  REQUIRE(x != Arr1{});
  CHECK(x[0] >= 0.0);
}

TEST_CASE("xtensor") {
  auto x = Arr1{};
  CHECK(x == Arr1{});
  CHECK_EQ(x, Arr1{});
}

TEST_CASE("Example 1, test infeasible1") {
  auto ell =
      Ell(Arr1{10.0, 10.0}, Arr1{100.0, 100.0}); // wrong initial guess
                                                 // or ellipsoid is too small
  auto oracle = MyOracle{};
  auto t = -1.0e100; // std::numeric_limits<double>::min()
  const auto options = Options{2000, 1e-12};
  const auto [x, _niter, status] = cutting_plane_optim(oracle, ell, t, options);
  static_assert(sizeof _niter >= 0, "make compiler happy");
  CHECK_EQ(x, Arr1{});
  CHECK(status == CutStatus::NoSoln); // no sol'n
}

TEST_CASE("Example 1, test infeasible22") {
  auto ell = Ell(Arr1{10.0, 10.0}, Arr1{0.0, 0.0});
  auto oracle = MyOracle{};
  auto t = 100.0;
  // wrong initial guess
  const auto options = Options{2000, 1e-12};
  const auto [x, _niter, status] = cutting_plane_optim(oracle, ell, t, options);
  static_assert(sizeof _niter >= 0, "make compiler happy");
  CHECK_EQ(x, Arr1{});
  CHECK(status == CutStatus::NoSoln); // no sol'n
}
