// -*- coding: utf-8 -*-
#include <doctest/doctest.h> // for ResultBuilder, Approx, CHECK_EQ

#include <cmath>                       // for exp
#include <ellalgo/cutting_plane.hpp>   // for cutting_plane_dc
#include <ellalgo/ell.hpp>             // for ell
#include <ellalgo/ell_stable.hpp>      // for ell_stable
#include <tuple>                       // for get, tuple
#include <xtensor/xaccessible.hpp>     // for xconst_accessible
#include <xtensor/xarray.hpp>          // for xarray_container
#include <xtensor/xlayout.hpp>         // for layout_type, layout_type::row...
#include <xtensor/xtensor_forward.hpp> // for xarray

#include <ellalgo/ell_config.hpp> // for CInfo, CUTStatus, CUTStatus::...

using Arr1 = xt::xarray<double, xt::layout_type::row_major>;

struct MyQuasicCvxOracle {
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

    double sqrtx = z[0];
    double ly = z[1];

    // constraint 1: exp(x) <= y, or sqrtx**2 <= ly
    double fj = sqrtx * sqrtx - ly;
    if (fj > 0.0) {
      return std::make_pair(std::make_pair(Arr1{2 * sqrtx, -1.0}, fj), false);
    }

    // constraint 2: x - y >= 1
    double tmp2 = std::exp(ly);
    double tmp3 = t * tmp2;
    fj = -sqrtx + tmp3;
    if (fj < 0.0) // feasible
    {
      t = sqrtx / tmp2;
      return std::make_pair(std::make_pair(Arr1{-1.0, sqrtx}, 0), true);
    }

    return std::make_pair(std::make_pair(Arr1{-1.0, tmp3}, fj), false);
  }
};

TEST_CASE("xtensor") {
  auto x = Arr1{};
  CHECK(x == Arr1{});
  CHECK_EQ(x, Arr1{});

  x = Arr1{1.0, 2.0};
  CHECK(x != Arr1{});
  CHECK_NE(x, Arr1{});
}

TEST_CASE("Quasiconvex 1, test feasible") {
  Ell E{10.0, Arr1{0.0, 0.0}};

  auto P = MyQuasicCvxOracle{};
  auto t = 0.0;
  const auto options = Options{2000, 1e-12};
  const auto result = cutting_plane_optim(P, E, t, options);
  const auto x = std::get<0>(result);
  REQUIRE(x != Arr1{});
  CHECK_EQ(-t, doctest::Approx(-0.4288673397));
  CHECK_EQ(x[0] * x[0], doctest::Approx(0.499876));
  CHECK_EQ(std::exp(x[1]), doctest::Approx(1.64852));
}

TEST_CASE("Quasiconvex 1, test feasible (stable)") {
  EllStable E{10.0, Arr1{0.0, 0.0}};
  auto P = MyQuasicCvxOracle{};
  auto t = 0.0;
  const auto options = Options{2000, 1e-12};
  const auto result = cutting_plane_optim(P, E, t, options);
  const auto x = std::get<0>(result);
  REQUIRE(x != Arr1{});
  // const auto x = *x_opt;
  // CHECK_EQ(-t, doctest::Approx(-0.4288673397));
  // CHECK_EQ(x[0] * x[0], doctest::Approx(0.5029823096));
  // CHECK_EQ(std::exp(x[1]), doctest::Approx(1.6536872635));
}
