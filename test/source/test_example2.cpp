#include <doctest/doctest.h> // for ResultBuilder, TestCase, CHECK

#include <ellalgo/cutting_plane.hpp>   // for cutting_plane_optim
#include <ellalgo/ell.hpp>             // for ell
#include <utility>                     // for pair
#include <xtensor/xaccessible.hpp>     // for xconst_accessible
#include <xtensor/xarray.hpp>          // for xarray_container
#include <xtensor/xlayout.hpp>         // for layout_type, layout_type::row...
#include <xtensor/xtensor_forward.hpp> // for xarray

#include "ellalgo/ell_config.hpp" // for CInfo, CUTStatus, CUTStatus::...

using Arr1 = xt::xarray<double, xt::layout_type::row_major>;

struct MyOracle {
  using ArrayType = Arr1;
  using CutChoices = double;
  using Cut = std::pair<Arr1, double>;

  /**
   * @brief
   *
   * @param[in] z
   * @return std::optional<Cut>
   */
  auto assess_feas(const Arr1 &z) -> Cut * {
    static auto cut1 = Cut{Arr1{1.0, 1.0}, 0.0};
    static auto cut2 = Cut{Arr1{-1.0, 1.0}, 0.0};

    const auto x = z[0];
    const auto y = z[1];

    // constraint 1: x + y <= 3
    const auto fj = x + y - 3.0;
    if (fj > 0.0) {
      cut1.second = fj;
      return &cut1;
    }
    // constraint 2: x - y >= 1
    const auto fj2 = -x + y + 1.0;
    if (fj2 > 0.0) {
      cut2.second = fj2;
      return &cut2;
    }
    return nullptr;
  }
};

TEST_CASE("Example 2, test feasible") {
  auto ell = Ell(Arr1{10.0, 10.0}, Arr1{0.0, 0.0});
  auto oracle = MyOracle{};
  const auto options = Options{2000, 1e-12};
  const auto cinfo = cutting_plane_feas(oracle, ell, options);
  const auto f1 = cinfo.feasible;
  CHECK(f1);
}
