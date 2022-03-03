#pragma once

// using CInfo = (bool, size_t, CutStatus);

struct Options {
    size_t max_iter;
    double tol;
};

/*
// #[derive(Debug, Clone, Copy)]
enum CutChoices {
    Single(double),
    Parallelstd::pair<double, std::optional<double>>,
};
*/
// #[derive(Debug, PartialEq, Eq)]
enum CutStatus {
    Success,
    NoSoln,
    NoEffect,
    SmallEnough,
};

/// TODO: support 1D problems

trait UpdateByCutChoices<SS> {
    typename ArrayType;  // double for 1D; ndarray::Arr1 for general

    auto update_by(SS & ss, const Self::ArrayType& grad)->std::pair<CutStatus, double>;
};

/// Oracle for feasibility problems
trait OracleFeas {
    typename ArrayType;   // double for 1D; ndarray::Arr1 for general
    typename CutChoices;  // double for single cut; (double, Option<double) for parallel cut
    auto assess_feas(const Self::ArrayType& x)->Option<(Self::ArrayType, Self::CutChoices)>;
};

/// Oracle for optimization problems
trait OracleOptim {
    typename ArrayType;   // double for 1D; ndarray::Arr1 for general
    typename CutChoices;  // double for single cut; (double, Option<double) for parallel cut
    auto assess_optim(

        const Self::ArrayType& x, double& t, )
        ->((Self::ArrayType, Self::CutChoices), bool);
};

/// Oracle for quantized optimization problems
trait OracleQ {
    typename ArrayType;   // double for 1D; ndarray::Arr1 for general
    typename CutChoices;  // double for single cut; (double, Option<double) for parallel cut
    auto assess_q(

        const Self::ArrayType& x, double& t, bool retry;)
        ->((Self::ArrayType, Self::CutChoices), bool, Self::ArrayType, bool, );
};

/// Oracle for binary search
trait OracleBS { auto assess_bs(double t)->bool; };

trait SearchSpace {
    typename ArrayType;  // double for 1D; ndarray::Arr1 for general
    auto xc() const->Self::ArrayType;
    auto update<T>((const Self::ArrayType&cut, T))
        ->std::pair<CutStatus, double>
            where UpdateByCutChoices<Self T, ArrayType = Self::ArrayType>,
        Self : Sized;
};
