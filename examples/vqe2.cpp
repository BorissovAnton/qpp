// vqe2.cpp
//
// Faithful(ish) C++ port of /mnt/data/vqe.py (shot-budgeted adaptive estimator)
// into a single-file implementation using qpp for statevector simulation.
//
// Key goal: match Python semantics of
//   VqeCircuit.cost_function(parameters, resource, query_number)
// including:
//   - commutation graph via symplectic inner product
//   - greedy clique selection (optimal_clique as written)
//   - entangling Clifford diagonalization of a commuting clique
//   (diagonalize/diagonalize_iter_)
//   - caching diagonalizers per clique
//   - Bayesian variance graph updates on powers of 2
//   - returning Σ c_i * naive_Mean(X[i,i]) after query_number shots
//
// NOTE: This file intentionally does NOT implement a classical optimizer.
// It provides the "energy oracle" analogous to vqe.py.
//
// Build (example):
//   g++ -O2 -std=c++20 vqe2.cpp -I<path-to-qpp-include> -o vqe2
//
// Usage (example):
//   ./vqe2 --shots 1024 --ansatz_layers 5 < hamiltonian.txt
//
// Input Hamiltonian format (simple long-form, one term per line):
//   <coeff> <paulis>
// Example:
//   -1.05 IIZX
//   0.23  ZIII
//
// Lines starting with # are ignored.

#include <qpp/qpp.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using idx = qpp::idx;
using realT = qpp::realT;
using cplx = qpp::cplx;
using ket = qpp::ket;

// ------------------------- Utilities -------------------------

static bool is_power(int a0, int a1) {
    // exact port of vqe.py is_power()
    if (a0 < 0) {
        return false;
    }
    if (a0 == 0) {
        return true;
    }
    while (true) {
        if (a0 == 1) {
            return true;
        }
        if (a0 % a1) {
            return false;
        }
        a0 /= a1;
    }
}

static inline bool bit_u64(uint64_t m, idx pos) { return (m >> pos) & 1ULL; }
static inline void toggle_bit_u64(uint64_t& m, idx pos) { m ^= (1ULL << pos); }

static inline int popcount_u64(uint64_t x) {
#if defined(__GNUG__) || defined(__clang__)
    return __builtin_popcountll(x);
#else
    // portable fallback
    int c = 0;
    while (x) {
        x &= (x - 1);
        ++c;
    }
    return c;
#endif
}

// Build a stable key for a clique (Python used str(list); we use CSV)
static std::string clique_key(const std::vector<int>& aa) {
    std::ostringstream oss;
    for (size_t i = 0; i < aa.size(); ++i) {
        if (i) {
            oss << ",";
        }
        oss << aa[i];
    }
    return oss.str();
}

// ------------------------- Hamiltonian -------------------------

struct PauliTerm {
    realT coeff{};
    std::string paulis; // length <= q, chars in {I,X,Y,Z}
};

static bool is_valid_pauli_char(char c) {
    return c == 'I' || c == 'X' || c == 'Y' || c == 'Z';
}

static std::vector<PauliTerm> read_hamiltonian_from_stdin(idx& out_n_qubits) {
    std::vector<PauliTerm> H;
    out_n_qubits = 0;

    std::string line;
    while (std::getline(std::cin, line)) {
        // trim leading whitespace
        size_t start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) {
            continue;
        }
        if (line[start] == '#') {
            continue;
        }

        std::istringstream iss(line.substr(start));
        std::string coeff_s;
        std::string pauli_s;
        if (!(iss >> coeff_s >> pauli_s)) {
            continue;
        }

        realT coeff{};
        try {
            coeff = static_cast<realT>(std::stod(coeff_s));
        } catch (...) {
            std::cerr << "Parse error: invalid coefficient: " << coeff_s
                      << "\n";
            std::exit(EXIT_FAILURE);
        }

        for (char c : pauli_s) {
            if (!is_valid_pauli_char(c)) {
                std::cerr << "Parse error: invalid Pauli char '" << c << "' in "
                          << pauli_s << " (allowed: I,X,Y,Z)\n";
                std::exit(EXIT_FAILURE);
            }
        }

        out_n_qubits =
            std::max<idx>(out_n_qubits, static_cast<idx>(pauli_s.size()));
        H.push_back(PauliTerm{coeff, pauli_s});
    }

    if (H.empty()) {
        std::cerr << "No Hamiltonian terms read from stdin.\n";
        std::exit(EXIT_FAILURE);
    }
    return H;
}

struct TestCase {
    std::string name;
    std::string description;
    idx n_qubits;
    int suggested_layers;
    int suggested_shots;
    std::vector<PauliTerm> H;
};

static PauliTerm T(realT c, const std::string& p) { return PauliTerm{c, p}; }

static TestCase get_testcase(int k) {
    switch (k) {
        case 1: {
            return TestCase{"2-qubit H2 molecule",
                            "Should converge to ~-1.8512 Ha for the ground "
                            "state (after optimization).",
                            2,
                            5,
                            2048,
                            {
                                T(-1.052373245772859, "II"),
                                T(0.39793742484318045, "ZI"),
                                T(-0.39793742484318045, "IZ"),
                                T(-0.01128010425623538, "ZZ"),
                                T(0.18093119978423156, "XX"),
                                T(0.18093119978423156, "YY"),
                            }};
        }
        case 2: {
            return TestCase{"4-qubit H2 molecule -- initial",
                            "energy should be approx -1.2370004192490118",
                            4,
                            6,
                            4096,
                            {
                                T(-0.81261, "IIII"),
                                T(0.171201, "ZIII"),
                                T(0.16862325, "IZII"),
                                T(-0.2227965, "IIZI"),
                                T(0.171201, "ZZII"),
                                T(0.12054625, "ZIZI"),
                                T(0.17434925, "IZIZ"),
                                T(0.04532175, "XZXI"),
                                T(0.04532175, "YZYI"),
                                T(0.165868, "ZZZI"),
                                T(0.12054625, "ZIZZ"),
                                T(-0.2227965, "IZZZ"),
                                T(0.04532175, "XZXZ"),
                                T(0.04532175, "YZYZ"),
                                T(0.165868, "ZZZZ"),
                            }};
        }
        case 6: {
            return TestCase{"4-qubit H2 molecule (Jordan-Wigner)",
                            "Hardware-efficient ansatz may have expressibility "
                            "limits; compare estimator vs exact ansatz energy.",
                            4,
                            6,
                            4096,
                            {
                                T(-0.810547980537326, "IIII"),
                                T(0.17218393261915543, "ZIII"),
                                T(0.17218393261915543, "IZII"),
                                T(-0.22575349222402472, "IIZI"),
                                T(-0.22575349222402472, "IIIZ"),
                                T(0.1209126326177663, "ZZII"),
                                T(0.16892753870087912, "ZIZI"),
                                T(0.165867024105366, "ZIIZ"),
                                T(0.165867024105366, "IZZI"),
                                T(0.16892753870087912, "IZIZ"),
                                T(0.17464343068300447, "IIZZ"),
                                T(-0.0452327999460578, "XXYY"),
                                T(0.0452327999460578, "XYYX"),
                                T(0.0452327999460578, "YXXY"),
                                T(-0.0452327999460578, "YYXX"),
                            }};
        }
        case 3: {
            return TestCase{
                "4-qubit Heisenberg chain (PBC)",
                "Antiferromagnetic Heisenberg model with periodic boundaries.",
                4,
                6,
                4096,
                {
                    T(-1, "XXII"),
                    T(-1, "YYII"),
                    T(-1, "ZZII"),
                    T(-1, "IXXI"),
                    T(-1, "IYYI"),
                    T(-1, "IZZI"),
                    T(-1, "IIXX"),
                    T(-1, "IIYY"),
                    T(-1, "IIZZ"),
                    T(-1, "XIIX"),
                    T(-1, "YIIY"),
                    T(-1, "ZIIZ"),
                }};
        }
        case 4: {
            return TestCase{"6-qubit transverse-field Ising (PBC, h/J=1)",
                            "TFIM critical point. Compare estimator to exact "
                            "ansatz energy; ground is a reference.",
                            6,
                            8,
                            8192,
                            {
                                T(-1, "ZZIIII"),
                                T(-1, "IZZIII"),
                                T(-1, "IIZZII"),
                                T(-1, "IIIZZI"),
                                T(-1, "IIIIZZ"),
                                T(-1, "ZIIIIZ"),
                                T(-1, "XIIIII"),
                                T(-1, "IXIIII"),
                                T(-1, "IIXIII"),
                                T(-1, "IIIXII"),
                                T(-1, "IIIIXI"),
                                T(-1, "IIIIIX"),
                            }};
        }
        case 5: {
            return TestCase{
                "4-qubit LiH molecule",
                "LiH Hamiltonian (4 qubits) for estimator validation.",
                4,
                6,
                4096,
                {
                    T(-7.49895, "IIII"), T(0.16199, "ZIII"),
                    T(0.01291, "YZYI"),  T(0.01291, "XZXI"),
                    T(0.16199, "IZII"),  T(0.01291, "IYZY"),
                    T(0.01291, "IXZX"),  T(-0.01324, "IIZI"),
                    T(-0.01324, "IIIZ"), T(0.12445, "ZZII"),
                    T(0.01154, "YIYI"),  T(0.01154, "XIXI"),
                    T(0.01154, "ZYZY"),  T(0.01154, "ZXZX"),
                    T(0.00293, "YXXY"),  T(-0.00293, "YYXX"),
                    T(-0.00293, "XXYY"), T(0.00293, "XYYX"),
                    T(0.05413, "ZIZI"),  T(0.05706, "ZIIZ"),
                    T(-0.00137, "YZYZ"), T(-0.00137, "XZXZ"),
                    T(0.05706, "IZZI"),  T(-0.00137, "IYIY"),
                    T(-0.00137, "IXIX"), T(0.05413, "IZIZ"),
                    T(0.0848, "IIZZ"),
                }};
        }
        default:
            std::cerr << "Unknown --test " << k << " (valid: 1..5)\n";
            std::exit(EXIT_FAILURE);
    }
}

// ------------------------- Circuit ops representation
// ------------------------- We avoid depending on QCircuit concatenation
// internals. We still use qpp gates/apply.

enum class OpType : uint8_t { RY, CZ, H, S, CNOT };

struct Op {
    OpType type{};
    idx q0{};
    idx q1{};      // for 2-qubit gates
    realT theta{}; // for RY
};

// Apply ops to a statevector
static ket apply_ops(const ket& in, idx n_qubits, const std::vector<Op>& ops) {
    ket psi = in;
    for (const auto& op : ops) {
        switch (op.type) {
            case OpType::RY:
                psi = qpp::apply(psi, qpp::gt.RY(op.theta), {op.q0});
                break;
            case OpType::CZ:
                psi = qpp::apply(psi, qpp::gt.CZ, {op.q0, op.q1});
                break;
            case OpType::H:
                psi = qpp::apply(psi, qpp::gt.H, {op.q0});
                break;
            case OpType::S:
                psi = qpp::apply(psi, qpp::gt.S, {op.q0});
                break;
            case OpType::CNOT:
                // If your qpp build lacks gt.CNOT, replace with applyCTRL using
                // gt.X.
                psi = qpp::apply(psi, qpp::gt.CNOT, {op.q0, op.q1});
                break;
        }
    }
    return psi;
}

// Resource = "one shot" sampler: given ops, return one measurement bitstring
// (computational basis)
class StatevectorResource {
  public:
    explicit StatevectorResource(idx n_qubits, uint64_t seed = 0xC0FFEEULL)
        : n_(n_qubits), rng_(seed) {}

    std::vector<bool> operator()(const std::vector<Op>& ops) {
        // start in |0...0>
        ket psi = qpp::mket(std::vector<idx>(n_, 0));
        psi = apply_ops(psi, n_, ops);

        // sample computational basis
        std::vector<double> probs(static_cast<size_t>(psi.size()));
        for (idx i = 0; i < static_cast<idx>(psi.size()); ++i) {
            probs[static_cast<size_t>(i)] = std::norm(psi(i));
        }

        std::discrete_distribution<size_t> dist(probs.begin(), probs.end());
        size_t outcome = dist(rng_);

        // Decode bits.
        // Convention: bit position i corresponds to qubit i.
        // If your qpp basis ordering differs, flip the indexing here.
        std::vector<bool> bits(static_cast<size_t>(n_), false);
        for (idx i = 0; i < n_; ++i) {
            // treat qubit 0 as most significant by default:
            idx shift = (n_ - 1 - i);
            bits[static_cast<size_t>(i)] = ((outcome >> shift) & 1ULL) != 0ULL;
        }
        return bits;
    }

  private:
    idx n_;
    std::mt19937_64 rng_;
};

// Convert bits -> uint64 mask with bit i = qubit i
static uint64_t bits_to_mask_u64(const std::vector<bool>& bits) {
    uint64_t m = 0;
    for (idx i = 0; i < static_cast<idx>(bits.size()); ++i) {
        if (bits[static_cast<size_t>(i)]) {
            m |= (1ULL << i);
        }
    }
    return m;
}

// ------------------------- Bayesian counts -------------------------

struct PairCounts {
    uint64_t pp = 0; // ( +1, +1 )
    uint64_t pm = 0; // ( +1, -1 )
    uint64_t mp = 0; // ( -1, +1 )
    uint64_t mm = 0; // ( -1, -1 )
};

static inline void inc_pair(PairCounts& c, int a, int b) {
    if (a == 1 && b == 1) {
        ++c.pp;
    } else if (a == 1 && b == -1) {
        ++c.pm;
    } else if (a == -1 && b == 1) {
        ++c.mp;
    } else if (a == -1 && b == -1) {
        ++c.mm;
    }
}

static realT naive_Mean(const PairCounts& xDict) {
    // vqe.py naive_Mean: uses only ++ and --
    const realT x0 = static_cast<realT>(xDict.pp);
    const realT x1 = static_cast<realT>(xDict.mm);
    if (x0 + x1 == 0) {
        return 0;
    }
    return (x0 - x1) / (x0 + x1);
}

static realT bayes_Var(const PairCounts& xDict) {
    // vqe.py bayes_Var
    const realT x0 = static_cast<realT>(xDict.pp);
    const realT x1 = static_cast<realT>(xDict.mm);
    return 4 * ((x0 + 1) * (x1 + 1)) / ((x0 + x1 + 2) * (x0 + x1 + 3));
}

static realT bayes_Cov(const PairCounts& xyDict, const PairCounts& xDict,
                       const PairCounts& yDict) {
    // exact port of vqe.py bayes_Cov
    const realT xy00 = static_cast<realT>(xyDict.pp);
    const realT xy01 = static_cast<realT>(xyDict.pm);
    const realT xy10 = static_cast<realT>(xyDict.mp);
    const realT xy11 = static_cast<realT>(xyDict.mm);

    const realT x0 = static_cast<realT>(xDict.pp);
    const realT x1 = static_cast<realT>(xDict.mm);
    const realT y0 = static_cast<realT>(yDict.pp);
    const realT y1 = static_cast<realT>(yDict.mm);

    const realT p00 =
        4 * ((x0 + 1) * (y0 + 1)) / ((x0 + x1 + 2) * (y0 + y1 + 2));
    const realT p01 =
        4 * ((x0 + 1) * (y1 + 1)) / ((x0 + x1 + 2) * (y0 + y1 + 2));
    const realT p10 =
        4 * ((x1 + 1) * (y0 + 1)) / ((x0 + x1 + 2) * (y0 + y1 + 2));
    const realT p11 =
        4 * ((x1 + 1) * (y1 + 1)) / ((x0 + x1 + 2) * (y0 + y1 + 2));

    const realT num =
        4 * ((xy00 + p00) * (xy11 + p11) - (xy01 + p01) * (xy10 + p10));
    const realT den =
        (xy00 + xy01 + xy10 + xy11 + 4) * (xy00 + xy01 + xy10 + xy11 + 5);
    return num / den;
}

static std::vector<std::vector<realT>>
bayes_variance_graph(const std::vector<std::vector<PairCounts>>& X,
                     const std::vector<realT>& coeffs) {
    const int p = static_cast<int>(coeffs.size());
    std::vector<std::vector<realT>> V(p, std::vector<realT>(p, 0));

    for (int i = 0; i < p; ++i) {
        for (int j = 0; j < p; ++j) {
            if (i == j) {
                V[i][j] = coeffs[i] * coeffs[i] * bayes_Var(X[i][i]);
            } else {
                V[i][j] = coeffs[i] * coeffs[j] *
                          bayes_Cov(X[i][j], X[i][i], X[j][j]);
            }
        }
    }
    return V;
}

// ------------------------- Symplectic commutation -------------------------

// inner_product_dictionary from vqe.py
static bool inner_product(char a, char b) {
    // returns True iff single-qubit Paulis anticommute
    // (I with anything commutes, same Pauli commutes, different non-I may
    // anticommute)
    if (a == 'I' || b == 'I') {
        return false;
    }
    if (a == b) {
        return false;
    }
    // X with Y/Z anticommutes; Y with X/Z; Z with X/Y
    return true;
}

static bool symplectic_inner_product(const PauliTerm& P0, const PauliTerm& P1,
                                     bool qubitwise) {
    bool sip = false;
    const size_t n = std::min(P0.paulis.size(), P1.paulis.size());
    for (size_t i = 0; i < n; ++i) {
        sip ^= inner_product(P0.paulis[i], P1.paulis[i]);
        if (sip && qubitwise) {
            return sip;
        }
    }
    return sip;
}

// Faster bitmask commutation (used for diagonalize cache building)
static bool sip_bitmask(uint64_t x0, uint64_t z0, uint64_t x1, uint64_t z1,
                        bool qubitwise) {
    const uint64_t anti = (x0 & z1) ^ (z0 & x1);
    if (qubitwise) {
        return anti != 0;
    }
    return (popcount_u64(anti) & 1) != 0;
}

// ------------------------- Diagonalization (Clifford synthesis)
// -------------------------

struct DiagCacheEntry {
    std::vector<Op> ops; // diagonalizing Clifford circuit (no measurement)
    std::vector<uint64_t>
        Z1; // Z-part after diagonalization (bitmask per clique element)
    std::vector<bool> neg; // sign flips per clique element
};

static void apply_S_on_masks(std::vector<uint64_t>& X, std::vector<uint64_t>& Z,
                             std::vector<bool>& neg, idx a) {
    // neg ^= X[:,a] & Z[:,a];  Z[:,a] ^= X[:,a]
    const uint64_t bit = (1ULL << a);
    for (size_t j = 0; j < X.size(); ++j) {
        const bool xb = (X[j] & bit) != 0;
        const bool zb = (Z[j] & bit) != 0;
        if (xb && zb) {
            neg[j] = !neg[j];
        }
        if (xb) {
            Z[j] ^= bit;
        }
    }
}

static void apply_H_on_masks(std::vector<uint64_t>& X, std::vector<uint64_t>& Z,
                             std::vector<bool>& neg, idx a) {
    // neg ^= X[:,a] & Z[:,a]; swap(X[:,a], Z[:,a])
    const uint64_t bit = (1ULL << a);
    for (size_t j = 0; j < X.size(); ++j) {
        const bool xb = (X[j] & bit) != 0;
        const bool zb = (Z[j] & bit) != 0;
        if (xb && zb) {
            neg[j] = !neg[j];
        }
        // swap bits: if xb != zb, flipping both swaps
        if (xb != zb) {
            X[j] ^= bit;
            Z[j] ^= bit;
        }
    }
}

static void apply_CNOT_forward_masks(std::vector<uint64_t>& X,
                                     std::vector<uint64_t>& Z,
                                     std::vector<bool>& neg, idx control,
                                     idx target) {
    // Python for CX q[control],q[target]:
    // neg ^= X[:,control] & Z[:,target] & (Z[:,control] == X[:,target])
    // X[:,target] ^= X[:,control]
    // Z[:,control] ^= Z[:,target]
    const uint64_t bc = (1ULL << control);
    const uint64_t bt = (1ULL << target);
    for (size_t j = 0; j < X.size(); ++j) {
        const bool x_c = (X[j] & bc) != 0;
        const bool z_t = (Z[j] & bt) != 0;
        const bool z_c = (Z[j] & bc) != 0;
        const bool x_t = (X[j] & bt) != 0; // pre-update
        if (x_c && z_t && (z_c == x_t)) {
            neg[j] = !neg[j];
        }
        if (x_c) {
            X[j] ^= bt;
        }
        if (z_t) {
            Z[j] ^= bc;
        }
    }
}

static void apply_CNOT_backward_masks(std::vector<uint64_t>& X,
                                      std::vector<uint64_t>& Z,
                                      std::vector<bool>& neg, idx control,
                                      idx target) {
    // Python for CX q[control],q[target] but in diagonalize_iter_ this is used
    // as: "CX q[i],q[a]" where control=i, target=a neg ^= X[:,control] &
    // Z[:,target] & (Z[:,control] == X[:,target]) X[:,target] ^= X[:,control]
    // Z[:,control] ^= Z[:,target]
    //
    // This is the SAME formula as "forward" with renamed wires. We keep
    // separate wrapper for clarity.
    apply_CNOT_forward_masks(X, Z, neg, control, target);
}

static void diagonalize_iter(std::vector<uint64_t>& X, std::vector<uint64_t>& Z,
                             std::vector<bool>& neg, std::vector<Op>& out_ops,
                             idx a, idx q) {
    // Exact control flow port of vqe.py diagonalize_iter_()
    const idx p = static_cast<idx>(X.size());
    const uint64_t ba = (1ULL << a);

    // if not any(X[:,a]): return
    bool anyXa = false;
    for (idx r = 0; r < p; ++r) {
        if (X[static_cast<size_t>(r)] & ba) {
            anyXa = true;
            break;
        }
    }
    if (!anyXa) {
        return;
    }

    // a1 = min i with X[i,a] == 1
    idx a1 = 0;
    while (a1 < p && ((X[static_cast<size_t>(a1)] & ba) == 0)) {
        ++a1;
    }
    if (a1 == p) {
        return;
    }

    // if any(X[a1,i] for i>a): for each i>a with X[a1,i] add CX a->i
    const uint64_t X_a1 = X[static_cast<size_t>(a1)];
    for (idx i = a + 1; i < q; ++i) {
        if (bit_u64(X_a1, i)) {
            out_ops.push_back(Op{OpType::CNOT, a, i, 0});
            apply_CNOT_forward_masks(X, Z, neg, a, i);
        }
    }

    // if any(Z[a1,i] for i>a):
    const uint64_t Z_a1 = Z[static_cast<size_t>(a1)];
    bool anyZa1_gt = false;
    for (idx i = a + 1; i < q; ++i) {
        if (bit_u64(Z_a1, i)) {
            anyZa1_gt = true;
            break;
        }
    }

    if (anyZa1_gt) {
        // if not Z[a1,a]: apply S on a
        if (!bit_u64(Z[static_cast<size_t>(a1)], a)) {
            out_ops.push_back(Op{OpType::S, a, 0, 0});
            apply_S_on_masks(X, Z, neg, a);
        }

        // for each i>a with Z[a1,i]: add backward CX i->a
        for (idx i = a + 1; i < q; ++i) {
            if (bit_u64(Z[static_cast<size_t>(a1)], i)) {
                out_ops.push_back(Op{OpType::CNOT, i, a, 0});
                // this is CX control=i target=a (same conjugation rule)
                apply_CNOT_backward_masks(X, Z, neg, i, a);
            }
        }
    }

    // if Z[a1,a]: apply S on a (Y -> X in python logic)
    if (bit_u64(Z[static_cast<size_t>(a1)], a)) {
        out_ops.push_back(Op{OpType::S, a, 0, 0});
        apply_S_on_masks(X, Z, neg, a);
    }
}

static DiagCacheEntry diagonalize_clique(const std::vector<uint64_t>& X_in,
                                         const std::vector<uint64_t>& Z_in,
                                         idx q) {
    // Port of vqe.py diagonalize(): returns ops, Z1, neg.
    std::vector<uint64_t> X = X_in;
    std::vector<uint64_t> Z = Z_in;
    std::vector<bool> neg(X.size(), false);
    std::vector<Op> ops;

    // for each qubit: diagonalize_iter_
    for (idx i = 0; i < q; ++i) {
        diagonalize_iter(X, Z, neg, ops, i, q);
    }

    // if any(X): for each qubit i with any X[:,i], apply H on i
    uint64_t x_or = 0;
    for (auto xm : X) {
        x_or |= xm;
    }

    if (x_or != 0) {
        for (idx i = 0; i < q; ++i) {
            if (bit_u64(x_or, i)) {
                ops.push_back(Op{OpType::H, i, 0, 0});
                apply_H_on_masks(X, Z, neg, i);
            }
        }
    }

    // At this point, X should be all zero for commuting sets (diagonal in Z
    // basis). We return Z and neg, same as Python.
    return DiagCacheEntry{ops, Z, neg};
}

// ------------------------- VQE2 Circuit (port of VqeCircuit in Python)
// -------------------------

class Vqe2Circuit {
  public:
    Vqe2Circuit(idx number_of_qubits, std::vector<PauliTerm> pauli_terms,
                bool qubitwise)
        : q_(number_of_qubits), terms_(std::move(pauli_terms)),
          qubitwise_(qubitwise) {
        p_ = static_cast<int>(terms_.size());
        if (q_ > 63) {
            std::cerr << "This sketch uses uint64_t bitmasks and supports up "
                         "to 63 qubits.\n";
            std::exit(EXIT_FAILURE);
        }

        // Build X/Z bitmasks per term (like self.X, self.Z in Python)
        xmask_.assign(p_, 0);
        zmask_.assign(p_, 0);
        for (int i = 0; i < p_; ++i) {
            const auto& s = terms_[i].paulis;
            if (static_cast<idx>(s.size()) > q_) {
                std::cerr << "Pauli string longer than q: " << s << "\n";
                std::exit(EXIT_FAILURE);
            }
            for (idx j = 0; j < static_cast<idx>(s.size()); ++j) {
                const char c = s[static_cast<size_t>(j)];
                if (c == 'X') {
                    xmask_[i] |= (1ULL << j);
                } else if (c == 'Z') {
                    zmask_[i] |= (1ULL << j);
                } else if (c == 'Y') {
                    xmask_[i] |= (1ULL << j);
                    zmask_[i] |= (1ULL << j);
                }
            }
        }

        // Commutation matrix C (bool) and neighbor sets N (set of commuting
        // indices)
        build_commutation_graph();

        // Coefficients vector
        coeffs_.resize(p_);
        for (int i = 0; i < p_; ++i) {
            coeffs_[i] = terms_[i].coeff;
        }
    }

    std::vector<realT> random_parameters(int ansatz_layers = 5,
                                         uint64_t seed = 12345) const {
        // Python: np.pi * np.random.random_sample(self.q * 5)
        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<realT> dist(0, qpp::pi);
        std::vector<realT> params(static_cast<size_t>(q_ * ansatz_layers));
        for (auto& x : params) {
            x = dist(rng);
        }
        return params;
    }

    // hardware_efficient ansatz port: RY layer then CZ chain per layer
    std::vector<Op> build_ansatz_ops(const std::vector<realT>& params) const {
        if (params.empty() || (params.size() % static_cast<size_t>(q_) != 0)) {
            std::cerr << "Params must have length multiple of q.\n";
            std::exit(EXIT_FAILURE);
        }
        const idx layers =
            static_cast<idx>(params.size() / static_cast<size_t>(q_));
        std::vector<Op> ops;
        ops.reserve(
            static_cast<size_t>(layers * (q_ + (q_ > 0 ? (q_ - 1) : 0))));

        for (idx layer = 0; layer < layers; ++layer) {
            for (idx i = 0; i < q_; ++i) {
                const realT theta = params[static_cast<size_t>(layer * q_ + i)];
                ops.push_back(Op{OpType::RY, i, 0, theta});
            }
            for (idx i = 0; i + 1 < q_; ++i) {
                ops.push_back(Op{OpType::CZ, i, i + 1, 0});
            }
        }
        return ops;
    }

    // Core port: cost_function(parameters, resource, query_number)
    realT cost_function(const std::vector<realT>& parameters,
                        StatevectorResource& resource, int query_number) {
        const std::vector<Op> ansatz = build_ansatz_ops(parameters);

        // X: p x p matrix of PairCounts (Python used dicts)
        std::vector<std::vector<PairCounts>> X(
            static_cast<size_t>(p_),
            std::vector<PairCounts>(static_cast<size_t>(p_)));
        // cache D: clique_key -> diagonalizer
        std::unordered_map<std::string, DiagCacheEntry> D;

        // S: p x p int matrix (Python used numpy int matrix)
        std::vector<std::vector<int>> S(
            static_cast<size_t>(p_),
            std::vector<int>(static_cast<size_t>(p_), 0));

        // xxx: list of cliques measured since last variance update
        std::vector<std::vector<int>> xxx;

        // S[diag] += 1
        for (int i = 0; i < p_; ++i) {
            S[static_cast<size_t>(i)][static_cast<size_t>(i)] += 1;
        }

        // initial V (Python does this at i=0 via variance_estimate_ with empty
        // xxx)
        auto V = bayes_variance_graph(X, coeffs_);

        for (int i = 0; i < query_number; ++i) {
            if (is_power(i, 2)) {
                V = variance_estimate_(ansatz, resource, X, xxx, D);
                xxx.clear();
            }
            std::vector<int> aa = optimal_clique(V, S);
            xxx.push_back(aa);

            // S[np.ix_(aa,aa)] += 1
            for (int u : aa) {
                for (int v : aa) {
                    S[static_cast<size_t>(u)][static_cast<size_t>(v)] += 1;
                }
            }
        }

        // S[diag] -= 1
        for (int i = 0; i < p_; ++i) {
            S[static_cast<size_t>(i)][static_cast<size_t>(i)] -= 1;
        }

        // final variance update on leftover xxx
        V = variance_estimate_(ansatz, resource, X, xxx, D);

        // return Σ c_i * naive_Mean(X[i,i])
        realT E = 0;
        for (int i = 0; i < p_; ++i) {
            E += coeffs_[i] *
                 naive_Mean(X[static_cast<size_t>(i)][static_cast<size_t>(i)]);
        }
        return E;
    }

  private:
    idx q_;
    int p_{};
    std::vector<PauliTerm> terms_;
    bool qubitwise_{false};

    std::vector<uint64_t> xmask_; // per term
    std::vector<uint64_t> zmask_; // per term

    // N[i] = set of indices commuting with i (excluding i)
    std::vector<std::vector<int>> neighbors_;
    std::vector<realT> coeffs_;

    void build_commutation_graph() {
        neighbors_.assign(static_cast<size_t>(p_), {});
        // Build C implicitly and fill neighbors
        for (int i = 0; i < p_; ++i) {
            for (int j = 0; j < p_; ++j) {
                if (i == j) {
                    continue;
                }
                // Python C[i,j] = not sip(...)
                bool sip = sip_bitmask(xmask_[i], zmask_[i], xmask_[j],
                                       zmask_[j], qubitwise_);
                bool commute = !sip;
                if (commute) {
                    neighbors_[static_cast<size_t>(i)].push_back(j);
                }
            }
            // ensure deterministic iteration like Python set intersection
            // behavior
            std::sort(neighbors_[static_cast<size_t>(i)].begin(),
                      neighbors_[static_cast<size_t>(i)].end());
        }
    }

    // sample_(resource, aa, D, circuit)
    std::vector<int>
    sample_clique_(StatevectorResource& resource,
                   const std::vector<Op>& ansatz_ops,
                   const std::vector<int>& aa,
                   std::unordered_map<std::string, DiagCacheEntry>& D) {
        const std::string key = clique_key(aa);

        DiagCacheEntry entry;
        auto it = D.find(key);
        if (it != D.end()) {
            entry = it->second;
        } else {
            // build clique-local X/Z arrays
            std::vector<uint64_t> Xc;
            std::vector<uint64_t> Zc;
            Xc.reserve(aa.size());
            Zc.reserve(aa.size());
            for (int idx_term : aa) {
                Xc.push_back(xmask_[static_cast<size_t>(idx_term)]);
                Zc.push_back(zmask_[static_cast<size_t>(idx_term)]);
            }
            entry = diagonalize_clique(Xc, Zc, q_);
            D.emplace(key, entry);
        }

        // Execute ansatz + diagonalizer
        std::vector<Op> full_ops = ansatz_ops;
        full_ops.insert(full_ops.end(), entry.ops.begin(), entry.ops.end());

        const std::vector<bool> bits = resource(full_ops);
        const uint64_t meas_mask = bits_to_mask_u64(bits);

        // Compute outcomes for each Pauli in clique:
        // (-1) ** (neg[i] ^ parity(Z1[i] & sample))
        std::vector<int> cc;
        cc.reserve(aa.size());
        for (size_t k = 0; k < aa.size(); ++k) {
            const uint64_t z_and = entry.Z1[k] & meas_mask;
            const bool parity = (popcount_u64(z_and) & 1) != 0;
            const bool sign = entry.neg[k] ^ parity;
            cc.push_back(sign ? -1 : 1);
        }
        return cc;
    }

    // variance_estimate_(circuit, resource, X, xxx, D)
    std::vector<std::vector<realT>>
    variance_estimate_(const std::vector<Op>& ansatz_ops,
                       StatevectorResource& resource,
                       std::vector<std::vector<PairCounts>>& X,
                       const std::vector<std::vector<int>>& xxx,
                       std::unordered_map<std::string, DiagCacheEntry>& D) {
        for (const auto& aa : xxx) {
            const std::vector<int> cc =
                sample_clique_(resource, ansatz_ops, aa, D);
            // for (a0,c0),(a1,c1) in product(zip(aa,cc), repeat=2):
            for (size_t u = 0; u < aa.size(); ++u) {
                for (size_t v = 0; v < aa.size(); ++v) {
                    const int a0 = aa[u], a1 = aa[v];
                    const int c0 = cc[u], c1 = cc[v];
                    inc_pair(
                        X[static_cast<size_t>(a0)][static_cast<size_t>(a1)], c0,
                        c1);
                }
            }
        }
        return bayes_variance_graph(X, coeffs_);
    }

    // optimal_clique(V, S)
    std::vector<int>
    optimal_clique(const std::vector<std::vector<realT>>& V,
                   const std::vector<std::vector<int>>& S) const {
        const int p = p_;
        std::vector<int> aa;
        std::vector<int> aa1(p);
        std::iota(aa1.begin(), aa1.end(), 0);

        // Python:
        // V1 = V * (S * s*s^T - S1 * s1*s1^T)
        // but then cc = [V1[a1,a1] for a1 in aa1]
        // and r = argmax(cc)
        //
        // We compute V1_diag[a] = V[a,a] * (1/Saa - 1/(Saa+1)).
        auto V1_diag = [&](int a) -> realT {
            const realT Saa = static_cast<realT>(
                S[static_cast<size_t>(a)][static_cast<size_t>(a)]);
            const realT term =
                (Saa != 0) ? (1.0 / Saa - 1.0 / (Saa + 1.0)) : 0.0;
            return V[static_cast<size_t>(a)][static_cast<size_t>(a)] * term;
        };

        while (!aa1.empty()) {
            std::vector<realT> cc(aa1.size());
            realT sum_cc = 0;
            for (size_t i = 0; i < aa1.size(); ++i) {
                cc[i] = V1_diag(aa1[i]);
                sum_cc += cc[i];
            }
            if (sum_cc == 0) {
                for (auto& x : cc) {
                    x = 1;
                }
            }

            // r = aa1[argmax cc], tie-break by first occurrence
            size_t best_i = 0;
            for (size_t i = 1; i < cc.size(); ++i) {
                if (cc[i] > cc[best_i]) {
                    best_i = i;
                }
            }
            const int r = aa1[best_i];
            aa.push_back(r);

            // aa1 = N[r] ∩ aa1
            const auto& Nr = neighbors_[static_cast<size_t>(r)];
            std::vector<int> next;
            next.reserve(aa1.size());
            // intersection of sorted vectors
            size_t i = 0, j = 0;
            // Ensure aa1 is sorted (it is, due to construction/intersections)
            while (i < aa1.size() && j < Nr.size()) {
                if (aa1[i] == Nr[j]) {
                    next.push_back(aa1[i]);
                    ++i;
                    ++j;
                } else if (aa1[i] < Nr[j]) {
                    ++i;
                } else {
                    ++j;
                }
            }
            aa1 = std::move(next);
        }

        std::sort(aa.begin(), aa.end());
        return aa;
    }
};

// ------------------------- Exact dense utilities -------------------------

static qpp::cmat pauli_char_to_mat(char c) {
    switch (c) {
        case 'I':
            return qpp::gt.Id2;
        case 'X':
            return qpp::gt.X;
        case 'Y':
            return qpp::gt.Y;
        case 'Z':
            return qpp::gt.Z;
        default:
            std::cerr << "Invalid Pauli char in pauli_char_to_mat\n";
            std::exit(EXIT_FAILURE);
    }
}

static qpp::cmat pauli_string_to_matrix(const std::string& p, idx n_qubits) {
    qpp::cmat op = qpp::cmat::Ones(1, 1);
    for (idx i = 0; i < n_qubits; ++i) {
        char c =
            (i < static_cast<idx>(p.size())) ? p[static_cast<size_t>(i)] : 'I';
        op = qpp::kron(op, pauli_char_to_mat(c));
    }
    return op;
}

static qpp::cmat build_hamiltonian_matrix(const std::vector<PauliTerm>& H,
                                          idx n_qubits) {
    const idx dim = idx{1} << n_qubits;
    qpp::cmat Hm = qpp::cmat::Zero(dim, dim);
    for (const auto& term : H) {
        Hm += term.coeff * pauli_string_to_matrix(term.paulis, n_qubits);
    }
    return Hm;
}

static realT exact_ground_energy_dense(const std::vector<PauliTerm>& H,
                                       idx n_qubits) {
    qpp::cmat Hm = build_hamiltonian_matrix(H, n_qubits);
    qpp::dyn_col_vect<realT> evals = qpp::hevals(Hm);
    return evals.minCoeff();
}

static realT exact_ansatz_energy_dense(const std::vector<PauliTerm>& H,
                                       idx n_qubits,
                                       const std::vector<Op>& ansatz_ops) {
    qpp::cmat Hm = build_hamiltonian_matrix(H, n_qubits);
    ket psi0 = qpp::mket(std::vector<idx>(n_qubits, 0));
    ket psi = apply_ops(psi0, n_qubits, ansatz_ops);
    cplx val = (psi.adjoint() * (Hm * psi))(0, 0);
    return static_cast<realT>(std::real(val));
}

// ------------------------- CLI -------------------------

struct CLI {
    int shots = 1024;
    int ansatz_layers = 5; // Python uses q*5 params by default
    bool qubitwise = false;
    uint64_t seed = 12345;
    bool verbose = true;
    int test = 0; // 0 = none, else 1..4
};

static CLI parse_cli(int argc, char** argv) {
    CLI cli;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--shots" && i + 1 < argc) {
            cli.shots = std::stoi(argv[++i]);
        } else if (a == "--ansatz_layers" && i + 1 < argc) {
            cli.ansatz_layers = std::stoi(argv[++i]);
        } else if (a == "--qubitwise") {
            cli.qubitwise = true;
        } else if (a == "--seed" && i + 1 < argc) {
            cli.seed = static_cast<uint64_t>(std::stoull(argv[++i]));
        } else if (a == "--test" && i + 1 < argc) {
            cli.test = std::stoi(argv[++i]);
        } else if (a == "--quiet") {
            cli.verbose = false;
        } else if (a == "--help" || a == "-h") {
            std::cerr
                << "vqe2 (port of vqe.py cost_function)\n\n"
                << "Reads Hamiltonian from stdin: lines '<coeff> <paulis>'\n"
                << "Options:\n"
                << "  --shots N           query_number (total measurement "
                   "calls)\n"
                << "  --ansatz_layers L   layers for hardware_efficient "
                   "(default 5)\n"
                << "  --qubitwise         use qubitwise commutation (early "
                   "exit) like vqe.py\n"
                << "  --seed S            RNG seed\n"
                << "  --test K            run built-in test Hamiltonian "
                   "K=1..5\n"
                << "  --quiet             suppress prints\n";
            std::exit(EXIT_SUCCESS);
        } else {
            std::cerr << "Unknown option: " << a << " (try --help)\n";
            std::exit(EXIT_FAILURE);
        }
    }
    return cli;
}

// ------------------------- main -------------------------

int main(int argc, char** argv) {
    CLI cli = parse_cli(argc, argv);

    idx n_qubits = 0;
    std::vector<PauliTerm> H;
    std::string test_name;
    std::string test_desc;

    if (cli.test != 0) {
        TestCase tc = get_testcase(cli.test);
        n_qubits = tc.n_qubits;
        H = tc.H;
        test_name = tc.name;
        test_desc = tc.description;
        if (cli.ansatz_layers == 5) {
            cli.ansatz_layers = tc.suggested_layers;
        }
        if (cli.shots == 1024) {
            cli.shots = tc.suggested_shots;
        }
    } else {
        H = read_hamiltonian_from_stdin(n_qubits);
    }

    if (cli.verbose) {
        std::cerr << ">> vqe2.cpp (port of vqe.py cost_function)\n";
        std::cerr << ">> Terms: " << H.size() << ", Qubits: " << n_qubits
                  << ", shots: " << cli.shots
                  << ", ansatz_layers: " << cli.ansatz_layers
                  << ", qubitwise: " << (cli.qubitwise ? "true" : "false")
                  << "\n";
        if (cli.test != 0) {
            std::cerr << ">> Test " << cli.test << ": " << test_name << "\n";
            std::cerr << ">> " << test_desc << "\n";
        }
    }

    Vqe2Circuit circ(n_qubits, H, cli.qubitwise);
    std::vector<realT> params =
        circ.random_parameters(cli.ansatz_layers, cli.seed);
    const std::vector<Op> ansatz_ops = circ.build_ansatz_ops(params);

    StatevectorResource resource(n_qubits, cli.seed ^ 0x9E3779B97F4A7C15ULL);

    realT E_gs = std::numeric_limits<realT>::quiet_NaN();
    realT E_ansatz_exact = std::numeric_limits<realT>::quiet_NaN();

    if (n_qubits <= 12) {
        E_gs = exact_ground_energy_dense(H, n_qubits);
        E_ansatz_exact = exact_ansatz_energy_dense(H, n_qubits, ansatz_ops);
        if (cli.verbose) {
            std::cerr << std::setprecision(12);
            std::cerr << ">> Exact ground energy (dense): " << E_gs << "\n";
            std::cerr << ">> Exact ansatz energy  (dense): " << E_ansatz_exact
                      << "\n";
        }
    } else if (cli.verbose) {
        std::cerr << ">> Skipping dense exact energies for n_qubits > 12\n";
    }

    const realT E_est = circ.cost_function(params, resource, cli.shots);

    if (cli.verbose) {
        std::cerr << std::setprecision(12);
        std::cerr << ">> vqe2 estimator energy:         " << E_est << "\n";
        if (n_qubits <= 12) {
            std::cerr << ">> |est - exact(ansatz)|:         "
                      << std::abs(E_est - E_ansatz_exact) << "\n";
            std::cerr << ">> (ref) exact ground energy:     " << E_gs << "\n";
            std::cerr << ">> (ref) gap to ground (est):     " << (E_est - E_gs)
                      << "\n";
            std::cerr << ">> (ref) gap to ground (ansatz):  "
                      << (E_ansatz_exact - E_gs) << "\n";
        }
    }

    std::cout << std::setprecision(12) << E_est << "\n";
    return 0;
}
