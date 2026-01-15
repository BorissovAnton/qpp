// Source: ./examples/vqe.cpp
//
// Variational Quantum Eigensolver (VQE) for finding ground state energy
// Demonstrates VQE on the H2 molecule Hamiltonian using a hardware-efficient
// ansatz with gradient descent optimization via the parameter shift rule.

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unistd.h>
#include <vector>

#include <nlohmann/json.hpp>
#include <qpp/qpp.hpp>

using json = nlohmann::json;

// Maximum qubits for exact diagonalization (for reference energy)
// 12 qubits = 4096x4096 matrix = ~256MB, diagonalization feasible
constexpr qpp::idx MAX_QUBITS_EXACT_DIAG = 12;

// Maximum qubits for VQE simulation (state vector only, no full matrix)
// 20 qubits = 2^20 complex doubles = ~16MB state vector
constexpr qpp::idx MAX_QUBITS_VQE = 20;

// CLI options
struct CLIOptions {
    qpp::idx layers = 3;
    qpp::idx max_iter = 100;
    qpp::realT learning_rate = 0.5;
    qpp::idx restarts = 1;
    bool json_output = false;
    bool quiet = false;
    bool show_help = false;
    bool use_default_h2 = false;        // --demo flag to use default H2
    int test_case = 0;                  // --test N to run hardcoded test
    bool layers_explicitly_set = false; // Track if user set --layers
    bool use_ring_entangler = false;    // --ring to close CZ chain
    bool alternate_rotations = false;   // --alt-rot to alternate RY/RZ
};

void print_usage(const char* program_name) {
    std::cerr
        << "Usage: " << program_name << " [options]\n"
        << "\nReads Hamiltonian from stdin in one of two formats:\n"
        << "  Format 1 (long):   -1.432 IIIIIIXYZIIIIXIII\n"
        << "  Format 2 (sparse): -1.432 X6 Y7 Z8 X13\n"
        << "\nOptions:\n"
        << "  --layers N    Ansatz layers (default: 3)\n"
        << "  --max-iter N  Max iterations (default: 100)\n"
        << "  --lr RATE     Learning rate (default: 0.5)\n"
        << "  --restarts N  Multi-start optimization runs (default: 1)\n"
        << "  --ring        Add CZ between last and first qubit\n"
        << "  --alt-rot     Alternate RY/RZ rotations per layer\n"
        << "  --json        Output JSON only\n"
        << "  --quiet       Suppress progress output\n"
        << "  --demo        Use default H2 Hamiltonian (no stdin required)\n"
        << "  --test N      Run hardcoded test case N (1-4)\n"
        << "  --help        Show this help\n"
        << "\nTest cases:\n"
        << "  1: 2-qubit H2 molecule (converges well)\n"
        << "  2: 4-qubit H2 molecule (shows ansatz limitations)\n"
        << "  3: 4-qubit Heisenberg XXZ chain (antiferromagnetic)\n"
        << "  4: 6-qubit transverse-field Ising model\n"
        << "\nMax qubits: " << MAX_QUBITS_VQE << " for VQE, "
        << MAX_QUBITS_EXACT_DIAG << " for exact diagonalization\n";
}

CLIOptions parse_args(int argc, char** argv) {
    CLIOptions opts;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            opts.show_help = true;
        } else if (arg == "--json") {
            opts.json_output = true;
        } else if (arg == "--quiet" || arg == "-q") {
            opts.quiet = true;
        } else if (arg == "--demo") {
            opts.use_default_h2 = true;
        } else if (arg == "--test" && i + 1 < argc) {
            opts.test_case = std::stoi(argv[++i]);
        } else if (arg == "--layers" && i + 1 < argc) {
            opts.layers = static_cast<qpp::idx>(std::stoul(argv[++i]));
            opts.layers_explicitly_set = true;
        } else if (arg == "--max-iter" && i + 1 < argc) {
            opts.max_iter = static_cast<qpp::idx>(std::stoul(argv[++i]));
        } else if (arg == "--lr" && i + 1 < argc) {
            opts.learning_rate = std::stod(argv[++i]);
        } else if (arg == "--restarts" && i + 1 < argc) {
            opts.restarts = static_cast<qpp::idx>(std::stoul(argv[++i]));
        } else if (arg == "--ring") {
            opts.use_ring_entangler = true;
        } else if (arg == "--alt-rot") {
            opts.alternate_rotations = true;
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            opts.show_help = true;
        }
    }
    return opts;
}

// Pauli term: coefficient * (tensor product of Paulis)
// e.g., {0.5, "XZI"} represents 0.5 * X_0 ⊗ Z_1 ⊗ I_2
struct PauliTerm {
    qpp::realT coefficient;
    std::string paulis; // "I", "X", "Y", "Z" per qubit
};

using Hamiltonian = std::vector<PauliTerm>;

// Check if a string contains only valid Pauli characters (long format)
bool is_valid_pauli_string(const std::string& s) {
    for (char c : s) {
        if (c != 'I' && c != 'X' && c != 'Y' && c != 'Z') {
            return false;
        }
    }
    return !s.empty();
}

// Check if the pauli part uses sparse format (e.g., "X0 Z2 Y3")
// Sparse format has pattern: letter followed by digit(s)
bool is_sparse_format(const std::string& pauli_part) {
    // If it's a valid long-form Pauli string, it's not sparse
    if (is_valid_pauli_string(pauli_part)) {
        return false;
    }
    // Check for pattern like "X0" or "Z12"
    for (size_t i = 0; i < pauli_part.size(); ++i) {
        char c = pauli_part[i];
        if (c == 'X' || c == 'Y' || c == 'Z') {
            if (i + 1 < pauli_part.size() && std::isdigit(pauli_part[i + 1])) {
                return true;
            }
        }
    }
    return false;
}

// Parse sparse format "X0 Z2 Y3" and return {pauli_string, max_qubit_index}
// Returns empty string on parse error
std::pair<std::string, qpp::idx>
parse_sparse_format(const std::string& sparse) {
    std::istringstream iss(sparse);
    std::string token;
    qpp::idx max_qubit = 0;
    std::vector<std::pair<qpp::idx, char>> terms;

    while (iss >> token) {
        if (token.empty()) {
            continue;
        }
        char pauli = token[0];
        if (pauli != 'X' && pauli != 'Y' && pauli != 'Z') {
            return {"", 0}; // Invalid Pauli character
        }
        if (token.size() < 2) {
            return {"", 0}; // Missing qubit index
        }
        try {
            qpp::idx qubit = static_cast<qpp::idx>(std::stoul(token.substr(1)));
            terms.push_back({qubit, pauli});
            max_qubit = std::max(max_qubit, qubit);
        } catch (...) {
            return {"", 0}; // Invalid number
        }
    }

    if (terms.empty()) {
        return {"", 0};
    }

    // Build the Pauli string (all I's, then fill in specified qubits)
    std::string paulis(max_qubit + 1, 'I');
    for (const auto& [qubit, pauli] : terms) {
        paulis[qubit] = pauli;
    }

    return {paulis, max_qubit};
}

// Parse a single line of Hamiltonian input
// Returns {success, coefficient, pauli_string, max_qubit_index}
struct ParseResult {
    bool success;
    qpp::realT coefficient;
    std::string paulis;
    qpp::idx max_qubit;
    std::string error_msg;
};

ParseResult parse_hamiltonian_line(const std::string& line) {
    ParseResult result{false, 0.0, "", 0, ""};

    // Skip empty lines and comments
    std::string trimmed = line;
    // Trim leading whitespace
    size_t start = trimmed.find_first_not_of(" \t\r\n");
    if (start == std::string::npos || trimmed[start] == '#') {
        result.success = true; // Empty/comment line is valid, just skip
        result.paulis = "";    // Signal to skip this line
        return result;
    }
    trimmed = trimmed.substr(start);

    // Parse coefficient and pauli part
    std::istringstream iss(trimmed);
    std::string coeff_str;
    iss >> coeff_str;

    try {
        result.coefficient = std::stod(coeff_str);
    } catch (...) {
        result.error_msg = "Invalid coefficient: " + coeff_str;
        return result;
    }

    // Get the rest of the line as the Pauli specification
    std::string pauli_part;
    std::getline(iss, pauli_part);
    // Trim leading whitespace from pauli_part
    start = pauli_part.find_first_not_of(" \t");
    if (start == std::string::npos) {
        result.error_msg = "Missing Pauli operators after coefficient";
        return result;
    }
    pauli_part = pauli_part.substr(start);

    // Determine format and parse
    if (is_sparse_format(pauli_part)) {
        auto [paulis, max_q] = parse_sparse_format(pauli_part);
        if (paulis.empty()) {
            result.error_msg = "Invalid sparse format: " + pauli_part;
            return result;
        }
        result.paulis = paulis;
        result.max_qubit = max_q;
    } else if (is_valid_pauli_string(pauli_part)) {
        result.paulis = pauli_part;
        result.max_qubit = static_cast<qpp::idx>(pauli_part.size() - 1);
    } else {
        result.error_msg = "Invalid Pauli string: " + pauli_part;
        return result;
    }

    result.success = true;
    return result;
}

// Read Hamiltonian from stdin, returns empty Hamiltonian on error
// Sets n_qubits to the detected number of qubits
// Sets error_msg if parsing fails
Hamiltonian read_hamiltonian_stdin(qpp::idx& n_qubits, std::string& error_msg) {
    Hamiltonian H;
    n_qubits = 0;
    error_msg = "";

    std::string line;
    int line_num = 0;

    while (std::getline(std::cin, line)) {
        ++line_num;
        auto result = parse_hamiltonian_line(line);

        if (!result.success) {
            error_msg =
                "Line " + std::to_string(line_num) + ": " + result.error_msg;
            return {};
        }

        // Skip empty/comment lines
        if (result.paulis.empty()) {
            continue;
        }

        // Track max qubits
        qpp::idx term_qubits = static_cast<qpp::idx>(result.paulis.size());
        n_qubits = std::max(n_qubits, term_qubits);

        H.push_back({result.coefficient, result.paulis});
    }

    // Normalize all Pauli strings to the same length (pad with I)
    for (auto& term : H) {
        while (term.paulis.size() < n_qubits) {
            term.paulis += 'I';
        }
    }

    if (H.empty()) {
        error_msg = "No valid Hamiltonian terms found in input";
    }

    return H;
}

// Convert a single Pauli character to its matrix representation
// Throws on invalid input to catch bugs early
qpp::cmat pauli_to_matrix(char p) {
    using namespace qpp;
    switch (p) {
        case 'X':
            return gt.X;
        case 'Y':
            return gt.Y;
        case 'Z':
            return gt.Z;
        case 'I':
            return gt.Id2;
        default:
            throw std::invalid_argument(
                std::string("Invalid Pauli character: '") + p + "'");
    }
}

// Convert a Pauli string to a matrix operator
// e.g., "XZI" -> X ⊗ Z ⊗ I (for 3 qubits)
// WARNING: This builds a full 2^n x 2^n matrix - only use for small n
qpp::cmat pauli_string_to_matrix(const std::string& paulis, qpp::idx n_qubits) {
    using namespace qpp;
    cmat result = cmat::Identity(1, 1);
    for (idx i = 0; i < n_qubits; ++i) {
        char p = (i < paulis.size()) ? paulis[i] : 'I';
        result = kron(result, pauli_to_matrix(p));
    }
    return result;
}

// Build the full Hamiltonian matrix from Pauli terms
// WARNING: O(4^n) memory - only use for small n (exact diagonalization)
qpp::cmat build_hamiltonian_matrix(const Hamiltonian& H, qpp::idx n_qubits) {
    using namespace qpp;
    idx dim = idx{1} << n_qubits; // 2^n using bit-shift (safe, no float issues)
    cmat H_mat = cmat::Zero(dim, dim);
    for (const auto& term : H) {
        H_mat +=
            term.coefficient * pauli_string_to_matrix(term.paulis, n_qubits);
    }
    return H_mat;
}

// =============================================================================
// Matrix-free Pauli string application (for scalable VQE)
// =============================================================================

// Apply a Pauli string to a state vector WITHOUT building the full matrix
// This is O(n * 2^n) instead of O(4^n) for building the matrix
// Returns P|psi> where P is the tensor product of Paulis
qpp::ket apply_pauli_string(const qpp::ket& psi, const std::string& paulis,
                            qpp::idx n_qubits) {
    using namespace qpp;
    ket result = psi;

    for (idx q = 0; q < n_qubits; ++q) {
        char p = (q < paulis.size()) ? paulis[q] : 'I';
        if (p == 'I') {
            continue; // Identity does nothing
        }
        // Apply single-qubit Pauli gate to qubit q
        // qpp::apply() is efficient - doesn't build full matrix
        result = apply(result, pauli_to_matrix(p), {q});
    }
    return result;
}

// Compute expectation value <psi|P|psi> without building full Pauli matrix
// This is the key optimization for scalable VQE
qpp::realT expectation_value_pauli_string(const qpp::ket& psi,
                                          const std::string& paulis,
                                          qpp::idx n_qubits) {
    using namespace qpp;
    // Compute P|psi>
    ket P_psi = apply_pauli_string(psi, paulis, n_qubits);
    // Compute <psi|P|psi> = psi^dagger * (P * psi)
    cplx inner = (adjoint(psi) * P_psi).value();
    return std::real(inner); // Pauli expectation values are always real
}

// Apply hardware-efficient ansatz: layers of rotations followed by CZ gates
qpp::ket apply_hardware_efficient_ansatz(const std::vector<qpp::realT>& params,
                                         qpp::idx n_qubits, qpp::idx layers,
                                         bool use_ring_entangler,
                                         bool alternate_rotations) {
    using namespace qpp;
    // Start with |00...0>
    ket psi = mket(std::vector<idx>(n_qubits, 0));

    idx param_idx = 0;
    for (idx layer = 0; layer < layers; ++layer) {
        // Single-qubit rotations
        for (idx q = 0; q < n_qubits; ++q) {
            if (alternate_rotations) {
                if (layer % 2 == 0) {
                    psi = apply(psi, gt.RY(params[param_idx++]), {q});
                } else {
                    psi = apply(psi, gt.RZ(params[param_idx++]), {q});
                }
            } else {
                psi = apply(psi, gt.RY(params[param_idx++]), {q});
            }
        }
        // CZ entangling gates between adjacent qubits
        for (idx q = 0; q + 1 < n_qubits; ++q) {
            psi = apply(psi, gt.CZ, {q, q + 1});
        }
        // Optional ring entangler: connect last to first
        if (use_ring_entangler && n_qubits > 2) {
            psi = apply(psi, gt.CZ, {n_qubits - 1, 0});
        }
    }
    return psi;
}

// Compute exact expectation value: <psi|O|psi>
// NOTE: This requires building the full observable matrix - prefer
// expectation_value_pauli_string() for Pauli operators
[[maybe_unused]]
qpp::realT expectation_value_exact(const qpp::ket& psi,
                                   const qpp::cmat& observable) {
    using namespace qpp;
    return std::real((adjoint(psi) * observable * psi).value());
}

// Compute expectation value of a single Pauli string via sampling
// Measures in the appropriate basis for each qubit based on Pauli type
// NOTE: This has correctness issues with non-qubitwise-commuting terms
// (cannot simultaneously diagonalize XX and ZZ without entangling gates)
// Currently unused - kept for reference/future improvement
[[maybe_unused]]
qpp::realT expectation_value_sampling(const qpp::ket& psi,
                                      const std::string& paulis,
                                      qpp::idx n_qubits, qpp::idx n_samples) {
    using namespace qpp;

    // Transform state to measurement basis for each qubit
    // X -> apply H (measures in X basis)
    // Y -> apply S†H (measures in Y basis)
    // Z -> no change (computational basis)
    // I -> no measurement needed (always contributes +1)
    ket psi_transformed = psi;
    std::vector<idx> qubits_to_measure;

    for (idx q = 0; q < n_qubits; ++q) {
        char p = (q < paulis.size()) ? paulis[q] : 'I';
        if (p == 'X') {
            psi_transformed = apply(psi_transformed, gt.H, {q});
            qubits_to_measure.push_back(q);
        } else if (p == 'Y') {
            // S† = [[1,0],[0,-i]], then H
            psi_transformed = apply(psi_transformed, adjoint(gt.S), {q});
            psi_transformed = apply(psi_transformed, gt.H, {q});
            qubits_to_measure.push_back(q);
        } else if (p == 'Z') {
            qubits_to_measure.push_back(q);
        }
        // 'I' qubits don't need measurement
    }

    // If no qubits to measure (all identity), expectation is 1
    if (qubits_to_measure.empty()) {
        return 1.0;
    }

    // Sample and compute average
    realT sum = 0.0;
    for (idx sample = 0; sample < n_samples; ++sample) {
        // IMPORTANT: Copy state before measurement to avoid mutating the
        // original measure_seq may collapse the state in-place depending on qpp
        // version
        ket psi_copy = psi_transformed;

        // Measure the relevant qubits
        auto [results, probs, final_state] =
            measure_seq(psi_copy, qubits_to_measure);

        // Compute parity: (-1)^(sum of measurement outcomes)
        idx parity = 0;
        for (auto r : results) {
            parity ^= r;
        }
        sum += (parity == 0) ? 1.0 : -1.0;
    }
    return sum / static_cast<realT>(n_samples);
}

// Evaluate total energy for given parameters using matrix-free approach
// This is O(n_terms * n_qubits * 2^n_qubits) instead of O(n_terms * 4^n_qubits)
qpp::realT evaluate_energy(const Hamiltonian& H,
                           const std::vector<qpp::realT>& params,
                           qpp::idx n_qubits, qpp::idx layers,
                           bool use_ring_entangler, bool alternate_rotations,
                           [[maybe_unused]] bool use_sampling = false,
                           [[maybe_unused]] qpp::idx n_samples = 1000) {
    using namespace qpp;
    ket psi = apply_hardware_efficient_ansatz(
        params, n_qubits, layers, use_ring_entangler, alternate_rotations);

    realT energy = 0.0;
    for (const auto& term : H) {
        // Use matrix-free expectation value computation
        energy += term.coefficient *
                  expectation_value_pauli_string(psi, term.paulis, n_qubits);
    }
    return energy;
}

// Compute gradient using parameter shift rule
// grad_i = [E(theta_i + pi/2) - E(theta_i - pi/2)] / 2
std::vector<qpp::realT>
parameter_shift_gradient(const Hamiltonian& H,
                         const std::vector<qpp::realT>& params,
                         qpp::idx n_qubits, qpp::idx layers,
                         bool use_ring_entangler, bool alternate_rotations,
                         bool use_sampling = false, qpp::idx n_samples = 1000) {
    using namespace qpp;
    const realT shift = pi / 2.0;

    std::vector<realT> grad(params.size());
    for (idx i = 0; i < params.size(); ++i) {
        auto params_plus = params;
        auto params_minus = params;
        params_plus[i] += shift;
        params_minus[i] -= shift;

        realT E_plus = evaluate_energy(H, params_plus, n_qubits, layers,
                                       use_ring_entangler, alternate_rotations,
                                       use_sampling, n_samples);
        realT E_minus = evaluate_energy(H, params_minus, n_qubits, layers,
                                        use_ring_entangler, alternate_rotations,
                                        use_sampling, n_samples);
        grad[i] = (E_plus - E_minus) / 2.0;
    }
    return grad;
}

// Compute L2 norm of gradient
qpp::realT gradient_norm(const std::vector<qpp::realT>& grad) {
    qpp::realT sum = 0.0;
    for (auto g : grad) {
        sum += g * g;
    }
    return std::sqrt(sum);
}

// Compute Hamiltonian scale (sum of absolute coefficients)
qpp::realT hamiltonian_scale(const Hamiltonian& H) {
    qpp::realT scale = 0.0;
    for (const auto& term : H) {
        scale += std::abs(term.coefficient);
    }
    return scale;
}

// Run a single VQE optimization with gradient descent and backtracking
std::tuple<std::vector<qpp::realT>, qpp::realT, qpp::idx> optimize_vqe_once(
    const Hamiltonian& H, qpp::idx n_qubits, qpp::idx layers,
    qpp::idx max_iterations, qpp::realT learning_rate, bool use_ring_entangler,
    bool alternate_rotations, qpp::realT convergence_threshold = 1e-6,
    bool use_sampling = false, qpp::idx n_samples = 1000, bool verbose = true) {
    using namespace qpp;

    // Random initial parameters in [0, 2*pi)
    idx n_params = n_qubits * layers;
    std::vector<realT> params(n_params);
    for (auto& p : params) {
        p = rand<realT>(0.0, 2.0 * pi);
    }

    realT energy =
        evaluate_energy(H, params, n_qubits, layers, use_ring_entangler,
                        alternate_rotations, use_sampling, n_samples);
    idx iter = 0;

    realT scale = hamiltonian_scale(H);
    realT base_lr = learning_rate / std::max<realT>(scale, 1.0);

    for (; iter < max_iterations; ++iter) {
        auto grad = parameter_shift_gradient(
            H, params, n_qubits, layers, use_ring_entangler,
            alternate_rotations, use_sampling, n_samples);
        realT grad_norm = gradient_norm(grad);

        if (verbose && iter % 10 == 0) {
            std::cout << "Iter " << std::setw(3) << iter
                      << ": E = " << std::fixed << std::setprecision(6)
                      << energy << ", |grad| = " << std::scientific
                      << std::setprecision(2) << grad_norm << '\n';
        }

        // Check convergence
        if (grad_norm < convergence_threshold) {
            if (verbose) {
                std::cout << ">> Converged at iteration " << iter << '\n';
            }
            break;
        }

        // Backtracking line search to enforce monotone energy decrease
        realT step = base_lr;
        bool accepted = false;
        std::vector<realT> trial_params(params.size());
        realT trial_energy = energy;

        for (int bt = 0; bt < 10; ++bt) {
            for (idx i = 0; i < params.size(); ++i) {
                trial_params[i] = params[i] - step * grad[i];
            }

            trial_energy = evaluate_energy(
                H, trial_params, n_qubits, layers, use_ring_entangler,
                alternate_rotations, use_sampling, n_samples);

            if (trial_energy <= energy) {
                accepted = true;
                break;
            }
            step *= 0.5;
        }

        if (accepted) {
            params = std::move(trial_params);
            energy = trial_energy;
        } else {
            base_lr *= 0.5;
        }
    }

    return {params, energy, iter};
}

// Run VQE optimization with multi-start (restarts)
std::tuple<std::vector<qpp::realT>, qpp::realT, qpp::idx>
optimize_vqe(const Hamiltonian& H, qpp::idx n_qubits, qpp::idx layers,
             qpp::idx max_iterations, qpp::realT learning_rate,
             qpp::idx restarts, bool use_ring_entangler,
             bool alternate_rotations, qpp::realT convergence_threshold = 1e-6,
             bool use_sampling = false, qpp::idx n_samples = 1000,
             bool verbose = true) {
    using namespace qpp;

    realT best_energy = std::numeric_limits<realT>::infinity();
    std::vector<realT> best_params;
    idx best_iters = 0;

    for (idx r = 0; r < restarts; ++r) {
        if (verbose && restarts > 1) {
            std::cout << ">> Restart " << (r + 1) << "/" << restarts << '\n';
        }
        auto [params, energy, iters] = optimize_vqe_once(
            H, n_qubits, layers, max_iterations, learning_rate,
            use_ring_entangler, alternate_rotations, convergence_threshold,
            use_sampling, n_samples, verbose);
        if (energy < best_energy) {
            best_energy = energy;
            best_params = std::move(params);
            best_iters = iters;
        }
    }

    return {best_params, best_energy, best_iters};
}

// =============================================================================
// Test Hamiltonians
// =============================================================================

// Test 1: H2 molecule Hamiltonian (2 qubits)
// Using STO-3G basis at equilibrium bond length (~0.735 Angstrom)
// H = g0*I + g1*Z0 + g2*Z1 + g3*Z0Z1 + g4*X0X1 + g5*Y0Y1
// Expected ground state: ~-1.85 Ha
// This should converge well with the hardware-efficient ansatz.
Hamiltonian create_h2_hamiltonian() {
    return {
        {-0.4804, "II"}, // Constant/identity term
        {+0.3435, "ZI"}, // Z on qubit 0
        {-0.4347, "IZ"}, // Z on qubit 1
        {+0.5716, "ZZ"}, // ZZ interaction
        {+0.0910, "XX"}, // XX interaction
        {+0.0910, "YY"}  // YY interaction
    };
}

// Test 2: H2 molecule Hamiltonian (4 qubits, Jordan-Wigner)
// Full 4-qubit representation with fermionic terms
// Expected ground state: ~-1.86 Ha (FCI energy)
// This demonstrates ansatz expressibility limitations - the hardware-efficient
// ansatz struggles to reach chemical accuracy due to the fermionic structure.
Hamiltonian create_h2_4qubit_hamiltonian() {
    return {
        {-0.8126179630, "IIII"}, // Identity
        {+0.1711977490, "ZIII"}, // One-body terms
        {-0.2227965370, "IZII"},
        {+0.1711977490, "IIZI"},
        {-0.2227965370, "IIIZ"},
        {+0.1686221915, "ZZII"}, // Two-body ZZ terms
        {+0.1205448220, "IZZI"},
        {+0.1659278215, "ZIIZ"},
        {+0.1205448220, "IZIZ"},
        {+0.1743478830, "IIZZ"},
        {+0.1205448220, "ZIZI"},
        {-0.0453026155,
         "XXYY"}, // Exchange terms (hard for HW-efficient ansatz)
        {-0.0453026155, "YYXX"},
        {+0.0453026155, "XYYX"},
        {+0.0453026155, "YXXY"},
    };
}

// Test 3: 4-qubit Heisenberg XXZ chain (antiferromagnetic)
// H = J * sum_i (X_i X_{i+1} + Y_i Y_{i+1} + Delta * Z_i Z_{i+1})
// With J = 1.0, Delta = 1.0 (isotropic XXX model), periodic boundary
// Expected ground state: ~-8.0 (for 4 sites with PBC)
// This is a canonical spin model that VQE should handle reasonably well.
Hamiltonian create_heisenberg_xxz_hamiltonian() {
    qpp::realT J = 1.0;
    qpp::realT Delta = 1.0; // Anisotropy parameter
    return {
        // XX terms (nearest neighbor)
        {J, "XXII"},
        {J, "IXXI"},
        {J, "IIXX"},
        {J, "XIIX"}, // PBC: last connects to first
        // YY terms
        {J, "YYII"},
        {J, "IYYI"},
        {J, "IIYY"},
        {J, "YIIY"},
        // ZZ terms
        {J * Delta, "ZZII"},
        {J * Delta, "IZZI"},
        {J * Delta, "IIZZ"},
        {J * Delta, "ZIIZ"},
    };
}

// Test 4: 6-qubit transverse-field Ising model
// H = -J * sum_i Z_i Z_{i+1} - h * sum_i X_i
// At the critical point h/J ~ 1, this is a challenging test
// Expected to show interesting behavior near the phase transition
Hamiltonian create_tfim_hamiltonian() {
    qpp::realT J = 1.0; // Ising coupling
    qpp::realT h = 1.0; // Transverse field (critical point)
    return {
        // ZZ interactions (open boundary conditions)
        // Qubits: 0  1  2  3  4  5
        {-J, "ZZIIII"}, // Z0 Z1
        {-J, "IZZIII"}, // Z1 Z2
        {-J, "IIZZII"}, // Z2 Z3
        {-J, "IIIZZI"}, // Z3 Z4
        {-J, "IIIIZZ"}, // Z4 Z5
        // Transverse field X terms on each qubit
        {-h, "XIIIII"}, // X0
        {-h, "IXIIII"}, // X1
        {-h, "IIXIII"}, // X2
        {-h, "IIIXII"}, // X3
        {-h, "IIIIXI"}, // X4
        {-h, "IIIIIX"}, // X5
    };
}

// Get test Hamiltonian by number, returns {Hamiltonian, n_qubits, description}
struct TestCase {
    Hamiltonian H;
    qpp::idx n_qubits;
    std::string name;
    std::string description;
    qpp::idx recommended_layers;
};

TestCase get_test_case(int test_num) {
    switch (test_num) {
        case 1:
            return {create_h2_hamiltonian(), 2, "2-qubit H2 molecule",
                    "Simple H2 in minimal basis. Should converge to ~-1.85 Ha "
                    "with high accuracy.",
                    3};
        case 2:
            return {create_h2_4qubit_hamiltonian(), 4,
                    "4-qubit H2 molecule (Jordan-Wigner)",
                    "Full H2 with fermionic terms. Hardware-efficient ansatz "
                    "struggles here - "
                    "expect ~18 mHa error due to ansatz expressibility "
                    "limitations.",
                    6};
        case 3:
            return {create_heisenberg_xxz_hamiltonian(), 4,
                    "4-qubit Heisenberg XXZ chain",
                    "Antiferromagnetic spin chain with periodic boundaries. "
                    "Tests VQE on a canonical condensed matter model.",
                    4};
        case 4:
            return {
                create_tfim_hamiltonian(), 6,
                "6-qubit transverse-field Ising model",
                "TFIM at critical point h/J=1. Larger system to test scaling. "
                "May require more layers and iterations.",
                6};
        default:
            return {{}, 0, "", "", 0};
    }
}

// Output error as JSON and exit
void json_error_exit(const std::string& error_msg) {
    json result;
    result["error"] = error_msg;
    std::cout << result.dump(2) << std::endl;
    std::exit(EXIT_FAILURE);
}

int main(int argc, char** argv) {
    using namespace qpp;

    // Parse CLI arguments
    CLIOptions opts = parse_args(argc, argv);

    if (opts.show_help) {
        print_usage(argv[0]);
        return EXIT_SUCCESS;
    }

    if (opts.restarts == 0) {
        if (opts.json_output) {
            json_error_exit("--restarts must be >= 1");
        } else {
            std::cerr << "Error: --restarts must be >= 1\n";
            return EXIT_FAILURE;
        }
    }

    Hamiltonian H;
    idx n_qubits;
    std::string test_name;
    std::string test_description;

    if (opts.test_case > 0) {
        // Run a hardcoded test case
        TestCase tc = get_test_case(opts.test_case);
        if (tc.n_qubits == 0) {
            if (opts.json_output) {
                json_error_exit("Invalid test case number. Use 1-4.");
            } else {
                std::cerr
                    << "Error: Invalid test case. Use --test 1, 2, 3, or 4.\n";
                return EXIT_FAILURE;
            }
        }
        H = tc.H;
        n_qubits = tc.n_qubits;
        test_name = tc.name;
        test_description = tc.description;
        // Use recommended layers if user didn't explicitly set --layers
        if (!opts.layers_explicitly_set) {
            opts.layers = tc.recommended_layers;
        }
    } else if (opts.use_default_h2) {
        // Use default H2 Hamiltonian for testing/demo
        H = create_h2_hamiltonian();
        n_qubits = 2;
    } else {
        // Read Hamiltonian from stdin
        std::string error_msg;
        H = read_hamiltonian_stdin(n_qubits, error_msg);
        if (!error_msg.empty()) {
            if (opts.json_output) {
                json_error_exit(error_msg);
            } else {
                std::cerr << "Error: " << error_msg << "\n";
                return EXIT_FAILURE;
            }
        }
        if (H.empty()) {
            // Input was all comments/empty lines
            if (opts.json_output) {
                json_error_exit("No valid Hamiltonian terms found in input");
            } else {
                std::cerr
                    << "Error: No valid Hamiltonian terms found in input\n";
                std::cerr
                    << "Use --demo, --test N, or pipe Hamiltonian via stdin\n";
                return EXIT_FAILURE;
            }
        }
    }

    // Check qubit limit for VQE (state vector must fit in memory)
    if (n_qubits > MAX_QUBITS_VQE) {
        // Use ldexp for safe memory estimate (avoids shift overflow UB)
        double state_mb = std::ldexp(16.0, static_cast<int>(n_qubits) - 20);
        std::ostringstream oss;
        oss << "Hamiltonian has " << n_qubits << " qubits, exceeds limit of "
            << MAX_QUBITS_VQE << " (state vector would require ~" << std::fixed
            << std::setprecision(0) << state_mb << " MB)";
        if (opts.json_output) {
            json_error_exit(oss.str());
        } else {
            std::cerr << "Error: " << oss.str() << "\n";
            return EXIT_FAILURE;
        }
    }

    idx layers = opts.layers;
    idx max_iterations = opts.max_iter;
    realT learning_rate = opts.learning_rate;
    bool verbose = !opts.quiet && !opts.json_output;

    // Compute exact ground state energy for reference (only for small systems)
    // For larger systems, skip exact diagonalization as it's O(2^3n)
    bool have_exact_reference = (n_qubits <= MAX_QUBITS_EXACT_DIAG);
    realT exact_ground_state = 0.0;
    std::vector<realT> eigenvalues;

    if (have_exact_reference) {
        cmat H_mat = build_hamiltonian_matrix(H, n_qubits);
        dyn_col_vect<realT> evals = hevals(H_mat);
        exact_ground_state = evals.minCoeff();
        for (Eigen::Index i = 0; i < evals.size(); ++i) {
            eigenvalues.push_back(evals[i]);
        }
    }

    if (verbose) {
        std::cout << ">> Variational Quantum Eigensolver (VQE)\n";
        if (!test_name.empty()) {
            std::cout << ">> Test: " << test_name << "\n";
            std::cout << ">> " << test_description << "\n";
        }
        std::cout << ">> Hamiltonian: " << H.size() << " Pauli terms\n";
        std::cout << ">> Qubits: " << n_qubits << ", Ansatz layers: " << layers
                  << '\n';
        std::cout << ">> Parameters: " << n_qubits * layers << "\n";
        if (opts.restarts > 1) {
            std::cout << ">> Restarts: " << opts.restarts << "\n";
        }
        if (opts.use_ring_entangler) {
            std::cout << ">> Entangler: ring CZ\n";
        }
        if (opts.alternate_rotations) {
            std::cout << ">> Rotations: alternating RY/RZ\n";
        }
        std::cout << '\n';
        if (have_exact_reference) {
            std::cout << ">> Exact ground state energy: " << std::fixed
                      << std::setprecision(6) << exact_ground_state << "\n\n";
        } else {
            std::cout << ">> (Exact diagonalization skipped for n > "
                      << MAX_QUBITS_EXACT_DIAG << " qubits)\n\n";
        }
        std::cout << ">> Running VQE...\n";
    }

    // Run VQE with exact expectation values
    Timer<> timer;

    auto [params, energy, iters] =
        optimize_vqe(H, n_qubits, layers, max_iterations, learning_rate,
                     opts.restarts, opts.use_ring_entangler,
                     opts.alternate_rotations, 1e-6, false, 0, verbose);

    timer.toc();
    realT runtime = timer.tics();
    realT error =
        have_exact_reference ? std::abs(energy - exact_ground_state) : 0.0;

    if (opts.json_output) {
        // JSON output for webapp integration
        json result;
        if (!test_name.empty()) {
            result["test_name"] = test_name;
            result["test_description"] = test_description;
        }
        result["n_qubits"] = n_qubits;
        result["n_terms"] = H.size();
        result["layers"] = layers;
        result["n_params"] = n_qubits * layers;
        result["learning_rate"] = learning_rate;
        result["restarts"] = opts.restarts;
        result["ring_entangler"] = opts.use_ring_entangler;
        result["alternate_rotations"] = opts.alternate_rotations;
        result["vqe_energy"] = energy;
        result["iterations"] = iters;
        result["runtime_seconds"] = runtime;

        if (have_exact_reference) {
            result["exact_ground_state"] = exact_ground_state;
            result["error"] = error;
            result["eigenvalues"] = eigenvalues;
        } else {
            result["exact_ground_state"] = nullptr; // JSON null
            result["error"] = nullptr;
        }

        // Optimized parameters
        result["optimized_params"] = params;

        // Hamiltonian terms for reference
        json terms = json::array();
        for (const auto& term : H) {
            terms.push_back(
                {{"coefficient", term.coefficient}, {"paulis", term.paulis}});
        }
        result["hamiltonian"] = terms;

        std::cout << result.dump(2) << std::endl;
    } else {
        // Human-readable output
        std::cout << "\n>> VQE completed in " << iters << " iterations\n";
        std::cout << ">> Final energy: " << std::fixed << std::setprecision(6)
                  << energy << "\n";
        if (have_exact_reference) {
            std::cout << ">> Exact ground state: " << exact_ground_state
                      << "\n";
            std::cout << ">> Error: " << std::scientific << std::setprecision(2)
                      << error << "\n";
        }
        std::cout << ">> Runtime: " << std::fixed << std::setprecision(2)
                  << runtime << " seconds\n";

        // Display the optimized state
        ket psi_opt = apply_hardware_efficient_ansatz(params, n_qubits, layers,
                                                      opts.use_ring_entangler,
                                                      opts.alternate_rotations);
        std::cout << "\n>> Optimized state:\n";
        std::cout << disp(dirac(psi_opt), IOManipDiracOpts{}.set_plus_op(" + "))
                  << '\n';
    }

    return EXIT_SUCCESS;
}
