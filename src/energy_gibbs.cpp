#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>

// All updates draw directly from the two full conditional distributions.
namespace {
double open_uniform(std::mt19937_64& rng) {
    const std::uint64_t bits = rng() >> 11;
    return bits == 0 ? 0x1.0p-54 : static_cast<double>(bits) * 0x1.0p-53;
}

double positive_root(double excess, double slope, double curvature) {
    if (excess <= 0.0) return 0.0;
    return 2.0 * excess /
           (slope + std::sqrt(slope * slope + 2.0 * curvature * excess));
}
}

extern "C" int sample_energy_gibbs(
    int count, const double* breaks, const double* heights,
    const double* slopes, const double* curvature, int mode_piece,
    double mode, double eta, int burn, int kept, std::uint64_t seed,
    double initial, double* output) {
    if (!(eta > 0.0) || burn < 0 || kept < 1 || mode_piece < 0 || mode_piece > count)
        return -1;
    std::mt19937_64 rng(seed);
    double beta = initial;
    for (int iteration = 0; iteration < burn + kept; ++iteration) {
        int piece = static_cast<int>(std::upper_bound(breaks, breaks + count, beta) - breaks);
        double energy;
        if (piece == mode_piece) {
            double distance = beta - mode;
            energy = 0.5 * curvature[piece] * distance * distance;
        } else if (piece < mode_piece) {
            double distance = breaks[piece] - beta;
            energy = heights[piece] - slopes[piece] * distance
                     + 0.5 * curvature[piece] * distance * distance;
        } else {
            double distance = beta - breaks[piece - 1];
            energy = heights[piece - 1] + slopes[piece - 1] * distance
                     + 0.5 * curvature[piece] * distance * distance;
        }
        double t = energy - std::log(open_uniform(rng)) / eta;

        int lo = 0, hi = mode_piece;
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            if (heights[mid] > t) lo = mid + 1;
            else hi = mid;
        }
        int left_piece = lo;
        double inner = left_piece == mode_piece ? mode : breaks[left_piece];
        double inner_energy = left_piece == mode_piece ? 0.0 : heights[left_piece];
        double outward_slope = left_piece == mode_piece ? 0.0 : -slopes[left_piece];
        double left = inner - positive_root(t - inner_energy, outward_slope, curvature[left_piece]);

        lo = mode_piece; hi = count;
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            if (heights[mid] < t) lo = mid + 1;
            else hi = mid;
        }
        int right_piece = lo;
        inner = right_piece == mode_piece ? mode : breaks[right_piece - 1];
        inner_energy = right_piece == mode_piece ? 0.0 : heights[right_piece - 1];
        outward_slope = right_piece == mode_piece ? 0.0 : slopes[right_piece - 1];
        double right = inner + positive_root(t - inner_energy, outward_slope, curvature[right_piece]);
        if (!(std::isfinite(left) && std::isfinite(right) && left < right))
            return iteration + 1;
        beta = left + open_uniform(rng) * (right - left);
        if (iteration >= burn) output[iteration - burn] = beta;
    }
    return 0;
}
