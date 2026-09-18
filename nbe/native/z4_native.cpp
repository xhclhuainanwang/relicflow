// C++17 hot path for the current-project VLL/Z4 model.
//
// The formulas below mirror models/VLL/VLL.wl and src/thermalavgl.wl.  The
// exported ABI is intentionally small so the Python runner can keep the
// existing RelicFlow cards, interpolation and output contract.

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <vector>

namespace {

constexpr double PI = 3.141592653589793238462643383279502884;
constexpr double FLOOR = 1.0e-120;
constexpr double X_BEXP = 100.0;

double clamp_exp(double x) {
    return std::exp(std::max(-745.0, std::min(700.0, x)));
}

double positive(double x) {
    return std::isfinite(x) && x > 0.0 ? x : FLOOR;
}

double i0(double x) {
    const double ax = std::abs(x);
    if (ax < 3.75) {
        const double y = (x / 3.75) * (x / 3.75);
        return 1.0 + y * (3.5156229 + y * (3.0899424 + y * (1.2067492 + y * (0.2659732 + y * (0.0360768 + y * 0.0045813)))));
    }
    const double y = 3.75 / ax;
    return std::exp(ax) / std::sqrt(ax) * (0.39894228 + y * (0.01328592 + y * (0.00225319 + y * (-0.00157565 + y * (0.00916281 + y * (-0.02057706 + y * (0.02635537 + y * (-0.01647633 + y * 0.00392377))))))));
}

double i1(double x) {
    const double ax = std::abs(x);
    double ans;
    if (ax < 3.75) {
        const double y = (x / 3.75) * (x / 3.75);
        ans = ax * (0.5 + y * (0.87890594 + y * (0.51498869 + y * (0.15084934 + y * (0.02658733 + y * (0.00301532 + y * 0.00032411))))));
    } else {
        const double y = 3.75 / ax;
        ans = std::exp(ax) / std::sqrt(ax) * (0.39894228 + y * (-0.03988024 + y * (-0.00362018 + y * (0.00163801 + y * (-0.01031555 + y * (0.02282967 + y * (-0.02895312 + y * (0.01787654 - y * 0.00420059))))))));
    }
    return x < 0.0 ? -ans : ans;
}

double k0_exact(double x) {
    const double xx = std::max(x, 1.0e-300);
    if (xx <= 2.0) {
        const double y = xx * xx / 4.0;
        return -std::log(xx / 2.0) * i0(xx) + (-0.57721566 + y * (0.42278420 + y * (0.23069756 + y * (0.03488590 + y * (0.00262698 + y * (0.00010750 + y * 0.00000740))))));
    }
    const double y = 2.0 / xx;
    return std::exp(-xx) / std::sqrt(xx) * (1.25331414 + y * (-0.07832358 + y * (0.02189568 + y * (-0.01062446 + y * (0.00587872 + y * (-0.00251540 + y * 0.00053208))))));
}

double k1_exact(double x) {
    const double xx = std::max(x, 1.0e-300);
    if (xx <= 2.0) {
        const double y = xx * xx / 4.0;
        return std::log(xx / 2.0) * i1(xx) + (1.0 / xx) * (1.0 + y * (0.15443144 + y * (-0.67278579 + y * (-0.18156897 + y * (-0.01919402 + y * (-0.00110404 + y * (-0.00004686)))))));
    }
    const double y = 2.0 / xx;
    return std::exp(-xx) / std::sqrt(xx) * (1.25331414 + y * (0.23498619 + y * (-0.03655620 + y * (0.01504268 + y * (-0.00780353 + y * (0.00325614 + y * (-0.00068245)))))));
}

double k1(double x) {
    if (x < X_BEXP) return positive(k1_exact(x));
    const double y = 1.0 / x;
    return positive(std::exp(-x) * std::sqrt(PI / (2.0 * x)) * (1.0 + 3.0 * y / 8.0 - 15.0 * y * y / 128.0 + 105.0 * y * y * y / 1024.0 - 4725.0 * std::pow(y, 4) / 32768.0 + 72765.0 * std::pow(y, 5) / 262144.0));
}

double k2(double x) {
    if (x < X_BEXP) return positive(k0_exact(x) + 2.0 * k1_exact(x) / std::max(x, 1.0e-300));
    const double y = 1.0 / x;
    return positive(std::exp(-x) * std::sqrt(PI / (2.0 * x)) * (1.0 + 15.0 * y / 8.0 + 105.0 * y * y / 128.0 - 315.0 * y * y * y / 1024.0 + 10395.0 * std::pow(y, 4) / 32768.0 - 135135.0 * std::pow(y, 5) / 262144.0));
}

double exp_dbk2(double x) {
    const double xx = std::max(x, 1.0e-12);
    if (xx < X_BEXP) return positive(std::exp(-xx) / std::max(k2(xx), FLOOR));
    const double y = 1.0 / xx;
    const double a = 1.0 + 15.0 * y / 8.0 + 105.0 * y * y / 128.0 - 315.0 * y * y * y / 1024.0 + 10395.0 * std::pow(y, 4) / 32768.0 - 135135.0 * std::pow(y, 5) / 262144.0;
    return positive(1.0 / (std::sqrt(PI / (2.0 * xx)) * a));
}

double der_dbk2(double x) {
    const double xx = std::max(x, 1.0e-12);
    if (xx < X_BEXP) return 0.5 * (-k1_exact(xx) - (k1_exact(xx) + 4.0 * k2(xx) / xx)) / std::max(k2(xx), FLOOR);
    return -1.0 - 0.5 / xx - 15.0 / (8.0 * xx * xx) + 15.0 / (8.0 * xx * xx * xx) - 135.0 / (128.0 * std::pow(xx, 4)) - 45.0 / (32.0 * std::pow(xx, 5)) + 7425.0 / (1024.0 * std::pow(xx, 6));
}

struct GL {
    std::vector<double> x;
    std::vector<double> w;
};

GL gauss_legendre(int n) {
    n = std::max(1, n);
    GL out{std::vector<double>(n), std::vector<double>(n)};
    const int m = (n + 1) / 2;
    for (int i = 0; i < m; ++i) {
        double z = std::cos(PI * (i + 0.75) / (n + 0.5));
        for (int it = 0; it < 80; ++it) {
            double p1 = 1.0;
            double p2 = 0.0;
            for (int j = 1; j <= n; ++j) {
                const double p3 = p2;
                p2 = p1;
                p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3) / j;
            }
            const double pp = n * (z * p1 - p2) / (z * z - 1.0);
            const double dz = p1 / pp;
            z -= dz;
            if (std::abs(dz) < 1.0e-15) break;
        }
        double p1 = 1.0;
        double p2 = 0.0;
        for (int j = 1; j <= n; ++j) {
            const double p3 = p2;
            p2 = p1;
            p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3) / j;
        }
        const double pp = n * (z * p1 - p2) / (z * z - 1.0);
        const double ww = 2.0 / ((1.0 - z * z) * pp * pp);
        out.x[i] = -z;
        out.x[n - 1 - i] = z;
        out.w[i] = ww;
        out.w[n - 1 - i] = ww;
    }
    return out;
}

const GL& cached_gauss_legendre(int n) {
    switch (n) {
        case 10: {
            static const GL gl = gauss_legendre(10);
            return gl;
        }
        case 16: {
            static const GL gl = gauss_legendre(16);
            return gl;
        }
        case 24: {
            static const GL gl = gauss_legendre(24);
            return gl;
        }
        case 32: {
            static const GL gl = gauss_legendre(32);
            return gl;
        }
        case 40: {
            static const GL gl = gauss_legendre(40);
            return gl;
        }
        case 48: {
            static const GL gl = gauss_legendre(48);
            return gl;
        }
        case 64: {
            static const GL gl = gauss_legendre(64);
            return gl;
        }
        case 80: {
            static const GL gl = gauss_legendre(80);
            return gl;
        }
        case 128: {
            static const GL gl = gauss_legendre(128);
            return gl;
        }
        case 256: {
            static const GL gl = gauss_legendre(256);
            return gl;
        }
        default:
            return gauss_legendre(n);
    }
}

double quad(const std::function<double(double)>& f, double a, double b, int n = 48) {
    if (!(b > a)) return 0.0;
    const GL& gl = cached_gauss_legendre(n);
    const double mid = 0.5 * (a + b);
    const double half = 0.5 * (b - a);
    double sum = 0.0;
    for (std::size_t i = 0; i < gl.x.size(); ++i) sum += gl.w[i] * f(mid + half * gl.x[i]);
    return half * sum;
}

struct P {
    double r, m, delta, y, lam, mL;
    double gammas, gammap;
    int q_order;

    P(const double* p, int pg = 3)
        : r(p[0]), m(p[1]), delta(p[2]), y(p[3]), lam(p[4]), mL(p[5]), gammas(0.0), gammap(0.0), q_order(std::max(32, std::min(96, 16 * std::max(pg, 1)))) {
        const double one = 1.0 + delta;
        const double phase = 4.0 * m * m - std::pow(mL - m * r, 2) * one;
        double rad = (1.0 - std::pow(mL - m * r, 2) * one / (4.0 * m * m));
        rad *= (1.0 - std::pow(mL + m * r, 2) * one / (4.0 * m * m));
        double brc = 0.0;
        if (-1.0 < delta && delta < 0.0) brc = m * 16.0 * y * y * std::sqrt(-delta) / (32.0 * PI * std::sqrt(one));
        double brl = 0.0;
        if (phase > 0.0 && rad > 0.0) brl = 4.0 * lam * lam * phase * std::sqrt(rad) / (32.0 * m * PI * std::sqrt(one));
        gammas = positive(brc + brl);
        gammap = (phase > 0.0 && rad > 0.0) ? positive(2.0 * y * y * phase * std::sqrt(rad) / (32.0 * m * PI)) : FLOOR;
    }

    double sig_s(double s) const {
        const double mpsi = m * r;
        const double a = std::pow(mL - m * r, 2);
        const double b = std::pow(mL + m * r, 2);
        const double rad = (s - a) * (s - b);
        if (s <= 4.0 * m * m || s <= b || rad <= 0.0) return FLOOR;
        const double mphi2 = 4.0 * m * m / std::max(1.0 + delta, FLOOR);
        const double denr = std::pow(s - mphi2, 2) + mphi2 * gammas * gammas;
        const double den = 32.0 * PI * (s - 2.0 * m * m) * denr;
        return positive(y * y * lam * lam * (s - mpsi * mpsi - mL * mL) * std::sqrt(rad) / den);
    }

    double sig_l(double s) const {
        // Active VLL svl: one-component s+t washout with interference.
        const double mpsi = r * m;
        const double mpsi2 = mpsi * mpsi;
        if (s <= mpsi2) return FLOOR;
        const double mphi2 = 4.0 * m * m / std::max(1.0 + delta, FLOOR);
        const double ds = 1.0 / std::max(std::pow(s - mphi2, 2) + mphi2 * gammas * gammas, FLOOR);
        const double tmin = 2.0 * mpsi2 - s;
        const double tmax = mpsi2 * mpsi2 / s;
        if (!(tmax > tmin)) return FLOOR;
        // Match the VLL card's Indeterminate for an on-shell bare t-channel
        // pole while preserving RelicFlow's finite positive-rate contract.
        if (tmin <= mphi2 && mphi2 <= tmax) return FLOOR;

        const double width_t = tmax - tmin;
        const double denom_min = tmin - mphi2;
        const double denom_max = tmax - mphi2;
        const double log_ratio = std::log(std::max(std::abs(denom_max), 1.0e-300) / std::max(std::abs(denom_min), 1.0e-300));
        const double q = mphi2 - mpsi2;
        const double integral_t2 = width_t + 2.0 * q * log_ratio - q * q * (1.0 / denom_max - 1.0 / denom_min);
        const double integral_interference = 2.0 * (s - mphi2) * ds * (s * width_t - (mpsi2 * mpsi2 - s * mphi2) * log_ratio);
        const double integral = 0.5 * std::pow(lam, 4) * (
            (s - mpsi2) * (s - mpsi2) * ds * width_t + integral_t2 + integral_interference
        );
        return positive(integral / (16.0 * PI * (s - mpsi2) * (s - mpsi2)));
    }

    static double kallen(double s, double a, double b) {
        return s * s + a * a + b * b - 2.0 * (s * a + s * b + a * b);
    }

    double sig_1(double s) const {
        const double m2 = r * m;
        const double m4 = 0.0;
        const double pin2 = kallen(s, m * m, m2 * m2);
        const double pout2 = kallen(s, m * m, m4 * m4);
        if (s <= (m + m2) * (m + m2) || pin2 <= 0.0 || pout2 <= 0.0) return FLOOR;
        const double root_s = std::sqrt(s);
        const double pin = std::sqrt(pin2) / (2.0 * root_s);
        const double pout = std::sqrt(pout2) / (2.0 * root_s);
        const double ex_in = (s + m * m - m2 * m2) / (2.0 * root_s);
        const double ex_out = (s + m * m - m4 * m4) / (2.0 * root_s);
        const auto f = [&](double c) {
            const double t = 2.0 * m * m - 2.0 * ex_in * ex_out + 2.0 * pin * pout * c;
            return y * y * lam * lam * (4.0 * m * m - t) * (m2 * m2 - t) /
                std::max(std::pow(t - 4.0 * m * m / std::max(1.0 + delta, FLOOR), 2), FLOOR);
        };
        const double avg = 0.5 * quad(f, -1.0, 1.0, std::max(16, q_order - 16));
        const double sigma = (1.0 / (16.0 * PI * s)) * (pout / pin) * avg;
        const double vrel = std::sqrt(std::max(pin2, 0.0)) / std::max(s - m * m - m2 * m2, FLOOR);
        return positive(sigma * vrel);
    }

    double sig_2(double s) const {
        const double m2 = r * m;
        const double pin2 = kallen(s, m2 * m2, m2 * m2);
        if (s <= 4.0 * m2 * m2 || pin2 <= 0.0) return FLOOR;
        const double root_s = std::sqrt(s);
        const double pin = std::sqrt(pin2) / (2.0 * root_s);
        const double pout = root_s / 2.0;
        const double mphi2 = 4.0 * m * m / std::max(1.0 + delta, FLOOR);
        const double root_term = std::sqrt(std::max(s * (s - 4.0 * m2 * m2), 0.0));
        const double tmin = m2 * m2 - s / 2.0 - root_term / 2.0;
        const double tmax = m2 * m2 - s / 2.0 + root_term / 2.0;
        const auto f = [&](double c) {
            const double t = tmin + (tmax - tmin) * (1.0 + c) / 2.0;
            const double u = 2.0 * m2 * m2 - s - t;
            const double dt = t - mphi2;
            const double du = u - mphi2;
            const double den_t = std::max(dt * dt, FLOOR);
            const double den_u = std::max(du * du, FLOOR);
            const double den_tu = dt * du;
            const double interference = std::abs(den_tu) < 1.0e-300 ? 0.0 :
                2.0 * (m2 * m2 * s - (m2 * m2 - t) * (m2 * m2 - u)) / den_tu;
            return std::pow(lam, 4) / 4.0 * (
                (m2 * m2 - t) * (m2 * m2 - t) / den_t
                + (m2 * m2 - u) * (m2 * m2 - u) / den_u
                - interference
            ) * (tmax - tmin) / 2.0;
        };
        const double integral = quad(f, -1.0, 1.0, std::max(16, q_order - 16));
        const double sigma = integral / (32.0 * PI * s * (s - 4.0 * m2 * m2));
        const double vrel = std::sqrt(std::max(kallen(s, m2 * m2, m2 * m2), 0.0)) /
            std::max(s - 2.0 * m2 * m2, FLOOR);
        return positive(sigma * vrel);
    }

    std::vector<double> limits(double base, double x) const {
        const double near = base * (1.0 + 20.0 / std::max(x, 1.0e-12));
        if (delta < 0.0) {
            const double res = 4.0 * m * m / (1.0 + delta);
            if (res < near) return {base, res, near};
            return {base, near, std::min(base * (1.0 + 30.0 / std::max(x, 1.0e-12)), 2.0 * res)};
        }
        return {base, near};
    }

    double integrate_segments(const std::function<double(double)>& f, const std::vector<double>& ls, int n = 48) const {
        double total = 0.0;
        for (std::size_t i = 1; i < ls.size(); ++i) total += quad(f, ls[i - 1], ls[i], n);
        return positive(total);
    }

    // Remove the square-root threshold cusp before applying fixed Gauss
    // quadrature.  The Python/MMA integral is adaptive in s; using
    // s = s_th (1 + u^2) gives the same threshold behavior with a smooth
    // integrand and keeps the native path fast.
    double integrate_threshold(const std::function<double(double)>& f, double threshold, double upper, int n = 48) const {
        if (!(upper > threshold) || !(threshold > 0.0)) return 0.0;
        const double umax = std::sqrt(std::max(upper / threshold - 1.0, 0.0));
        const auto transformed = [&](double u) {
            const double s = threshold * (1.0 + u * u);
            return f(s) * (2.0 * threshold * u);
        };
        return quad(transformed, 0.0, umax, n);
    }

    double integrate_resonant(const std::function<double(double)>& f, double a, double b, double resonance, double width, int n = 48) const {
        if (!(b > a)) return 0.0;
        if (!(resonance > a && resonance < b && width > 0.0 && std::isfinite(width))) return quad(f, a, b, n);
        const double ta = std::atan((a - resonance) / width);
        const double tb = std::atan((b - resonance) / width);
        const auto transformed = [&](double theta) {
            const double c = std::cos(theta);
            const double sec2 = 1.0 / std::max(c * c, 1.0e-300);
            const double s = resonance + width * std::tan(theta);
            return f(s) * width * sec2;
        };
        return quad(transformed, ta, tb, n);
    }

    double integrate_resonant_segments(const std::function<double(double)>& f, const std::vector<double>& ls, double resonance, double width, int n = 48) const {
        if (ls.size() >= 2 && resonance > ls.front() && resonance < ls.back() && width > 0.0 && std::isfinite(width)) {
            return positive(integrate_resonant(f, ls.front(), ls.back(), resonance, width, n));
        }
        double total = 0.0;
        for (std::size_t i = 1; i < ls.size(); ++i) total += integrate_resonant(f, ls[i - 1], ls[i], resonance, width, n);
        return positive(total);
    }

    double thermal_identical(double x, const std::function<double(double)>& sigma) const {
        const double xx = std::max(x, 1.0e-8);
        const double temp = m / xx;
        const double threshold = 4.0 * m * m;
        const auto ls = limits(threshold, xx);
        const auto f = [&](double s) {
            if (xx < X_BEXP) {
                const double z = std::sqrt(std::max(s, 0.0)) / temp;
                const double den = 8.0 * temp * std::pow(m, 4) * std::pow(std::max(k2(xx), FLOOR), 2);
                return (s - 2.0 * m * m) * std::sqrt(std::max(s - 4.0 * m * m, 0.0)) * k1(z) / den * sigma(s);
            }
            const double st = s / (4.0 * m * m);
            if (st <= 1.0) return 0.0;
            const double poly = -15.0 + 24.0 * std::sqrt(st) * (-15.0 + 4.0 * xx) + 16.0 * st * (285.0 - 120.0 * xx + 32.0 * xx * xx);
            const double pref = 1.0 / (4.0 * m * m) * clamp_exp(-2.0 * (-1.0 + std::sqrt(st)) * xx) * std::sqrt(st - 1.0) * (-1.0 + 2.0 * st) * std::sqrt(1.0 / xx) * poly / (256.0 * std::sqrt(PI) * std::pow(st, 1.25));
            return pref * sigma(s);
        };
        const double resonance = threshold / (1.0 + delta);
        const double width = 2.0 * gammas * m / std::sqrt(std::max(1.0 + delta, FLOOR));
        // The threshold-variable integral is already smooth.  The first few
        // low-x points still probe the broadest interval, so use a higher
        // order there to preserve the cancellation in the Y_{B-L} source.
        const int order = xx < 2.0 ? std::max(q_order, 128) : (xx < X_BEXP ? q_order : q_order + 16);
        // For delta >= 0 the pole lies below the physical threshold, so the
        // active MMA/Python path has no resonant change of variables.
        if (!(delta < 0.0 && resonance > threshold && resonance < ls.back() && width > 0.0)) {
            return integrate_threshold(f, threshold, ls.back(), order);
        }
        return integrate_resonant_segments(f, ls, resonance, width, order);
    }

    double thermal_pair(double x, const std::function<double(double)>& sigma, int kind) const {
        const double xx = std::max(x, 1.0e-8);
        const double temp = m / xx;
        double q0, m1, m2, high_norm;
        if (kind == 0) {
            q0 = r * r * m * m + mL * mL;
            m1 = mL;
            m2 = r * m;
            high_norm = 2.0 * q0;
        } else if (kind == 1) {
            q0 = r * r * m * m + m * m;
            m1 = m;
            m2 = r * m;
            high_norm = 2.0 * q0;
        } else {
            q0 = 2.0 * r * r * m * m;
            m1 = r * m;
            m2 = r * m;
            high_norm = 2.0 * (r * r * m * m + m * m);
        }
        const auto f = [&](double s) {
            if (xx < X_BEXP) {
                const double den = 8.0 * temp * m1 * m1 * m2 * m2 * std::max(k2(xx * m1 / m), FLOOR) * std::max(k2(xx * m2 / m), FLOOR);
                return (s - q0) * std::sqrt(std::max(s - 2.0 * q0, 0.0)) * k1(std::sqrt(std::max(s, 0.0)) / temp) / den * sigma(s);
            }
            const double st = s / std::max(high_norm, FLOOR);
            if (st <= 1.0) return 0.0;
            const double poly = -15.0 + 24.0 * std::sqrt(st) * (-15.0 + 4.0 * X_BEXP) + 16.0 * st * (285.0 - 120.0 * X_BEXP + 32.0 * X_BEXP * X_BEXP);
            const double pref = 1.0 / std::max(high_norm, FLOOR) * clamp_exp(-2.0 * (-1.0 + std::sqrt(st)) * X_BEXP) * std::sqrt(st - 1.0) * (-1.0 + 2.0 * st) * std::sqrt(1.0 / X_BEXP) * poly / (256.0 * std::sqrt(PI) * std::pow(st, 1.25));
            return pref * sigma(s);
        };
        const double resonance = 4.0 * m * m / (1.0 + delta);
        const double width_source = kind == 0 ? gammas : gammap;
        const double width = 2.0 * width_source * m / std::sqrt(std::max(1.0 + delta, FLOOR));
        const double threshold = 2.0 * q0;
        auto ls = limits(threshold, xx);
        if (kind == 0 && resonance > threshold) {
            const double near = threshold * (1.0 + 20.0 / std::max(xx, 1.0e-12));
            if (resonance < near) {
                ls = {threshold, resonance, near};
            } else {
                // Match MMA TAsvxl: the extended upper limit is based on
                // 4*m^2, not the initial-state threshold 2*q0.  Using
                // threshold here drops below the Breit-Wigner pole at a
                // finite x and creates an artificial thermal-rate jump.
                ls = {threshold, near, std::min(4.0 * m * m * (1.0 + 30.0 / std::max(xx, 1.0e-12)), 2.0 * resonance)};
            }
        }
        // svxl can contain a much narrower Breit-Wigner peak than the other
        // thermal channels.  Resolve it with a dedicated high-order rule;
        // svxl1/svxl2 remain on the lighter threshold rule.
        const int order = kind == 0 ? std::max(256, q_order + 32) : std::min(128, q_order + 32);
        // Match the active thermalavgl.wl card: svxl carries a resonance
        // whenever it lies above its initial-state threshold, while svxl1
        // and svxl2 use the threshold integral without this breakpoint.
        const bool use_resonance = kind == 0 && resonance > threshold && resonance < ls.back() && width > 0.0;
        if (!use_resonance) {
            return integrate_threshold(f, threshold, ls.back(), order);
        }
        return integrate_resonant_segments(f, ls, resonance, width, order);
    }

    double thermal_sv2x(double x) const {
        const double xx = std::max(x, 1.0e-8);
        const auto ls = limits(1.0, xx);
        if (xx < X_BEXP) {
            const double k2x = std::max(k2(xx), FLOOR);
            const double expdb = exp_dbk2(xx);
            const auto ff = [&](double s) {
                const auto fep = [&](double ep) {
                    const double root = std::sqrt(std::max((s - 1.0) * (ep * ep - 1.0), 0.0));
                    const double a = std::sqrt(std::max(s, 0.0)) * ep;
                    if (a <= root) return 0.0;
                    const double ratio = std::max((a - root) / std::max(a + root, FLOOR), FLOOR);
                    return (2.0 * std::sqrt(std::max(s, 0.0)) * (2.0 * s - 1.0) * std::pow(xx, 3) / 3.0) * expdb * expdb * clamp_exp(-2.0 * std::sqrt(std::max(s, 0.0)) * xx * ep + 2.0 * xx) * std::log(ratio);
                };
                return quad(fep, 1.0, 1.0 + 20.0 / xx, std::max(16, q_order - 8));
            };
            const auto f = [&](double s) {
                const double first = (2.0 * std::sqrt(std::max(s, 0.0)) * (2.0 * s - 1.0) * std::pow(xx, 3)) / (3.0 * k2x * k2x) * std::sqrt(std::max((s - 1.0) * s, 0.0)) * k2(2.0 * std::sqrt(std::max(s, 0.0)) * xx) / std::max(std::sqrt(std::max(s, 0.0)) * xx, FLOOR);
                return (first + ff(s)) * sig_s(4.0 * m * m * s);
            };
            const double resonance = 1.0 / (1.0 + delta);
            const double width = 2.0 * gammas * m / std::sqrt(std::max(1.0 + delta, FLOOR)) / (4.0 * m * m);
            const int order = std::max(16, q_order - 8);
            if (!(delta < 0.0 && resonance > ls.front() && resonance < ls.back() && width > 0.0)) {
                return integrate_threshold(f, ls.front(), ls.back(), order);
            }
            return integrate_resonant_segments(f, ls, resonance, width, order);
        }
        const auto f = [&](double s) {
            if (s <= 1.0) return 0.0;
            // Stable form of the source polynomial, expanded in
            // r = sqrt(s)-1.  The direct s-form suffers catastrophic
            // cancellation near the high-x integration endpoint s=1.
            const double root_s = std::sqrt(s);
            const double r = root_s - 1.0;
            const double p4 = 512.0 * xx * xx - 1920.0 * xx + 4560.0;
            const double p3 = 2048.0 * xx * xx - 6560.0 * xx + 14040.0;
            const double p2 = 2560.0 * xx * xx - 6240.0 * xx + 11145.0;
            const double p1 = 1024.0 * xx * xx - 832.0 * xx - 270.0;
            const double p0 = 768.0 * xx - 2160.0;
            const double poly = (((p4 * r + p3) * r + p2) * r + p1) * r + p0;
            double kernel = clamp_exp(-2.0 * r * xx) * std::sqrt((s - 1.0) / s) * (-1.0 + 2.0 * s) * poly;
            kernel /= 768.0 * std::sqrt(PI) * std::pow(s, 0.75) * std::sqrt(1.0 / xx);
            return sig_s(4.0 * m * m * s) * kernel;
        };
        const double resonance = 1.0 / (1.0 + delta);
        const double width = 2.0 * gammas * m / std::sqrt(std::max(1.0 + delta, FLOOR)) / (4.0 * m * m);
        return integrate_resonant_segments(f, ls, resonance, width, q_order + 16);
    }
};

void rel_temp_correction(double x, double* out) {
    const double xx = std::max(x, 1.0e-12);
    constexpr int bins = 220;
    const double lmin = 1.0e-4 * std::sqrt(2.0 * xx);
    const double lmax = 8.0 * std::sqrt(2.0 * xx);
    const double dl = (lmax - lmin) / (bins - 1.0);
    const double expdb = exp_dbk2(xx);
    double sum34 = 0.0;
    double sum34p1 = 0.0;
    double sum34p2 = 0.0;
    for (int i = 0; i < bins; ++i) {
        const double l = lmin + (lmax - lmin) * i / (bins - 1.0);
        const double denom = xx * xx + l * l;
        const double fdm = clamp_exp(-(std::sqrt(denom) - xx)) * expdb;
        sum34 += l * l * l * l * l * l / std::pow(denom, 1.5) * fdm;
        sum34p1 += l * l * l * l * l * l / std::pow(denom, 2.5) * fdm;
        sum34p2 += l * l * l * l * l * l / (denom * denom) * fdm;
    }
    const double w34 = (1.0 / 3.0) * std::pow(xx, -2.0) * dl * sum34;
    double w34p = -(2.0 / xx + der_dbk2(xx)) * w34;
    w34p -= (1.0 / xx) * dl * sum34p1;
    w34p -= (1.0 / (3.0 * xx)) * dl * sum34p2;
    out[0] = w34;
    out[1] = w34p;
}

void full_cell(const P& p, const double* cfg, double xi, double yeq, double y, double expdb, double derdb, double* out) {
    const int dim = std::max(4, static_cast<int>(cfg[0]));
    const int nmu = std::max(1, static_cast<int>(cfg[1]));
    const int nu = std::max(1, static_cast<int>(cfg[2]));
    const double ubase = cfg[3];
    const double utail = cfg[4];
    const double ucap = cfg[5];
    const double ysafe = std::max(std::abs(y), FLOOR);
    const double yeqdy = yeq / ysafe;
    const double xdm = xi * yeqdy;
    const double qmin = 1.0e-4 * std::sqrt(std::max(2.0 * xi / std::max(yeqdy, FLOOR), FLOOR));
    const double qmax = 8.0 * std::sqrt(std::max(2.0 * xi / std::max(yeqdy, FLOOR), FLOOR));
    const double dq = (qmax - qmin) / std::max(dim - 1, 1);
    std::vector<double> q(dim), xq(dim), fdm(dim);
    for (int i = 0; i < dim; ++i) {
        q[i] = qmin + i * dq;
        xq[i] = std::sqrt(xi * xi + q[i] * q[i]);
        fdm[i] = clamp_exp(-(yeqdy * xq[i] - xdm)) * expdb;
    }
    const GL gmu = gauss_legendre(nmu);
    const GL gu = gauss_legendre(nu);
    std::vector<double> matrix(dim * dim, 0.0);
    const double mPsihat = p.r * xi;
    const double mPhihat = 2.0 * xi / std::sqrt(1.0 + p.delta);
    const double mLhat = p.mL * xi / p.m;
    const double gy = p.y * p.y * p.lam * p.lam;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < dim - 1; ++i) {
        for (int j = i + 1; j < dim; ++j) {
            const double dx = xq[j] - xq[i];
            const double dx2 = dx * dx;
            const double umin = std::sqrt(std::max(std::pow(mPsihat + dx, 2) - mLhat * mLhat, 0.0));
            const double umax = std::min(std::max(umin + utail, ubase), ucap);
            if (!(umin < umax)) continue;
            const double scale = umax - umin;
            std::vector<double> u(nu), usq(nu), kp2(nu), base(nu);
            for (int b = 0; b < nu; ++b) {
                u[b] = umin + scale * (gu.x[b] + 1.0) / 2.0;
                usq[b] = u[b] * u[b];
                const double omega = std::sqrt(usq[b] + mLhat * mLhat);
                const double omegap = omega - dx;
                base[b] = 0.0;
                kp2[b] = 0.0;
                if (omegap > mPsihat) {
                    kp2[b] = omegap * omegap - mPsihat * mPsihat;
                    if (kp2[b] > 0.0) {
                        const double z1 = 0.5 * omega;
                        const double z2 = 0.5 * omegap;
                        const double ch1 = z1 > 50.0 ? 0.5 * clamp_exp(z1) : std::cosh(z1);
                        const double ch2 = z2 > 50.0 ? 0.5 * clamp_exp(z2) : std::cosh(z2);
                        base[b] = scale * gu.w[b] / 2.0 * (u[b] / std::max(omega, FLOOR)) / std::max(4.0 * ch1 * ch2, FLOOR);
                    }
                }
            }
            double wij = 0.0;
            for (int k = 0; k < nmu; ++k) {
                const double mu = gmu.x[k];
                const double qh2 = std::max(q[i] * q[i] + q[j] * q[j] - 2.0 * q[i] * q[j] * mu, 1.0e-12);
                const double qh = std::sqrt(qh2);
                const double that = dx2 - qh2;
                // Spin/channel-summed chi L -> anti-chi Psi conversion
                // kernel.  The factor 2 counts the two degenerate SU(2)_L
                // bath components and belongs in |M|^2, not in the
                // collision-rate prefactor.
                const double msq = 2.0 * gy * (4.0 * xi * xi - that) * (mPsihat * mPsihat - that) / std::max(std::pow(that - mPhihat * mPhihat, 2), FLOOR);
                double ihat = 0.0;
                for (int b = 0; b < nu; ++b) {
                    if (base[b] == 0.0) continue;
                    const double omega = std::sqrt(usq[b] + mLhat * mLhat);
                    const double cos_th = (usq[b] + qh2 - kp2[b]) / std::max(2.0 * u[b] * qh, FLOOR);
                    if (std::abs(cos_th) <= 1.0) ihat += base[b];
                }
                const double pref0 = xi * xi / (32.0 * PI * 2.0 * xq[i] * xq[j]);
                wij += gmu.w[k] * (pref0 / qh) * msq * ihat;
            }
            wij *= 0.5;
            matrix[i * dim + j] = q[j] * q[j] * wij;
            matrix[j * dim + i] = q[i] * q[i] * wij;
        }
    }
    for (int i = 0; i < dim; ++i) {
        double diag = 0.0;
        for (int j = 0; j < dim; ++j) diag -= matrix[i * dim + j] * clamp_exp((xq[i] - xq[j]) / 2.0);
        matrix[i * dim + i] = diag;
    }
    for (int i = 0; i < dim - 1; ++i) {
        for (int j = i + 1; j < dim; ++j) {
            const double dx = xq[i] - xq[j];
            matrix[i * dim + j] *= clamp_exp(-dx / 2.0);
            matrix[j * dim + i] *= clamp_exp(dx / 2.0);
        }
    }
    const double c = xdm * std::pow(xi, -7.0) * dq * dq / (6.0 * PI * PI);
    double av = 0.0;
    double avx = 0.0;
    for (int i = 0; i < dim; ++i) {
        double row = 0.0;
        double rowx = 0.0;
        for (int j = 0; j < dim; ++j) {
            row += matrix[i * dim + j] * fdm[j];
            rowx += matrix[i * dim + j] * xq[j] * fdm[j];
        }
        const double sec = q[i] * q[i] * q[i] * q[i] / xq[i];
        av += sec * row;
        avx += sec * rowx;
    }
    av *= c;
    const double der = (xdm * derdb - 1.0) * av / ysafe + c * (yeqdy / ysafe) * avx;
    out[0] = av;
    out[1] = der;
}

}  // namespace

extern "C" {

__declspec(dllexport) int z4_native_version() { return 1; }

__declspec(dllexport) void z4_rates(int n, const double* xs, const double* params, int pg, double* out) {
    const P p(params, pg);
    for (int i = 0; i < n; ++i) {
        const double x = std::max(xs[i], 1.0e-8);
        out[6 * i + 0] = p.thermal_identical(x, [&](double s) { return p.sig_s(s); });
        out[6 * i + 1] = p.thermal_sv2x(x);
        out[6 * i + 2] = p.thermal_pair(x, [&](double s) { return p.sig_l(s); }, 0);
        out[6 * i + 3] = p.thermal_pair(x, [&](double s) { return p.sig_1(s); }, 1);
        out[6 * i + 4] = p.thermal_pair(x, [&](double s) { return p.sig_2(s); }, 2);
        out[6 * i + 5] = p.thermal_identical(x, [&](double) { return p.gammas; });
        for (int j = 0; j < 6; ++j) out[6 * i + j] = positive(out[6 * i + j]);
    }
}

__declspec(dllexport) void z4_full_cell(const double* params, const double* config, double xi, double yeq, double y, double expdb, double derdb, double* out) {
    full_cell(P(params), config, xi, yeq, y, expdb, derdb, out);
}

__declspec(dllexport) void z4_rel_temp(int n, const double* xs, double* out) {
    for (int i = 0; i < n; ++i) rel_temp_correction(xs[i], out + 2 * i);
}

}
