TITLE Hyperpolarization-activated cation current

VERBATIM
struct Symbol;
extern void __nrn_cvode_abstol(Symbol**, double*, int) asm("__Z13_cvode_abstolPP6SymbolPdi");
int use_cachevec = 0;
void _cvode_abstol(Symbol** s, double* d, int i) {
    __nrn_cvode_abstol(s, d, i);
}
ENDVERBATIM

NEURON {
    SUFFIX Ih
    NONSPECIFIC_CURRENT ihcn
    RANGE gIhbar, ehcn, ihcn, gIh, m, minf, mtau
    GLOBAL vhalf, k, tau0, tauA, tauVhalf, tauSlope
}

UNITS {
    (mA) = (milliamp)
    (mV) = (millivolt)
    (S) = (siemens)
}

PARAMETER {
    gIhbar = 0.0001 (S/cm2)
    ehcn = -45 (mV)
    vhalf = -90 (mV)
    k = 10 (mV)
    tau0 = 5 (ms)
    tauA = 40 (ms)
    tauVhalf = -75 (mV)
    tauSlope = 12 (mV)
}

ASSIGNED {
    v (mV)
    ihcn (mA/cm2)
    gIh (S/cm2)
    minf
    mtau (ms)
}

STATE {
    m
}

INITIAL {
    rates(v)
    m = minf
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    gIh = gIhbar * m
    ihcn = gIh * (v - ehcn)
}

DERIVATIVE states {
    rates(v)
    m' = (minf - m) / mtau
}

PROCEDURE rates(v (mV)) {
    minf = 1 / (1 + exp((v - vhalf) / k))
    mtau = tau0 + tauA / (1 + exp((v - tauVhalf) / tauSlope))
}
