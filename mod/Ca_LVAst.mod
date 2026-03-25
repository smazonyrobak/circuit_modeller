TITLE T-type low-threshold calcium current

NEURON {
    SUFFIX Ca_LVAst
    USEION ca READ cai, cao WRITE ica VALENCE 2
    RANGE gCa_LVAstbar, gCa_LVA, ica_lva, eca_lva, m, h, minf, hinf, mtau, htau
    GLOBAL vhalf_m, k_m, vhalf_h, k_h
    GLOBAL tau_m_base, tau_m_scale, tau_m_vhalf, tau_m_slope
    GLOBAL tau_h_base, tau_h_scale, tau_h_vhalf, tau_h_slope
}

UNITS {
    (mA) = (milliamp)
    (mV) = (millivolt)
    (S) = (siemens)
}

PARAMETER {
    gCa_LVAstbar = 0.001 (S/cm2)
    vhalf_m = -57 (mV)
    k_m = 6.2 (mV)
    vhalf_h = -81 (mV)
    k_h = -4 (mV)
    tau_m_base = 0.5 (ms)
    tau_m_scale = 2.0 (ms)
    tau_m_vhalf = -40 (mV)
    tau_m_slope = 10 (mV)
    tau_h_base = 8 (ms)
    tau_h_scale = 25 (ms)
    tau_h_vhalf = -50 (mV)
    tau_h_slope = 7 (mV)
}

ASSIGNED {
    v (mV)
    cai (mM)
    cao (mM)
    ica (mA/cm2)
    ica_lva (mA/cm2)
    eca_lva (mV)
    gCa_LVA (S/cm2)
    minf
    hinf
    mtau (ms)
    htau (ms)
    cai_safe (mM)
}

STATE {
    m
    h
}

INITIAL {
    rates(v)
    m = minf
    h = hinf
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    cai_safe = cai
    if (cai_safe < 1e-7) {
        cai_safe = 1e-7
    }
    eca_lva = 0.0430866 * (celsius + 273.15) * log(cao / cai_safe)
    gCa_LVA = gCa_LVAstbar * m^2 * h
    ica_lva = gCa_LVA * (v - eca_lva)
    ica = ica_lva
}

DERIVATIVE states {
    rates(v)
    m' = (minf - m) / mtau
    h' = (hinf - h) / htau
}

PROCEDURE rates(v (mV)) {
    minf = 1 / (1 + exp(-(v - vhalf_m) / k_m))
    hinf = 1 / (1 + exp(-(v - vhalf_h) / k_h))
    mtau = tau_m_base + tau_m_scale / (1 + exp((v - tau_m_vhalf) / tau_m_slope))
    htau = tau_h_base + tau_h_scale / (1 + exp((v - tau_h_vhalf) / tau_h_slope))
}
