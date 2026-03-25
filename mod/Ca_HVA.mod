TITLE High-voltage activated calcium current

NEURON {
    SUFFIX Ca_HVA
    USEION ca READ cai, cao WRITE ica VALENCE 2
    RANGE gCa_HVAbar, gCa_HVA, ica_hva, eca_hva, m, h, minf, hinf, mtau, htau
    GLOBAL vhalf_m, k_m, tau_m, vhalf_h, k_h, tau_h
}

UNITS {
    (mA) = (milliamp)
    (mV) = (millivolt)
    (S) = (siemens)
}

PARAMETER {
    gCa_HVAbar = 0.0002 (S/cm2)
    vhalf_m = -55 (mV)
    k_m = 6 (mV)
    tau_m = 1 (ms)
    vhalf_h = -60 (mV)
    k_h = -7 (mV)
    tau_h = 80 (ms)
}

ASSIGNED {
    v (mV)
    cai (mM)
    cao (mM)
    ica (mA/cm2)
    ica_hva (mA/cm2)
    eca_hva (mV)
    gCa_HVA (S/cm2)
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
    eca_hva = 0.0430866 * (celsius + 273.15) * log(cao / cai_safe)
    gCa_HVA = gCa_HVAbar * m^2 * h
    ica_hva = gCa_HVA * (v - eca_hva)
    ica = ica_hva
}

DERIVATIVE states {
    rates(v)
    m' = (minf - m) / mtau
    h' = (hinf - h) / htau
}

PROCEDURE rates(v (mV)) {
    minf = 1 / (1 + exp(-(v - vhalf_m) / k_m))
    hinf = 1 / (1 + exp(-(v - vhalf_h) / k_h))
    mtau = tau_m
    htau = tau_h
}
