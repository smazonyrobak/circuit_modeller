TITLE Calcium-activated SK potassium current

NEURON {
    SUFFIX SK_E2
    USEION ca READ cai
    USEION k READ ek WRITE ik
    RANGE gSK_E2bar, gSK_E2, ik_sk, z, zinf, ztau
    GLOBAL Kd, hill_power, tau
}

UNITS {
    (mA) = (milliamp)
    (mV) = (millivolt)
    (mM) = (milli/liter)
    (S) = (siemens)
}

PARAMETER {
    gSK_E2bar = 0.0015 (S/cm2)
    Kd = 0.00035 (mM)
    hill_power = 4
    tau = 4 (ms)
}

ASSIGNED {
    v (mV)
    cai (mM)
    ek (mV)
    ik (mA/cm2)
    ik_sk (mA/cm2)
    gSK_E2 (S/cm2)
    zinf
    ztau (ms)
    cai_safe (mM)
}

STATE {
    z
}

INITIAL {
    rates(cai)
    z = zinf
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    gSK_E2 = gSK_E2bar * z
    ik_sk = gSK_E2 * (v - ek)
    ik = ik_sk
}

DERIVATIVE states {
    rates(cai)
    z' = (zinf - z) / ztau
}

PROCEDURE rates(cai (mM)) {
    cai_safe = cai
    if (cai_safe < 1e-7) {
        cai_safe = 1e-7
    }
    zinf = pow(cai_safe, hill_power) / (pow(cai_safe, hill_power) + pow(Kd, hill_power))
    ztau = tau
}
