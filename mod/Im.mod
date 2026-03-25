TITLE M-type potassium current

NEURON {
    SUFFIX Im
    USEION k READ ek WRITE ik
    RANGE gImbar, gIm, ik_im, p, pinf, ptau
    GLOBAL vhalf, k, tau
}

UNITS {
    (mA) = (milliamp)
    (mV) = (millivolt)
    (S) = (siemens)
}

PARAMETER {
    gImbar = 0.00008 (S/cm2)
    vhalf = -35 (mV)
    k = 10 (mV)
    tau = 40 (ms)
}

ASSIGNED {
    v (mV)
    ek (mV)
    ik (mA/cm2)
    ik_im (mA/cm2)
    gIm (S/cm2)
    pinf
    ptau (ms)
}

STATE {
    p
}

INITIAL {
    rates(v)
    p = pinf
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    gIm = gImbar * p
    ik_im = gIm * (v - ek)
    ik = ik_im
}

DERIVATIVE states {
    rates(v)
    p' = (pinf - p) / ptau
}

PROCEDURE rates(v (mV)) {
    pinf = 1 / (1 + exp(-(v - vhalf) / k))
    ptau = tau
}
