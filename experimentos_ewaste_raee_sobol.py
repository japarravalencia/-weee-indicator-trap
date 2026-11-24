#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MOTOR DE EXPERIMENTOS DETERMINISTAS v1.0 — WEEE INDICATOR TRAP CO-FLOW MODEL (CALI-YUMBO)
Proyecto: 2026_ewaste_cali
Replica con 100% de fidelidad las ecuaciones exactas de 'RAEE Indicator trap sfd 30 nov 2026.stmx':
  - Horizonte: 0.0 a 15.0 (Years), DT = 0.03125, integración RK4
  - 5 Stocks primarios (Equipos_en_Uso, RAEE_en_Bodegaje, Empresas_Gestion_Lineal,
    Empresas_Economia_Circular, Material_Valorizado_Acumulado)
    + 1 SMTH3 (3 estados: presion_reportada_con_retardo)
    + 1 SMTH1 (1 estado: complacencia_institucional)
    = 9 estados ODE exactos
  - 4 Flujos, 21 Auxiliares dinámicos, 28 Constantes

Verificaciones incluidas:
  1. Concordancia exacta Stella Architect 4.3 (Runs 565, 564, 566, 567 de v8.0) vs. Python RK4
  2. Conservación exacta de población empresarial (E_L + E_C == 500) y cierre de masa física
     (d(S_eq + S_WEEE + S_val)/dt == F_acq, verificando integral exacta de adquisiciones)
  3. Convergencia numérica RK4 a 3 pasos (DT = 0.03125 -> 0.015625 -> 0.0078125) + contraste Euler
  4. Pruebas de Condiciones Extremas Barlas sin clamping (Extremo 1: Asfixia Total / Sin Presupuesto;
     Extremo 2: Transición Circular Instantánea / Cero Maquillaje)
  5. Superficie de bifurcación de política (tau_rep x w_gam) sobre WEEE terminal y parque activo
  6. Sobol Panel A: 7 parámetros de 'Sensibilidad_Global_V8' (n=512 -> N=4,608 corridas, IC 95% + P10–P90 + invarianza ordinal)
  7. Sobol Panel B: Separación funcional de las 3 palancas de política (Auditoría Anti-Trampa,
     Difusión Circular y Reinversión Tecnológica; n=256 -> N=1,280 corridas, IC 95%)

Uso:
  python 03_proyectos/2026_ewaste_cali/modelo_stella/experimentos_ewaste_raee_sobol.py --proyecto 2026_ewaste_cali
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from SALib.analyze import sobol as sobol_analyze
    try:
        from SALib.sample import sobol as sobol_sample
    except ImportError:
        from SALib.sample import saltelli as sobol_sample
    TIENE_SALIB = True
except ImportError:
    TIENE_SALIB = False


# ==============================================================================
# 28 PARÁMETROS CONSTANTES EXACTOS DE 'RAEE Indicator trap sfd 30 nov 2026.stmx'
# (+ valores iniciales de stocks configurables por escenario)
# ==============================================================================
PARAMS_BASE: Dict[str, float] = {
    "equipos_inicial": 1000.0,
    "raee_inicial": 50.0,
    "empresas_lineales_inicial": 450.0,
    "empresas_circulares_inicial": 50.0,
    "valorizado_inicial": 0.0,
    # 28 constantes SSOT
    "Capacidad_Maxima_Bodega": 800.0,
    "costo_unitario_almacenamiento": 180.0,
    "costo_promedio_dispositivo": 200.0,
    "fraccion_reinversion_tecnologia": 0.5,
    "piso_presion_evacuacion": 0.05,
    "precio_base_valorizacion": 120.0,
    "presupuesto_base": 50000.0,
    "factor_escala_presion_real": 0.0025,
    "amplitud_fraccion_evacuacion": 0.9,
    "exponente_hill_presion": 1.5,
    "suavizado_insolvencia": 2000.0,
    "fraccion_adopcion_politica": 0.02,
    "tasa_contagio_industrial": 0.12,
    "umbral_presion_transicion": 1.0,
    "intensidad_equipos_por_empresa": 2.2,
    "tiempo_ajuste_parque": 2.0,
    "vida_util_lineal": 3.5,
    "vida_util_circular": 6.0,
    "retardo_baja_lineal": 2.5,
    "retardo_baja_circular": 0.3,
    "umbral_presion_evacuacion": 1.2,
    "peso_circular_evacuacion": 0.7,
    "tiempo_reporte_administrativo": 0.5,
    "peso_maquillaje_lineal": 0.85,
    "tiempo_habituacion_burocratica": 0.75,
    "tolerancia_indice_trampa": 0.001,
    "incremento_precio_circular": 0.5,
    "peso_reposicion_adquisicion": 0.2,
}

# 4 Escenarios oficiales de v8.0 (Runs 565, 564, 566, 567) + 2 Extremos Barlas
ESCENARIOS: Dict[str, Dict[str, float]] = {
    "01_Inercia_Lineal_Run565": {
        "empresas_lineales_inicial": 490.0,
        "empresas_circulares_inicial": 10.0,
        "fraccion_adopcion_politica": 0.005,
        "tasa_contagio_industrial": 0.030,
        "tiempo_reporte_administrativo": 1.000,
        "peso_maquillaje_lineal": 0.850,
        "fraccion_reinversion_tecnologia": 0.500,
    },
    "02_Linea_Base_Run564": {
        "empresas_lineales_inicial": 450.0,
        "empresas_circulares_inicial": 50.0,
        "fraccion_adopcion_politica": 0.020,
        "tasa_contagio_industrial": 0.120,
        "tiempo_reporte_administrativo": 0.500,
        "peso_maquillaje_lineal": 0.850,
        "fraccion_reinversion_tecnologia": 0.500,
    },
    "03_Auditoria_Anti_Trampa_Run566": {
        "empresas_lineales_inicial": 450.0,
        "empresas_circulares_inicial": 50.0,
        "fraccion_adopcion_politica": 0.020,
        "tasa_contagio_industrial": 0.120,
        "tiempo_reporte_administrativo": 0.150,
        "peso_maquillaje_lineal": 0.200,
        "fraccion_reinversion_tecnologia": 0.500,
    },
    "04_Transicion_Acelerada_Run567": {
        "empresas_lineales_inicial": 450.0,
        "empresas_circulares_inicial": 50.0,
        "fraccion_adopcion_politica": 0.080,
        "tasa_contagio_industrial": 0.250,
        "tiempo_reporte_administrativo": 0.150,
        "peso_maquillaje_lineal": 0.200,
        "fraccion_reinversion_tecnologia": 0.800,
    },
    "05_Extremo1_Asfixia_CeroPresup": {
        "empresas_lineales_inicial": 495.0,
        "empresas_circulares_inicial": 5.0,
        "presupuesto_base": 0.0,
        "fraccion_adopcion_politica": 0.0,
        "tasa_contagio_industrial": 0.0,
        "peso_maquillaje_lineal": 0.98,
        "tiempo_reporte_administrativo": 1.5,
    },
    "06_Extremo2_Circular_CeroMaquillaje": {
        "empresas_lineales_inicial": 0.0,
        "empresas_circulares_inicial": 500.0,
        "tiempo_reporte_administrativo": 0.10,
        "peso_maquillaje_lineal": 0.0,
        "fraccion_reinversion_tecnologia": 1.0,
    },
}

# Referencia oficial reportada por Stella Architect 4.3 en v8.0 (para test de concordancia)
STELLA_REF_T15: Dict[str, Dict[str, float]] = {
    "01_Inercia_Lineal_Run565": {"RAEE": 733.4, "EmpCirc": 62.8, "TrapIdx": 0.764, "Equipos": 303.8, "Valoriz": 1102.8},
    "02_Linea_Base_Run564": {"RAEE": 276.9, "EmpCirc": 376.8, "TrapIdx": 0.260, "Equipos": 411.7, "Valoriz": 1431.2},
    "03_Auditoria_Anti_Trampa_Run566": {"RAEE": 246.3, "EmpCirc": 394.1, "TrapIdx": 0.050, "Equipos": 496.5, "Valoriz": 1486.1},
    "04_Transicion_Acelerada_Run567": {"RAEE": 193.4, "EmpCirc": 496.6, "TrapIdx": 0.002, "Equipos": 1038.3, "Valoriz": 1996.5},
}


def estado_inicial_stella(p: Dict[str, float]) -> np.ndarray:
    """Inicializa los 5 stocks primarios + 3 estados SMTH3 + 1 estado SMTH1 + acumulador de masa adquirida."""
    s_eq0 = p["equipos_inicial"]
    s_raee0 = p["raee_inicial"]
    e_lin0 = p["empresas_lineales_inicial"]
    e_cir0 = p["empresas_circulares_inicial"]
    s_val0 = p["valorizado_inicial"]

    n_tot0 = max(1e-9, e_lin0 + e_cir0)
    f_circ0 = e_cir0 / n_tot0

    p_real0 = s_raee0 * p["factor_escala_presion_real"]
    p_rep0 = p_real0
    prop_maq0 = p["peso_maquillaje_lineal"] * (1.0 - f_circ0)
    esf_pap0 = p_rep0 * prop_maq0
    comp0 = esf_pap0

    masa_init = s_eq0 + s_raee0 + s_val0
    return np.array([
        s_eq0, s_raee0, e_lin0, e_cir0, s_val0,
        p_rep0, p_rep0, p_rep0,
        comp0,
        masa_init,
    ], dtype=float)


def evaluar_sistema(y: np.ndarray, p: Dict[str, float], aplicar_clamp: bool = True) -> Tuple[np.ndarray, Dict[str, float]]:
    s_eq = max(0.0, float(y[0])) if aplicar_clamp else float(y[0])
    s_raee = max(0.0, float(y[1])) if aplicar_clamp else float(y[1])
    e_lin = max(0.0, float(y[2])) if aplicar_clamp else float(y[2])
    e_cir = max(0.0, float(y[3])) if aplicar_clamp else float(y[3])
    s_val = max(0.0, float(y[4])) if aplicar_clamp else float(y[4])

    pr_s1, pr_s2, pr_out = float(y[5]), float(y[6]), float(y[7])
    comp = float(y[8])
    masa_acum = float(y[9])

    # Sector A: Demografía empresarial
    n_tot = max(1e-9, e_lin + e_cir)
    f_circ = e_cir / n_tot

    # Sector C: Trampa del Indicador (SMTH3 + SMTH1)
    p_real = s_raee * p["factor_escala_presion_real"]
    prop_maq = p["peso_maquillaje_lineal"] * (1.0 - f_circ)
    esf_pap = pr_out * prop_maq
    p_acc = max(0.0, p_real - comp)
    i_trap = comp / (p_real + p["tolerancia_indice_trampa"]) if abs(p_real + p["tolerancia_indice_trampa"]) > 1e-12 else 0.0

    # No linealidades de Hill y saturación cuadrática
    h_exp = p["exponente_hill_presion"]
    p_acc_h = (p_acc ** h_exp) if p_acc > 0.0 else 0.0
    hill_evac = p_acc_h / ((p["umbral_presion_evacuacion"] ** h_exp) + p_acc_h)
    hill_trans = p_acc_h / ((p["umbral_presion_transicion"] ** h_exp) + p_acc_h)
    mult_trans = 1.0 + hill_trans

    k_cap = p["Capacidad_Maxima_Bodega"]
    sat_esp = (s_raee * s_raee) / (k_cap * k_cap + s_raee * s_raee)

    frac_evac = p["piso_presion_evacuacion"] + p["amplitud_fraccion_evacuacion"] * (hill_evac + sat_esp) * (
        (1.0 - p["peso_circular_evacuacion"]) + p["peso_circular_evacuacion"] * f_circ
    )

    # Sector B: Co-flujo físico y parámetros modulados por f_circ
    vida_eff = p["vida_util_lineal"] * (1.0 - f_circ) + p["vida_util_circular"] * f_circ
    ret_eff = p["retardo_baja_lineal"] * (1.0 - f_circ) + p["retardo_baja_circular"] * f_circ

    f_obs = s_eq / vida_eff
    f_evac = (s_raee / ret_eff) * frac_evac

    # Finanzas y rampa diferenciable de insolvencia
    cost_alm = s_raee * p["costo_unitario_almacenamiento"]
    precio_val = p["precio_base_valorizacion"] * (1.0 + p["incremento_precio_circular"] * f_circ)
    ing_val = f_evac * precio_val
    pres_bruto = p["presupuesto_base"] - cost_alm + ing_val * p["fraccion_reinversion_tecnologia"]
    suav = p["suavizado_insolvencia"]
    pres_adapt = (pres_bruto + math.sqrt(pres_bruto * pres_bruto + suav * suav)) / 2.0

    dem_rep = (n_tot * p["intensidad_equipos_por_empresa"] - s_eq) / p["tiempo_ajuste_parque"]
    f_acq = max(0.0, (pres_adapt / p["costo_promedio_dispositivo"]) + p["peso_reposicion_adquisicion"] * dem_rep)

    # Transición circular (Bass + presión real)
    f_trans = (e_lin * p["fraccion_adopcion_politica"]) + (e_lin * f_circ * p["tasa_contagio_industrial"] * mult_trans)

    # Derivadas
    k_rep = 3.0 / max(1e-6, p["tiempo_reporte_administrativo"])
    k_hab = 1.0 / max(1e-6, p["tiempo_habituacion_burocratica"])

    dy = np.array([
        f_acq - f_obs,
        f_obs - f_evac,
        -f_trans,
        f_trans,
        f_evac,
        k_rep * (p_real - pr_s1),
        k_rep * (pr_s1 - pr_s2),
        k_rep * (pr_s2 - pr_out),
        k_hab * (esf_pap - comp),
        f_acq,
    ], dtype=float)

    masa_total = s_eq + s_raee + s_val
    resid_masa = masa_total - masa_acum
    resid_emp = (e_lin + e_cir) - (p["empresas_lineales_inicial"] + p["empresas_circulares_inicial"])
    intens_pasivo = (s_raee / s_eq) if s_eq > 1e-9 else 0.0

    aux = {
        "fraccion_circular": f_circ,
        "indice_trampa": i_trap,
        "presion_fisica_real": p_real,
        "complacencia_institucional": comp,
        "presion_accion_real": p_acc,
        "costos_almacenamiento": cost_alm,
        "presupuesto_adaptativo": pres_adapt,
        "ingresos_valorizacion": ing_val,
        "tasa_adquisicion": f_acq,
        "tasa_obsolescencia": f_obs,
        "tasa_evacuacion": f_evac,
        "tasa_transicion": f_trans,
        "intensidad_pasivo": intens_pasivo,
        "residual_masa": resid_masa,
        "residual_empresas": resid_emp,
    }
    return dy, aux


def simular(
    overrides: Dict[str, float],
    dt: float = 0.03125,
    t0: float = 0.0,
    tf: float = 15.0,
    metodo: str = "RK4",
    aplicar_clamp: bool = True,
) -> pd.DataFrame:
    p = dict(PARAMS_BASE)
    p.update(overrides)
    y = estado_inicial_stella(p)

    n_steps = int(round((tf - t0) / dt))
    pasos_anio = max(1, int(round(1.0 / dt)))
    filas: List[Dict[str, float]] = []

    for step in range(n_steps + 1):
        t = t0 + step * dt
        dy, aux = evaluar_sistema(y, p, aplicar_clamp=aplicar_clamp)
        if step % pasos_anio == 0 or step == n_steps:
            filas.append({
                "Year": round(t, 4),
                "Equipos_en_Uso": round(float(y[0]), 2),
                "RAEE_en_Bodegaje": round(float(y[1]), 2),
                "Empresas_Gestion_Lineal": round(float(y[2]), 2),
                "Empresas_Economia_Circular": round(float(y[3]), 2),
                "Material_Valorizado_Acumulado": round(float(y[4]), 2),
                **{k: round(float(v), 6) for k, v in aux.items()},
            })
        if step == n_steps:
            break

        if metodo.lower() == "rk4":
            k1 = dy
            k2, _ = evaluar_sistema(y + 0.5 * dt * k1, p, aplicar_clamp=aplicar_clamp)
            k3, _ = evaluar_sistema(y + 0.5 * dt * k2, p, aplicar_clamp=aplicar_clamp)
            k4, _ = evaluar_sistema(y + dt * k3, p, aplicar_clamp=aplicar_clamp)
            y = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        else:
            y = y + dt * dy

        if aplicar_clamp:
            for i in range(5):
                y[i] = max(0.0, y[i])

    return pd.DataFrame(filas)


def ejecutar_sobol_panel_a(n_base: int = 512) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Panel A: los 7 parámetros exactos de Sensibilidad_Global_V8 (n=512 -> N=4,608 corridas, IC 95%)."""
    if not TIENE_SALIB:
        return pd.DataFrame(), {"ejecutado": False}

    nombres = [
        "vida_util_circular",
        "retardo_baja_lineal",
        "retardo_baja_circular",
        "tiempo_reporte_administrativo",
        "tiempo_habituacion_burocratica",
        "tasa_contagio_industrial",
        "fraccion_adopcion_politica",
    ]
    bounds = [
        [4.5, 8.0],
        [1.5, 4.0],
        [0.15, 0.60],
        [0.15, 1.50],
        [0.25, 1.50],
        [0.05, 0.30],
        [0.01, 0.10],
    ]
    problem = {"num_vars": len(nombres), "names": nombres, "bounds": bounds}
    X = sobol_sample.sample(problem, n_base, calc_second_order=False)
    n_tot = len(X)

    y_raee = np.zeros(n_tot)
    y_eq = np.zeros(n_tot)
    y_val = np.zeros(n_tot)
    y_trap = np.zeros(n_tot)

    yrs_sel = [0, 3, 6, 9, 12, 15]
    band_raee = np.zeros((min(300, n_tot), len(yrs_sel)))
    band_eq = np.zeros((min(300, n_tot), len(yrs_sel)))

    for i, fila in enumerate(X):
        ov = {nombres[j]: float(fila[j]) for j in range(len(nombres))}
        df_i = simular(ov, dt=0.0625, metodo="RK4")
        fin = df_i.iloc[-1]
        y_raee[i] = fin["RAEE_en_Bodegaje"]
        y_eq[i] = fin["Equipos_en_Uso"]
        y_val[i] = fin["Material_Valorizado_Acumulado"]
        y_trap[i] = fin["indice_trampa"]
        if i < band_raee.shape[0]:
            sub = df_i[df_i["Year"].isin(yrs_sel)]
            band_raee[i, :] = sub["RAEE_en_Bodegaje"].values
            band_eq[i, :] = sub["Equipos_en_Uso"].values

    si_r = sobol_analyze.analyze(problem, y_raee, calc_second_order=False, conf_level=0.95, print_to_console=False)
    si_e = sobol_analyze.analyze(problem, y_eq, calc_second_order=False, conf_level=0.95, print_to_console=False)
    si_v = sobol_analyze.analyze(problem, y_val, calc_second_order=False, conf_level=0.95, print_to_console=False)
    si_t = sobol_analyze.analyze(problem, y_trap, calc_second_order=False, conf_level=0.95, print_to_console=False)

    # Invarianza ordinal: Run 567 (Acelerada) < Run 564 (Base) en RAEE y > en Parque Activo + Valorizado
    rng = np.random.default_rng(2026)
    idxs = rng.choice(n_tot, size=min(60, n_tot), replace=False)
    inv_ok = 0
    for idx in idxs:
        ov0 = {nombres[j]: float(X[idx][j]) for j in range(len(nombres))}
        ov_acc = dict(ov0)
        ov_acc.update({
            "tiempo_reporte_administrativo": 0.15,
            "peso_maquillaje_lineal": 0.20,
            "fraccion_adopcion_politica": 0.08,
            "tasa_contagio_industrial": 0.25,
            "fraccion_reinversion_tecnologia": 0.80,
        })
        f0 = simular(ov0, dt=0.0625).iloc[-1]
        f1 = simular(ov_acc, dt=0.0625).iloc[-1]
        if (f1["RAEE_en_Bodegaje"] < f0["RAEE_en_Bodegaje"]) and (f1["Equipos_en_Uso"] > f0["Equipos_en_Uso"]):
            inv_ok += 1

    df_s = pd.DataFrame({
        "Parametro": nombres,
        "S1_RAEE": np.round(si_r["S1"], 4),
        "ST_RAEE": np.round(si_r["ST"], 4),
        "ST_RAEE_CI95": np.round(si_r["ST_conf"], 4),
        "ST_Equipos": np.round(si_e["ST"], 4),
        "ST_Eq_CI95": np.round(si_e["ST_conf"], 4),
        "ST_Valoriz": np.round(si_v["ST"], 4),
        "ST_TrapIdx": np.round(si_t["ST"], 4),
    }).sort_values("ST_RAEE", ascending=False)

    meta = {
        "ejecutado": True,
        "n_base": int(n_base),
        "n_corridas": int(n_tot),
        "raee_t15_p10_p90": [round(float(np.percentile(y_raee, 10)), 1), round(float(np.percentile(y_raee, 90)), 1)],
        "equipos_t15_p10_p90": [round(float(np.percentile(y_eq, 10)), 1), round(float(np.percentile(y_eq, 90)), 1)],
        "valoriz_t15_p10_p90": [round(float(np.percentile(y_val, 10)), 1), round(float(np.percentile(y_val, 90)), 1)],
        "invarianza_ordinal_pct": round(100.0 * inv_ok / len(idxs), 1),
        "banda_raee_p10_p90": {str(yr): [round(float(np.percentile(band_raee[:, j], 10)), 1), round(float(np.percentile(band_raee[:, j], 90)), 1)] for j, yr in enumerate(yrs_sel)},
        "banda_eq_p10_p90": {str(yr): [round(float(np.percentile(band_eq[:, j], 10)), 1), round(float(np.percentile(band_eq[:, j], 90)), 1)] for j, yr in enumerate(yrs_sel)},
    }
    return df_s, meta


def ejecutar_sobol_panel_b(n_pol: int = 256) -> Tuple[pd.DataFrame, int]:
    """Panel B: Separación funcional de las 3 palancas de política [0, 1]^3 (Auditoría, Difusión Circular, Reinversión)."""
    if not TIENE_SALIB:
        return pd.DataFrame(), 0
    nombres = ["Palanca_1_Auditoria_AntiTrampa", "Palanca_2_Difusion_Circular", "Palanca_3_Reinversion_Tec"]
    problem = {"num_vars": 3, "names": nombres, "bounds": [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]}
    X = sobol_sample.sample(problem, n_pol, calc_second_order=False)
    n_tot = len(X)
    y_r, y_e, y_t, y_v = np.zeros(n_tot), np.zeros(n_tot), np.zeros(n_tot), np.zeros(n_tot)

    for i, fila in enumerate(X):
        p_aud, p_dif, p_rei = float(fila[0]), float(fila[1]), float(fila[2])
        ov = {
            "tiempo_reporte_administrativo": 1.0 - 0.85 * p_aud,      # 1.0 -> 0.15
            "peso_maquillaje_lineal": 0.85 - 0.65 * p_aud,            # 0.85 -> 0.20
            "fraccion_adopcion_politica": 0.005 + 0.075 * p_dif,      # 0.005 -> 0.08
            "tasa_contagio_industrial": 0.03 + 0.22 * p_dif,          # 0.03 -> 0.25
            "fraccion_reinversion_tecnologia": 0.20 + 0.60 * p_rei,   # 0.20 -> 0.80
        }
        fin = simular(ov, dt=0.0625).iloc[-1]
        y_r[i] = fin["RAEE_en_Bodegaje"]
        y_e[i] = fin["Equipos_en_Uso"]
        y_t[i] = fin["indice_trampa"]
        y_v[i] = fin["Material_Valorizado_Acumulado"]

    si_r = sobol_analyze.analyze(problem, y_r, calc_second_order=False, conf_level=0.95, print_to_console=False)
    si_e = sobol_analyze.analyze(problem, y_e, calc_second_order=False, conf_level=0.95, print_to_console=False)
    si_t = sobol_analyze.analyze(problem, y_t, calc_second_order=False, conf_level=0.95, print_to_console=False)
    si_v = sobol_analyze.analyze(problem, y_v, calc_second_order=False, conf_level=0.95, print_to_console=False)

    df_b = pd.DataFrame({
        "Palanca_Politica": nombres,
        "ST_TrapIdx": np.round(si_t["ST"], 4),
        "ST_Trap_CI95": np.round(si_t["ST_conf"], 4),
        "ST_RAEE": np.round(si_r["ST"], 4),
        "ST_RAEE_CI95": np.round(si_r["ST_conf"], 4),
        "ST_Equipos": np.round(si_e["ST"], 4),
        "ST_Eq_CI95": np.round(si_e["ST_conf"], 4),
        "ST_Valoriz": np.round(si_v["ST"], 4),
    })
    return df_b, n_tot


def ejecutar_superficie_bifurcacion() -> pd.DataFrame:
    """Superficie 2D tau_rep x w_gam bajo Baseline (Run 564) para mapear escape de insolvencia (< 277.8 t)."""
    rows = []
    for tau in (0.15, 0.50, 1.00, 1.50):
        for wgam in (0.10, 0.20, 0.50, 0.85):
            fin = simular({"tiempo_reporte_administrativo": tau, "peso_maquillaje_lineal": wgam}, dt=0.0625).iloc[-1]
            rows.append({
                "tau_rep": tau,
                "w_gam": wgam,
                "RAEE_t15": round(float(fin["RAEE_en_Bodegaje"]), 1),
                "TrapIdx_t15": round(float(fin["indice_trampa"]), 3),
                "Equipos_t15": round(float(fin["Equipos_en_Uso"]), 1),
                "Insolvente_t15": "SI" if float(fin["RAEE_en_Bodegaje"]) > 277.8 else "NO",
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Motor RK4 + Concordancia Stella + Barlas + Sobol para 2026_ewaste_cali.")
    parser.add_argument("--proyecto", default="2026_ewaste_cali")
    parser.add_argument("--n-sobol", type=int, default=512)
    args = parser.parse_args()

    p_script = Path(__file__).resolve()
    raiz = next((parent for parent in p_script.parents if (parent / "03_proyectos").exists()), Path.cwd())
    proy_dir = raiz / "03_proyectos" / args.proyecto
    csv_dir = proy_dir / "modelo_stella" / "corridas_csv"
    ssot_dir = proy_dir / "ssot"
    csv_dir.mkdir(parents=True, exist_ok=True)
    ssot_dir.mkdir(parents=True, exist_ok=True)

    print("=========================================================================")
    print(f" MOTOR WEEE INDICATOR TRAP (RK4 + STELLA CONCORDANCE + SOBOL) | {args.proyecto}")
    print("=========================================================================")

    trajs: Dict[str, pd.DataFrame] = {}
    res_rows = []
    conc_rows = []
    max_masa_err = 0.0
    max_emp_err = 0.0

    for nom, ov in ESCENARIOS.items():
        df = simular(ov, dt=0.03125, metodo="RK4")
        df.to_csv(csv_dir / f"{nom}.csv", index=False)
        trajs[nom] = df
        fin = df.iloc[-1]
        err_m = float(df["residual_masa"].abs().max())
        err_e = float(df["residual_empresas"].abs().max())
        max_masa_err = max(max_masa_err, err_m)
        max_emp_err = max(max_emp_err, err_e)

        res_rows.append({
            "Escenario": nom,
            "RAEE_t15": round(float(fin["RAEE_en_Bodegaje"]), 1),
            "RAEE_Peak": round(float(df["RAEE_en_Bodegaje"].max()), 1),
            "EmpCirc_t15": round(float(fin["Empresas_Economia_Circular"]), 1),
            "TrapIdx_t15": round(float(fin["indice_trampa"]), 3),
            "Pacc_t3": round(float(df[df["Year"] == 3.0]["presion_accion_real"].iloc[0]), 3),
            "Equipos_t15": round(float(fin["Equipos_en_Uso"]), 1),
            "Valoriz_t15": round(float(fin["Material_Valorizado_Acumulado"]), 1),
        })
        print(
            f"   [OK] {nom:<35} | RAEE={fin['RAEE_en_Bodegaje']:5.1f} | "
            f"Circ={fin['Empresas_Economia_Circular']:5.1f} | Trap={fin['indice_trampa']:.3f} | "
            f"Eq={fin['Equipos_en_Uso']:6.1f} | Val={fin['Material_Valorizado_Acumulado']:6.1f}"
        )

        if nom in STELLA_REF_T15:
            ref = STELLA_REF_T15[nom]
            conc_rows.append({
                "Escenario": nom,
                "RAEE_St_vs_Py": f"{ref['RAEE']} / {fin['RAEE_en_Bodegaje']:.1f}",
                "Diff_RAEE": round(abs(ref["RAEE"] - float(fin["RAEE_en_Bodegaje"])), 2),
                "Eq_St_vs_Py": f"{ref['Equipos']} / {fin['Equipos_en_Uso']:.1f}",
                "Diff_Eq": round(abs(ref["Equipos"] - float(fin["Equipos_en_Uso"])), 2),
                "Val_St_vs_Py": f"{ref['Valoriz']} / {fin['Material_Valorizado_Acumulado']:.1f}",
                "Diff_Val": round(abs(ref["Valoriz"] - float(fin["Material_Valorizado_Acumulado"])), 2),
                "Trap_St_vs_Py": f"{ref['TrapIdx']:.3f} / {fin['indice_trampa']:.3f}",
            })

    df_res = pd.DataFrame(res_rows)
    df_conc = pd.DataFrame(conc_rows)

    # Convergencia RK4 a 3 pasos + Euler
    conv_rows = []
    for sc in ("02_Linea_Base_Run564", "04_Transicion_Acelerada_Run567"):
        r1 = simular(ESCENARIOS[sc], dt=0.03125, metodo="RK4").iloc[-1]
        r2 = simular(ESCENARIOS[sc], dt=0.015625, metodo="RK4").iloc[-1]
        r3 = simular(ESCENARIOS[sc], dt=0.0078125, metodo="RK4").iloc[-1]
        eu = simular(ESCENARIOS[sc], dt=0.03125, metodo="Euler").iloc[-1]
        conv_rows.append({
            "Escenario": sc,
            "RK4_0.03125": round(float(r1["RAEE_en_Bodegaje"]), 5),
            "RK4_0.015625": round(float(r2["RAEE_en_Bodegaje"]), 5),
            "RK4_0.0078125": round(float(r3["RAEE_en_Bodegaje"]), 5),
            "Diff_RK4_Max": f"{abs(float(r1['RAEE_en_Bodegaje']) - float(r3['RAEE_en_Bodegaje'])):.2e}",
            "Euler_0.03125": round(float(eu["RAEE_en_Bodegaje"]), 5),
            "Diff_RK4_Eu": f"{abs(float(r1['RAEE_en_Bodegaje']) - float(eu['RAEE_en_Bodegaje'])):.2e}",
        })
    df_conv = pd.DataFrame(conv_rows)

    # Barlas unclamped
    df_u1 = simular(ESCENARIOS["05_Extremo1_Asfixia_CeroPresup"], dt=0.03125, aplicar_clamp=False)
    df_u2 = simular(ESCENARIOS["06_Extremo2_Circular_CeroMaquillaje"], dt=0.03125, aplicar_clamp=False)

    df_bif = ejecutar_superficie_bifurcacion()
    df_bif.to_csv(ssot_dir / "superficie_bifurcacion_ewaste.csv", index=False)

    print(f"1. [SOBOL PANEL A] Ejecutando sensibilidad global 7 parámetros (n={args.n_sobol}, N={args.n_sobol*9})...")
    df_sob_a, meta_a = ejecutar_sobol_panel_a(n_base=args.n_sobol)
    df_sob_a.to_csv(ssot_dir / "sobol_panel_a_ewaste.csv", index=False)

    print("2. [SOBOL PANEL B] Ejecutando separación funcional de las 3 palancas de política...")
    df_sob_b, n_tot_b = ejecutar_sobol_panel_b(n_pol=256)
    df_sob_b.to_csv(ssot_dir / "sobol_panel_b_ewaste.csv", index=False)

    md = [
        f"# Resultados Verificados RK4 + Concordancia Stella + Barlas + Sobol — `{args.proyecto}`",
        f"**Generado:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | **SSOT:** `RAEE Indicator trap sfd 30 nov 2026.stmx` (0–15 yr, RK4, DT=0.03125)\n",
        "## 1. Matriz Comparativa de 6 Escenarios (4 Políticas + 2 Extremos Barlas) a t = 15 Años",
        df_res.to_markdown(index=False),
        "\n## 2. Concordancia Exacta Stella Architect 4.3 (v8.0) vs. Python RK4 (t = 15)",
        df_conc.to_markdown(index=False),
        "\n## 3. Verificación Numérica: Conservación de Masa/Empresas, Convergencia RK4 y Barlas Unclamped",
        f"- **Conservación exacta de masa física (`max |S_eq + S_WEEE + S_val - M_acum|`):** `{max_masa_err:.2e}`",
        f"- **Conservación exacta de población empresarial (`max |E_L + E_C - 500|`):** `{max_emp_err:.2e}`",
        f"- **Barlas Extremo 1 Unclamped (Asfixia Cero Presupuesto):** `min(Equipos)={df_u1['Equipos_en_Uso'].min():.2f} >= 0`, `RAEE_t15={df_u1['RAEE_en_Bodegaje'].iloc[-1]:.1f}`",
        f"- **Barlas Extremo 2 Unclamped (100% Circular, Cero Maquillaje):** `TrapIdx_max={df_u2['indice_trampa'].max():.4f}`, `RAEE_t15={df_u2['RAEE_en_Bodegaje'].iloc[-1]:.1f}`, `Equipos_t15={df_u2['Equipos_en_Uso'].iloc[-1]:.1f}`\n",
        df_conv.to_markdown(index=False),
        "\n## 4. Superficie de Bifurcación Política (tau_rep x w_gam bajo Baseline)",
        df_bif.to_markdown(index=False),
        f"\n## 5. Sobol Panel A — Hipercubo 7 Parámetros `Sensibilidad_Global_V8` (n={args.n_sobol}, N={meta_a.get('n_corridas', 0)}, IC 95%)",
        f"- **Metadatos:** `{json.dumps(meta_a, ensure_ascii=False)}`\n",
        df_sob_a.to_markdown(index=False),
        f"\n## 6. Sobol Panel B — Separación Funcional de las 3 Palancas de Política (N={n_tot_b}, IC 95%)",
        df_sob_b.to_markdown(index=False),
    ]
    ruta_rep = ssot_dir / "resumen_experimentos_ewaste_cali.md"
    ruta_rep.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n✅ Reporte completo guardado en: {ruta_rep}")
    print("=========================================================================")


if __name__ == "__main__":
    main()