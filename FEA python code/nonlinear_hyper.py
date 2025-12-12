# -*- coding: utf-8 -*-
"""
nonlinear_hyper.py
[FINAL CORRECTED VERSION]

Features:
- 2D Plane Stress/Strain (Compressible & Mixed u-p) with 'tl' and 'abaqus' models.
- 3D Compressible (Displacement-only) for Linear Hex8 / Tet4.

Fixes:
- Fixed 'AttributeError' by correcting indentation of assemble methods.
- Fixed 3D Load Calculation (Area-based).
- Fixed Stress Sign (Absolute Jacobian).
- Preserved ALL legacy 2D functions.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Sequence
import numpy as np
import math

# ---------------------------------------------------------------------------
# Sympy import (optional) and global storages
# ---------------------------------------------------------------------------
try:
    import sympy as sp
except ImportError:
    sp = None

_SYMPY_PK1_FUN_2D = None        
_S_SYMPY_SD_2D_TL = None        
_S_SYMPY_SD_2D_ABAQUS = None    
_MIXED_PK1_FUN_2D = None        

# [NEW] 3D Globals
_SYMPY_SD_3D_TL = None
_SYMPY_SD_3D_ABAQUS = None


# ---------------------------------------------------------------------------
# [NEW] Helper: Area Calculation (Ported from Linear Assembly Code)
# ---------------------------------------------------------------------------
def _area_of_nodes(X_nodes: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute area, centroid, and unit normal of a 3D face defined by 3 or 4 nodes.
    """
    m = X_nodes.shape[0]
    ctr = X_nodes.mean(axis=0)
    
    if m == 3: # Tri face
        a = X_nodes[1] - X_nodes[0]
        b = X_nodes[2] - X_nodes[0]
        n = np.cross(a, b)
        norm_n = np.linalg.norm(n)
        A = 0.5 * norm_n
        n_hat = n / (norm_n if norm_n > 0 else 1.0)
        return float(A), ctr, n_hat
        
    elif m == 4: # Quad face
        n1 = np.cross(X_nodes[1]-X_nodes[0], X_nodes[2]-X_nodes[0])
        n2 = np.cross(X_nodes[2]-X_nodes[0], X_nodes[3]-X_nodes[0])
        A = 0.5 * (np.linalg.norm(n1) + np.linalg.norm(n2))
        n_vec = n1 + n2
        norm_n = np.linalg.norm(n_vec)
        n_hat = n_vec / (norm_n if norm_n > 0 else 1.0)
        return float(A), ctr, n_hat
    else:
        return 0.0, ctr, np.array([0.,0.,1.])


# ---------------------------------------------------------------------------
# 2. Sympy Generators (Legacy Preserved)
# ---------------------------------------------------------------------------

def _build_sympy_pk1_tangent_2d():
    global _SYMPY_PK1_FUN_2D
    if _SYMPY_PK1_FUN_2D is not None: return _SYMPY_PK1_FUN_2D
    if sp is None: raise ImportError("sympy required.")
    F11, F12, F21, F22, mu_s, lam_s = sp.symbols("F11 F12 F21 F22 mu lam", real=True)
    F = sp.Matrix([[F11, F12], [F21, F22]])
    J = F.det(); FinvT = F.inv().T
    P = mu_s * (F - FinvT) + lam_s * sp.log(J) * FinvT
    P_flat = sp.Matrix([P[0, 0], P[0, 1], P[1, 0], P[1, 1]])
    F_flat = [F11, F12, F21, F22]
    A_flat = sp.zeros(4, 4)
    for iP in range(4):
        for iF in range(4): A_flat[iP, iF] = sp.diff(P_flat[iP], F_flat[iF])
    PK1_fun = sp.lambdify((F11, F12, F21, F22, mu_s, lam_s), (P_flat, A_flat), "numpy")
    _SYMPY_PK1_FUN_2D = PK1_fun
    return PK1_fun

def _compute_PK1_and_tangent_2d(F: np.ndarray, mu: float, lam: float):
    PK1_fun = _build_sympy_pk1_tangent_2d()
    P_flat, A_flat = PK1_fun(float(F[0,0]), float(F[0,1]), float(F[1,0]), float(F[1,1]), float(mu), float(lam))
    return np.asarray(P_flat, dtype=float).reshape(2, 2), np.asarray(A_flat, dtype=float).reshape(2, 2, 2, 2)

# ---------------------------------------------------------------------------
# [NEW] 3D Mixed u-p Formulation (PK1 & Tangent)
# ---------------------------------------------------------------------------
_MIXED_PK1_FUN_3D = None

def _build_mixed_pk1_tangent_3d():
    global _MIXED_PK1_FUN_3D
    if _MIXED_PK1_FUN_3D is not None: return _MIXED_PK1_FUN_3D
    if sp is None: raise ImportError("sympy required.")
    
    # 3D F components (9 vars)
    vars_str = "F11 F12 F13 F21 F22 F23 F31 F32 F33 mu p"
    F11,F12,F13,F21,F22,F23,F31,F32,F33, mu_s, p_s = sp.symbols(vars_str, real=True)
    
    F = sp.Matrix([[F11,F12,F13], [F21,F22,F23], [F31,F32,F33]])
    C = F.T * F
    J = F.det()
    FinvT = F.inv().T
    trC = sp.trace(C)
    
    # Incompressible Neo-Hookean (Mixed form)
    # P_iso = mu * J^(-2/3) * (F - (trC/3)*F^-T)
    # P_vol = J * p * F^-T
    Jm23 = J**(-sp.Rational(2, 3))
    P_iso = mu_s * (Jm23 * F - (Jm23 * trC / 3.0) * FinvT)
    P_vol = J * p_s * FinvT
    P = P_iso + P_vol
    
    # 4th Order Tensor A = dP/dF (Material Tangent)
    # Shape: (3, 3, 3, 3)
    A = sp.MutableDenseNDimArray.zeros(3, 3, 3, 3)
    
    # Generating derivatives
    # Note: To speed up, we flatten F for the loop or iterate explicitly
    # indices: i(row of P), J(col of P), k(row of F), L(col of F)
    for i in range(3):
        for Jind in range(3):
            for k in range(3):
                for L in range(3):
                    A[i, Jind, k, L] = sp.diff(P[i, Jind], F[k, L])
                    
    # Flatten output for lambdify
    A_vec = sp.Matrix([A[i, j, k, l] for i in range(3) for j in range(3) for k in range(3) for l in range(3)])
    
    args = (F11,F12,F13,F21,F22,F23,F31,F32,F33, mu_s, p_s)
    A_fun = sp.lambdify(args, A_vec, modules="numpy")
    _MIXED_PK1_FUN_3D = A_fun
    return A_fun

def _compute_PK1_tangent_mixed_3d(F: np.ndarray, mu: float, p: float) -> np.ndarray:
    A_fun = _build_mixed_pk1_tangent_3d()
    # Unpack F (3x3)
    args = [F[i,j] for i in range(3) for j in range(3)]
    args.append(float(mu))
    args.append(float(p))
    
    A_flat = A_fun(*args)
    return np.asarray(A_flat, dtype=float).reshape(3, 3, 3, 3)

def _build_mixed_pk1_tangent_2d():
    global _MIXED_PK1_FUN_2D
    if _MIXED_PK1_FUN_2D is not None: return _MIXED_PK1_FUN_2D
    if sp is None: raise ImportError("sympy required.")
    F11, F12, F21, F22, mu_s, p_s = sp.symbols("F11 F12 F21 F22 mu p", real=True)
    F = sp.Matrix([[F11, F12], [F21, F22]])
    C = F.T * F; J = F.det(); Jm23 = J**(-sp.Rational(2, 3)); trC = sp.trace(C); FinvT = F.inv().T
    P_iso = mu_s * (Jm23 * F - (Jm23 * trC / 3.0) * FinvT)
    P_vol = J * p_s * FinvT
    P = P_iso + P_vol
    A = sp.MutableDenseNDimArray.zeros(2, 2, 2, 2)
    for i in range(2):
        for Jind in range(2):
            for k in range(2):
                for L in range(2): A[i, Jind, k, L] = sp.diff(P[i, Jind], F[k, L])
    A_vec = sp.Matrix([A[i, Jind, k, L] for i in range(2) for Jind in range(2) for k in range(2) for L in range(2)])
    A_fun = sp.lambdify((F11, F12, F21, F22, mu_s, p_s), A_vec, modules="numpy")
    _MIXED_PK1_FUN_2D = A_fun
    return A_fun

def _compute_PK1_tangent_mixed_2d(F: np.ndarray, mu: float, p: float) -> np.ndarray:
    A_fun = _build_mixed_pk1_tangent_2d()
    A_flat = A_fun(float(F[0,0]), float(F[0,1]), float(F[1,0]), float(F[1,1]), float(mu), float(p))
    return np.asarray(A_flat, dtype=float).reshape(2, 2, 2, 2)


# ---------------------------------------------------------------------------
# [2D] Sympy-based PK2 stress S and material tensor D (TL)
# ---------------------------------------------------------------------------
def _build_neo_hookean_SD_2d_tl():
    global _S_SYMPY_SD_2D_TL
    if _S_SYMPY_SD_2D_TL is not None: return _S_SYMPY_SD_2D_TL
    if sp is None: raise ImportError("sympy required.")
    E11, E22, G12, mu_s, lam_s = sp.symbols("E11 E22 G12 mu lam", real=True)
    E = sp.Matrix([[E11, G12/2], [G12/2, E22]])
    I2 = sp.eye(2); C = 2 * E + I2; J = sp.sqrt(C.det()); I1 = sp.trace(C)
    Psi = mu_s/2 * (I1 - 2 - 2*sp.log(J)) + lam_s/2 * (sp.log(J)**2)
    S_vec = sp.Matrix([sp.diff(Psi, v) for v in [E11, E22, G12]])
    D = sp.zeros(3, 3)
    vars_ = [E11, E22, G12]
    for i in range(3):
        for j in range(3): D[i, j] = sp.diff(S_vec[i], vars_[j])
    func = sp.lambdify((E11, E22, G12, mu_s, lam_s), (S_vec, D), "numpy")
    _S_SYMPY_SD_2D_TL = func
    return func

# [NEW] Plane Stress 전용 (Incompressible Assumption for nu=0.5)
def _build_neo_hookean_SD_2d_plane_stress_incomp():
    if sp is None: raise ImportError("sympy required.")
    
    # Sympy 심볼 정의
    E11, E22, G12, mu_s = sp.symbols("E11 E22 G12 mu", real=True)
    
    # 2D 변형 텐서 및 C 정의
    E_mat = sp.Matrix([[E11, G12/2], [G12/2, E22]])
    I2 = sp.eye(2)
    C_2d = 2 * E_mat + I2
    
    # [핵심] Plane Stress에서는 두께 방향 stretch(lambda_3)가 변함
    # Incompressible 가정 (J = 1) -> lambda_3^2 = 1 / det(C_2d)
    det_C_2d = C_2d.det()
    C33 = 1.0 / det_C_2d  
    
    # 3D Invariant I1 계산 (C11 + C22 + C33)
    I1 = C_2d.trace() + C33
    
    # Strain Energy Density (Incompressible Neo-Hookean)
    # Plane Stress에서는 압력항(Volume penalty)이 필요 없음 (두께가 변하며 부피 유지)
    Psi = mu_s / 2.0 * (I1 - 3)
    
    # PK2 Stress (S) 및 Material Tangent (D) 유도
    vars_ = [E11, E22, G12]
    S_vec = sp.Matrix([sp.diff(Psi, v) for v in vars_])
    D = sp.zeros(3, 3)
    for i in range(3):
        for j in range(3):
            D[i, j] = sp.diff(S_vec[i], vars_[j])
            
    # Python 함수로 변환 (lambda는 사용하지 않음)
    func = sp.lambdify((E11, E22, G12, mu_s), (S_vec, D), "numpy")
    return func

# 글로벌 변수 추가 (파일 상단 쪽에 선언 필요 없으면 함수 내에서 호출됨)
_S_SYMPY_SD_2D_PS = None

def _compute_S_D_voigt_2d_tl(F: np.ndarray, mu: float, lam: float):
    SD_fun = _build_neo_hookean_SD_2d_tl()
    C = F.T @ F; E11 = 0.5*(C[0,0]-1); E22 = 0.5*(C[1,1]-1); G12 = float(C[0,1])
    S_vec, D = SD_fun(E11, E22, G12, float(mu), float(lam))
    S_vec = np.asarray(S_vec, dtype=float).reshape(3)
    D = np.asarray(D, dtype=float).reshape(3, 3)
    S = np.array([[S_vec[0], S_vec[2]], [S_vec[2], S_vec[1]]], dtype=float)
    return S, S_vec, D


# ---------------------------------------------------------------------------
# [2D] Abaqus Neo-Hookean W (3D) → plane strain 2D
# ---------------------------------------------------------------------------
def _build_neo_hookean_SD_2d_abaqus_plane_strain():
    global _S_SYMPY_SD_2D_ABAQUS
    if _S_SYMPY_SD_2D_ABAQUS is not None: return _S_SYMPY_SD_2D_ABAQUS
    if sp is None: raise ImportError("sympy required.")
    E11, E22, G12, mu_s, lam_s = sp.symbols("E11 E22 G12 mu lam", real=True)
    E2 = sp.Matrix([[E11, G12/2], [G12/2, E22]])
    I2 = sp.eye(2); C2 = 2 * E2 + I2
    C = sp.Matrix([[C2[0,0], C2[0,1], 0], [C2[1,0], C2[1,1], 0], [0,0,1]])
    J = sp.sqrt(C.det()); I1 = sp.trace(C)
    K_s = lam_s + 2*mu_s/3; C10_s = mu_s/2
    Psi = C10_s * (J**(-sp.Rational(2,3)) * I1 - 3) + (K_s / 4.0) * (J - 1) ** 2
    S_vec = sp.Matrix([sp.diff(Psi, v) for v in [E11, E22, G12]])
    D = sp.zeros(3, 3)
    vars_ = [E11, E22, G12]
    for i in range(3):
        for j in range(3): D[i, j] = sp.diff(S_vec[i], vars_[j])
    func = sp.lambdify((E11, E22, G12, mu_s, lam_s), (S_vec, D), "numpy")
    _S_SYMPY_SD_2D_ABAQUS = func
    return func

def _compute_S_D_voigt_2d_abaqus_plane_strain(F: np.ndarray, mu: float, lam: float):
    SD_fun = _build_neo_hookean_SD_2d_abaqus_plane_strain()
    C = F.T @ F; E11 = 0.5*(C[0,0]-1); E22 = 0.5*(C[1,1]-1); G12 = float(C[0,1])
    S_vec, D = SD_fun(E11, E22, G12, float(mu), float(lam))
    S_vec = np.asarray(S_vec, dtype=float).reshape(3)
    D = np.asarray(D, dtype=float).reshape(3, 3)
    S = np.array([[S_vec[0], S_vec[2]], [S_vec[2], S_vec[1]]], dtype=float)
    return S, S_vec, D


# ---------------------------------------------------------------------------
# [NEW] 3D Standard Compressible Neo-Hookean
# ---------------------------------------------------------------------------
def _build_neo_hookean_SD_3d_tl():
    global _SYMPY_SD_3D_TL
    if _SYMPY_SD_3D_TL is not None: return _SYMPY_SD_3D_TL
    if sp is None: raise ImportError("sympy required.")
    E11, E22, E33, G12, G23, G13, mu_s, lam_s = sp.symbols("E11 E22 E33 G12 G23 G13 mu lam", real=True)
    E = sp.Matrix([[E11, G12/2, G13/2], [G12/2, E22, G23/2], [G13/2, G23/2, E33]])
    I3 = sp.eye(3); C = 2 * E + I3; J = sp.sqrt(C.det()); I1 = sp.trace(C)
    Psi = mu_s/2 * (I1 - 3 - 2*sp.log(J)) + lam_s/2 * (sp.log(J)**2)
    vars_ = [E11, E22, E33, G12, G23, G13]
    S_vec = sp.Matrix([sp.diff(Psi, v) for v in vars_])
    D = sp.zeros(6, 6)
    for i in range(6):
        for j in range(6): D[i, j] = sp.diff(S_vec[i], vars_[j])
    func = sp.lambdify((E11, E22, E33, G12, G23, G13, mu_s, lam_s), (S_vec, D), "numpy")
    _SYMPY_SD_3D_TL = func
    return func

def _compute_S_D_voigt_3d_tl(F: np.ndarray, mu: float, lam: float):
    SD_fun = _build_neo_hookean_SD_3d_tl()
    C = F.T @ F; E = 0.5*(C - np.eye(3))
    E11, E22, E33 = E[0,0], E[1,1], E[2,2]
    G12, G23, G13 = float(C[0,1]), float(C[1,2]), float(C[0,2])
    S_raw, D = SD_fun(E11,E22,E33,G12,G23,G13, float(mu), float(lam))
    S_vec = np.asarray(S_raw, float).reshape(6)
    D = np.asarray(D, float).reshape(6, 6)
    s11, s22, s33, s12, s23, s13 = S_vec
    S = np.array([[s11, s12, s13], [s12, s22, s23], [s13, s23, s33]], dtype=float)
    return S, S_vec, D

def _build_neo_hookean_SD_3d_abaqus():
    global _SYMPY_SD_3D_ABAQUS
    if _SYMPY_SD_3D_ABAQUS is not None: return _SYMPY_SD_3D_ABAQUS
    if sp is None: raise ImportError("sympy required.")
    E11,E22,E33,G12,G23,G13, mu_s, lam_s = sp.symbols("E11 E22 E33 G12 G23 G13 mu lam", real=True)
    E = sp.Matrix([[E11, G12/2, G13/2], [G12/2, E22, G23/2], [G13/2, G23/2, E33]])
    C = 2*E + sp.eye(3); J = sp.sqrt(C.det()); I1 = sp.trace(C)
    K_s = lam_s + 2*mu_s/3; C10_s = mu_s/2
    Psi = C10_s*(J**(-sp.Rational(2,3))*I1 - 3) + (K_s/4.0)*(J-1)**2
    S_vec = sp.Matrix([sp.diff(Psi, v) for v in [E11,E22,E33,G12,G23,G13]])
    D = sp.zeros(6,6)
    for i in range(6):
        for j in range(6): D[i,j] = sp.diff(S_vec[i], [E11,E22,E33,G12,G23,G13][j])
    SD_fun = sp.lambdify((E11,E22,E33,G12,G23,G13, mu_s, lam_s), (S_vec, D), "numpy")
    _SYMPY_SD_3D_ABAQUS = SD_fun
    return SD_fun

def _compute_S_D_voigt_3d_abaqus(F: np.ndarray, mu: float, lam: float):
    SD_fun = _build_neo_hookean_SD_3d_abaqus()
    C = F.T @ F; E = 0.5*(C - np.eye(3))
    E11, E22, E33 = E[0,0], E[1,1], E[2,2]
    G12, G23, G13 = float(C[0,1]), float(C[1,2]), float(C[0,2])
    S_vec, D = SD_fun(E11,E22,E33,G12,G23,G13, float(mu), float(lam))
    S_vec = np.asarray(S_vec, float).reshape(6)
    D = np.asarray(D, float).reshape(6,6)
    S = np.array([[S_vec[0], S_vec[3], S_vec[5]], [S_vec[3], S_vec[1], S_vec[4]], [S_vec[5], S_vec[4], S_vec[2]]], float)
    return S, S_vec, D


# ---------------------------------------------------------------------------
# Dispatcher : TL vs Abaqus W
# ---------------------------------------------------------------------------
def _compute_S_D_voigt(F, mu, lam, hyper_model="tl", plane_stress=False, dim=2):
    if int(dim) == 2:
        # Plane Stress가 켜져있으면 무조건 Plane Stress 전용 함수 사용
        if plane_stress:
            global _S_SYMPY_SD_2D_PS
            if _S_SYMPY_SD_2D_PS is None:
                _S_SYMPY_SD_2D_PS = _build_neo_hookean_SD_2d_plane_stress_incomp()
            
            # Strain 계산
            C = F.T @ F
            E11 = 0.5 * (C[0,0] - 1)
            E22 = 0.5 * (C[1,1] - 1)
            G12 = float(C[0,1])
            
            # Plane Stress 함수 호출 (lam 불필요)
            S_vec, D = _S_SYMPY_SD_2D_PS(E11, E22, G12, float(mu))
            
            S_vec = np.asarray(S_vec, dtype=float).reshape(3)
            D = np.asarray(D, dtype=float).reshape(3, 3)
            S = np.array([[S_vec[0], S_vec[2]], [S_vec[2], S_vec[1]]], dtype=float)
            return S, S_vec, D
            
        else:
            # Plane Strain (기존 로직 유지)
            model = (hyper_model or "tl").lower()
            if model == "abaqus": return _compute_S_D_voigt_2d_abaqus_plane_strain(F, mu, lam)
            else: return _compute_S_D_voigt_2d_tl(F, mu, lam)

    # 3D (기존 유지)
    model = (hyper_model or "tl").lower()
    if model == "abaqus": return _compute_S_D_voigt_3d_abaqus(F, mu, lam)
    return _compute_S_D_voigt_3d_tl(F, mu, lam)

def _calc_mises(s):
    if len(s) == 4: return float(np.sqrt(0.5*((s[0]-s[1])**2 + (s[1]-s[2])**2 + (s[2]-s[0])**2 + 6*s[3]**2)))
    elif len(s) == 6: return float(np.sqrt(0.5*((s[0]-s[1])**2 + (s[1]-s[2])**2 + (s[2]-s[0])**2 + 6*(s[3]**2 + s[4]**2 + s[5]**2))))
    return 0.0

def _calc_mises_3d(s11: float, s22: float, s33: float, s12: float, s23: float, s13: float) -> float:
    return float(np.sqrt(0.5*((s11-s22)**2 + (s22-s33)**2 + (s33-s11)**2 + 6.0*(s12**2+s23**2+s13**2))))


# ---------------------------------------------------------------------------
# Main Class: NonlinearHyper
# ---------------------------------------------------------------------------
class NonlinearHyper:
    def __init__(
        self,
        NL: np.ndarray,
        conn_rows: np.ndarray,
        element_shape: str,
        element_order: str,
        integration: str,
        thickness: float,
        sf,
        material: Dict[str, float],
        plane_stress: bool,
        hyper_model: str = "tl",
        dim: int = 2,
    ) -> None:
        
        NL_arr = np.asarray(NL, float)
        inferred_dim = int(NL_arr.shape[1])
        if dim not in (2, 3): dim = inferred_dim
        if int(dim) != inferred_dim: dim = inferred_dim
        self.dim = int(dim)
        self._dim_tag = "2D" if self.dim == 2 else "3D"

        self.X = NL_arr[:, :self.dim]
        self.conn = np.asarray(conn_rows, int)
        self.NoN = self.X.shape[0]
        self.NoE, self.NPE = self.conn.shape

        self.element_shape = element_shape
        self.element_order = element_order
        self.integration = integration
        self.t = float(thickness)

        self.mu = float(material["mu"])
        self.lam = float(material["lam"])
        self.plane_stress = bool(plane_stress) if self.dim == 2 else False
        self.hyper_model = str(hyper_model or "tl").lower()
        
        # [FIX 1] Auto-Detect and Fix Inverted Elements (Negative Jacobian)
        self._check_and_fix_elements()

        self.K_bulk = self.lam + 2.0 * self.mu / 3.0
        self.bulkmodInv = 1.0 / self.K_bulk if self.K_bulk > 0.0 else 0.0

        self.is_incompressible = (not self.plane_stress) and (self.lam > 20.0 * self.mu)

        self.is_p2p1_element = (self.dim==2 and self.is_incompressible and element_shape=="Tri" and element_order=="Quadratic")
        self.is_taylor_hood = self.is_p2p1_element

        if self.is_incompressible: print(f"[INFO] {self.dim}D Incompressible Mixed Active.")
        else: print(f"[INFO] {self.dim}D Compressible Active.")

        self._gpN_u, self._gpdN_u, self.W = self._extract_gp_from_sf(sf, self.NPE)
        self._J0_list = []; self._invJT_list = []; self._precompute_reference()

        self.p_elem = None; self.p_nodal = None; self.Np = 0
        if self.is_incompressible:
            if self.is_taylor_hood:
                ncorner = 3
                corner_nodes = self.conn[:, :ncorner].flatten()
                self.p_nodes = np.unique(corner_nodes)
                self.Np = len(self.p_nodes)
                self.p_dof_map = {nid: i for i, nid in enumerate(self.p_nodes)}
                self._gpN_p, _, _ = self._extract_gp_from_sf(sf, ncorner)
            else:
                self.Np = self.NoE; self.p_elem = np.zeros(self.NoE, float)

    def _check_and_fix_elements(self):
        """
        Calculate initial volume/area. If negative, swap nodes to fix winding order.
        """
        fixed_count = 0
        for i, c in enumerate(self.conn):
            Xe = self.X[c]
            detJ = 0.0
            if self.dim == 2:
                v1 = Xe[1] - Xe[0]; v2 = Xe[2] - Xe[0]
                detJ = np.cross(v1, v2)
                if detJ < 0: 
                    self.conn[i, [1, 2]] = self.conn[i, [2, 1]]
                    fixed_count += 1
            elif self.dim == 3:
                if self.NPE == 4: # Tet
                    J = np.stack([Xe[1]-Xe[0], Xe[2]-Xe[0], Xe[3]-Xe[0]], axis=1)
                    detJ = np.linalg.det(J)
                    if detJ < 0: 
                        self.conn[i, [0, 1]] = self.conn[i, [1, 0]]
                        fixed_count += 1
                elif self.NPE == 8: # Hex (Approx)
                    J = np.stack([Xe[1]-Xe[0], Xe[3]-Xe[0], Xe[4]-Xe[0]], axis=1)
                    detJ = np.linalg.det(J)
                    if detJ < 0:
                        self.conn[i, [1, 3]] = self.conn[i, [3, 1]]
                        self.conn[i, [5, 7]] = self.conn[i, [7, 5]]
                        fixed_count += 1
        if fixed_count > 0:
            print(f"[INFO] Auto-corrected {fixed_count} inverted elements (Negative Volume Fix).")

    def _extract_gp_from_sf(self, sf, npe_target):
        if not hasattr(sf, "gp"): raise AttributeError("SF object missing 'gp'")
        gp = np.array(sf.gp, float); W = np.array(sf.W, float)
        gpN, gpdN = [], []
        for g in gp:
            N = np.array(sf.shape(g, npe_target, self._dim_tag), float).reshape(-1)
            dN = np.array(sf.gradshape(g, npe_target, self._dim_tag), float)
            gpN.append(N); gpdN.append(dN)
        if W.size == 1: W = np.repeat(W, len(gpN))
        return np.asarray(gpN), np.asarray(gpdN), W

    def _precompute_reference(self):
        for e, conn in enumerate(self.conn):
            X_e = self.X[conn]
            J0_e, invJT_e = [], []
            for igp in range(len(self.W)):
                dNdXi = self._gpdN_u[igp]
                J0 = X_e.T @ dNdXi.T
                J0_e.append(J0); invJT_e.append(np.linalg.inv(J0).T)
            self._J0_list.append(J0_e); self._invJT_list.append(invJT_e)

    def _compute_BL(self, F, dNdX):
        NPE = self.NPE
        if self.dim == 2:
            BL = np.zeros((3, 2*NPE), float)
            F11,F12,F21,F22 = F[0,0],F[0,1],F[1,0],F[1,1]
            for a in range(NPE):
                dx, dy = dNdX[0, a], dNdX[1, a]
                BL[0,2*a:2*a+2] = [F11*dx, F21*dx]
                BL[1,2*a:2*a+2] = [F12*dy, F22*dy]
                BL[2,2*a:2*a+2] = [F12*dx+F11*dy, F22*dx+F21*dy]
            return BL
        else:
            BL = np.zeros((6, 3*NPE), float)
            for a in range(NPE):
                dx, dy, dz = dNdX[0,a], dNdX[1,a], dNdX[2,a]
                BL[0, 3*a:3*a+3] = F[:,0]*dx
                BL[1, 3*a:3*a+3] = F[:,1]*dy
                BL[2, 3*a:3*a+3] = F[:,2]*dz
                BL[3, 3*a:3*a+3] = F[:,0]*dy + F[:,1]*dx
                BL[4, 3*a:3*a+3] = F[:,1]*dz + F[:,2]*dy
                BL[5, 3*a:3*a+3] = F[:,0]*dz + F[:,2]*dx
            return BL

    def _elem_residual_compressible(self, e, ue):
        ndof = self.NPE * self.dim; re = np.zeros(ndof, float)
        for igp in range(len(self.W)):
            dNdXi = self._gpdN_u[igp]; dNdX = self._invJT_list[e][igp] @ dNdXi
            F = np.eye(self.dim) + ue.T @ dNdX.T
            S, S_vec, _ = _compute_S_D_voigt(F, self.mu, self.lam, self.hyper_model, self.plane_stress, self.dim)
            BL = self._compute_BL(F, dNdX)
            w = self.W[igp] * float(np.linalg.det(self._J0_list[e][igp]))
            if self.dim == 2: w *= self.t
            re += BL.T @ S_vec * w
        return re.reshape(-1)

    def _elem_residual_mixed(self, e, ue, pe):
        mu, Kinv = self.mu, self.bulkmodInv
        dim = self.dim  # 2 or 3
        
        # Pressure Node 개수 설정 (3D Taylor-Hood Tet10 등은 별도 로직 필요, 여기선 Q1P0 가정)
        # 2D P2P1(Taylor-Hood)인지 3D인지에 따라 분기
        is_th = (dim == 2 and self.is_p2p1_element) 
        NPE_p = 3 if is_th else 1 
        
        re_p = np.zeros(NPE_p, float)
        re_u = np.zeros((self.NPE, dim), float)
        
        for igp in range(len(self.W)):
            dNdXi = self._gpdN_u[igp]
            dNdX = self._invJT_list[e][igp] @ dNdXi
            
            # F = I + grad(u)
            F = np.eye(dim) + ue.T @ dNdX.T
            J = float(np.linalg.det(F))
            FinvT = np.linalg.inv(F).T
            
            if is_th: p_at_gp = float(self._gpN_p[igp] @ pe)
            else: p_at_gp = float(pe[0])
            
            # Generalized Mixed PK1 Stress (P)
            Jm23 = J**(-2/3)
            trC = float(np.trace(F.T @ F))
            P_iso = mu * (Jm23 * F - (Jm23 * trC / 3.0) * FinvT)
            P = P_iso + J * p_at_gp * FinvT
            
            w = self.W[igp] * float(np.linalg.det(self._J0_list[e][igp]))
            if dim == 2: w *= self.t
            
            # Residual R_u = Integral(P : gradN)
            for a in range(self.NPE):
                gradNa = dNdX[:, a]
                # P @ gradNa (Vector)
                term = P @ gradNa
                re_u[a, :] += term * w
            
            # Residual R_p (Continuity)
            Np = self._gpN_p[igp] if is_th else np.array([1.])
            re_p += (J - 1.0 - p_at_gp * Kinv) * Np * w
            
        return re_u.reshape(-1), re_p

    def _elem_tangent_compressible_analytic(self, e, ue):
        ndof = self.NPE * self.dim; Ke = np.zeros((ndof, ndof), float)
        for igp in range(len(self.W)):
            dNdXi = self._gpdN_u[igp]; dNdX = self._invJT_list[e][igp] @ dNdXi
            F = np.eye(self.dim) + ue.T @ dNdX.T
            S, S_vec, D_voigt = _compute_S_D_voigt(F, self.mu, self.lam, self.hyper_model, self.plane_stress, self.dim)
            BL = self._compute_BL(F, dNdX)
            w = self.W[igp] * float(np.linalg.det(self._J0_list[e][igp]))
            if self.dim == 2: w *= self.t
            
            Ke += BL.T @ D_voigt @ BL * w
            
            for a in range(self.NPE):
                gradNa = dNdX[:, a]
                for b in range(self.NPE):
                    gradNb = dNdX[:, b]
                    g_ab = float(gradNa.T @ S @ gradNb) * w
                    for d in range(self.dim): Ke[self.dim*a+d, self.dim*b+d] += g_ab
        return Ke

    def _elem_tangent_mixed_analytic(self, e, ue, pe):
        mu, Kinv = self.mu, self.bulkmodInv
        dim = self.dim  # 2 or 3
        nu = dim * self.NPE
        
        is_th = (dim == 2 and self.is_p2p1_element)
        np_ = 3 if is_th else 1
        
        K_uu = np.zeros((nu, nu))
        K_up = np.zeros((nu, np_))
        K_pp = np.zeros((np_, np_))
        
        # Geometric Stiffness는 K_uu에 더해집니다.
        
        for igp in range(len(self.W)):
            dNdX = self._invJT_list[e][igp] @ self._gpdN_u[igp]
            F = np.eye(dim) + ue.T @ dNdX.T
            J = float(np.linalg.det(F))
            FinvT = np.linalg.inv(F).T
            
            if is_th: p_at_gp = float(self._gpN_p[igp] @ pe)
            else: p_at_gp = float(pe[0])
            
            # 1. Material Tangent (A4 Tensor) Calculation
            if dim == 2:
                A4 = _compute_PK1_tangent_mixed_2d(F, mu, p_at_gp)
            else:
                A4 = _compute_PK1_tangent_mixed_3d(F, mu, p_at_gp)
            
            # 2. Compute PK2 Stress (S) for Geometric Stiffness
            # P is needed to get S = F^-1 * P
            Jm23 = J**(-2/3)
            P_iso = mu * (Jm23 * F - (Jm23 * np.trace(F.T @ F) / 3.0) * FinvT)
            P = P_iso + J * p_at_gp * FinvT
            S = np.linalg.inv(F) @ P
            
            w = self.W[igp] * float(np.linalg.det(self._J0_list[e][igp]))
            if dim == 2: w *= self.t
            
            # Assembly Loop
            for a in range(self.NPE):
                gNa = dNdX[:, a]  # Gradient of Shape function a
                for b in range(self.NPE):
                    gNb = dNdX[:, b] # Gradient of Shape function b
                    
                    # [K_uu part 1] Material Stiffness: gradNa : A : gradNb
                    # Index mapping: A_ijkl * dN_a/dX_j * dN_b/dX_l
                    # Output index in matrix: (dim*a + i, dim*b + k)
                    
                    for i in range(dim):
                        for k in range(dim):
                            val = 0.0
                            for j in range(dim):
                                for l in range(dim):
                                    val += A4[i, j, k, l] * gNa[j] * gNb[l]
                            K_uu[dim*a+i, dim*b+k] += val * w
                    
                    # [K_uu part 2] Geometric Stiffness: (gradNa . S . gradNb) * I
                    # Scalar value g_ab applied to diagonal blocks
                    g_ab = float(gNa.T @ S @ gNb) * w
                    for i in range(dim):
                        K_uu[dim*a+i, dim*b+i] += g_ab
            
            # [K_up] Mixed Coupling Terms
            Np = self._gpN_p[igp] if is_th else np.array([1.])
            for a in range(self.NPE):
                gNa = dNdX[:, a]
                for i in range(dim):
                    # Derivative of constraint equation J-1 approx => J * F^-T : gradNa
                    gi = float(J * (FinvT[i, :] @ gNa))
                    for m in range(np_):
                        val = gi * Np[m] * w
                        K_up[dim*a+i, m] += val
            
            # [K_pp] Penalty / Stabilization
            for m in range(np_):
                for n in range(np_):
                    K_pp[m, n] += (-Kinv) * Np[m] * Np[n] * w
        
        # Block Assembly
        nt = nu + np_
        Ke = np.zeros((nt, nt))
        Ke[:nu, :nu] = K_uu
        Ke[:nu, nu:] = K_up
        Ke[nu:, :nu] = K_up.T
        Ke[nu:, nu:] = K_pp
        
        return Ke

    def _assemble_external_u(self, b0, traction_edges, conc_forces):
        ndof = self.dim * self.NoN
        R_ext = np.zeros(ndof, float)
        b_vec = np.array(b0, float).reshape(-1)
        if b_vec.size < self.dim: b_vec = np.concatenate([b_vec, np.zeros(self.dim-b_vec.size)])
        
        if np.linalg.norm(b_vec) > 0:
            for e, conn in enumerate(self.conn):
                for igp, w_raw in enumerate(self.W):
                    w = w_raw * float(np.linalg.det(self._J0_list[e][igp]))
                    if self.dim==2: w*=self.t
                    for a, A in enumerate(conn):
                        for k in range(self.dim): R_ext[self.dim*A+k] += self._gpN_u[igp][a]*b_vec[k]*w
        
        if self.dim == 2:
            for item in traction_edges or []:
                T = np.array(item.get("T", item.get("traction", [0,0])), float)
                if "nodes3" in item:
                    i,k,j = item["nodes3"]; idxs=[i-1,k-1,j-1]
                    L = np.linalg.norm(self.X[j-1]-self.X[i-1])
                    if L==0: continue
                    for ix, wt in zip(idxs, [1/6, 4/6, 1/6]): R_ext[2*ix:2*ix+2] += T*L*wt*self.t
                elif "nodes" in item:
                    i,j=item["nodes"]; idxs=[i-1,j-1]
                    L = np.linalg.norm(self.X[j-1]-self.X[i-1])
                    if L==0: continue
                    for ix in idxs: R_ext[2*ix:2*ix+2] += T*L*0.5*self.t

        elif self.dim == 3:
            # [FIX 2] Area-based Linear Logic (Ported from Assembly...py)
            for item in traction_edges or []:
                nlist = item.get("seed_nodes", item.get("nodes", []))
                if not nlist: continue
                node_idxs = [int(n)-1 for n in nlist]
                X_face = self.X[node_idxs, :]
                
                area, face_ctr, n_hat = _area_of_nodes(X_face)
                if area <= 0: continue
                
                # Outward check
                e_idx = item.get("element")
                if e_idx is not None:
                    elem_ctr = self.X[self.conn[e_idx],:].mean(axis=0)
                    if np.dot(n_hat, face_ctr - elem_ctr) < 0: n_hat = -n_hat
                
                load_type = item.get("type", None)
                val = item.get("value", None)

                if load_type == "pressure" and val is not None:
                    tvec = -float(val) * n_hat
                elif load_type == "traction" and val is not None:
                    tvec = np.array(val, dtype=float)
                # Fallback
                elif "pressure" in item and item["pressure"] is not None:
                    tvec = -float(item["pressure"]) * n_hat
                elif "traction" in item and item["traction"] is not None:
                    tvec = np.array(item["traction"], dtype=float)
                else:
                    continue
                
                f_node = (area / len(node_idxs)) * tvec
                for r in node_idxs:
                    R_ext[3*r : 3*r+3] += f_node

        for c in (conc_forces or []):
            F = np.array(c.get("F", [0]*self.dim), float)
            for n in c.get("nodes", []):
                idx = int(n)-1
                if 0<=idx<self.NoN:
                    for k in range(self.dim): R_ext[self.dim*idx+k] += F[k]
        return R_ext

    # [FIX: Inserted _assemble_R_compressible inside class with correct indentation]
    def _assemble_R_compressible(self, u, b0, tr, conc=[]):
        R_int = np.zeros(self.dim*self.NoN)
        for e, conn in enumerate(self.conn):
            re = self._elem_residual_compressible(e, u.reshape(-1, self.dim)[conn])
            dofs = [self.dim*n+k for n in conn for k in range(self.dim)]
            R_int[dofs] += re
        return R_int - self._assemble_external_u(b0, tr, conc)

    def _assemble_R_mixed(self, u, p, b0, tr, conc=[]):
        # [FIX] 차원을 self.dim으로 일반화 (기존 2로 하드코딩 되어 있었음)
        ndofu = self.dim * self.NoN
        R = np.zeros(ndofu + self.Np)
        
        for e, conn in enumerate(self.conn):
            # [FIX] Reshape to (Nodes, dim)
            ue = u.reshape(-1, self.dim)[conn]
            
            if self.is_taylor_hood:
                # Taylor-Hood (2D Quadratic Tri) only
                pdofs = [self.p_dof_map[n] for n in conn[:(3 if self.is_th_tri else 4)]]
                pe = p[pdofs]
                gdofs = [ndofu + d for d in pdofs]
            else:
                # Constant Pressure (Q1P0 for 2D/3D)
                pe = np.array([p[e]])
                gdofs = [ndofu + e]
            
            ru, rp = self._elem_residual_mixed(e, ue, pe)
            
            # [FIX] DOF Indexing using self.dim
            udofs = [self.dim * n + k for n in conn for k in range(self.dim)]
            
            R[udofs] += ru
            R[gdofs] += rp
            
        R[:ndofu] -= self._assemble_external_u(b0, tr, conc)
        return R

    def _assemble_K_mixed(self, u, p):
        # [FIX] 차원을 self.dim으로 일반화
        ndofu = self.dim * self.NoN
        nt = ndofu + self.Np
        K = np.zeros((nt, nt))
        
        for e, conn in enumerate(self.conn):
            if self.is_taylor_hood:
                pdofs = [self.p_dof_map[n] for n in conn[:(3 if self.is_th_tri else 4)]]
                pe = p[pdofs]
                gdofs = [ndofu + d for d in pdofs]
            else:
                pe = np.array([p[e]])
                gdofs = [ndofu + e]
                
            # [FIX] Reshape to (Nodes, dim)
            ue = u.reshape(-1, self.dim)[conn]
            
            Ke = self._elem_tangent_mixed_analytic(e, ue, pe)
            
            # [FIX] DOF Indexing using self.dim
            udofs = [self.dim * n + k for n in conn for k in range(self.dim)]
            
            all_dofs = udofs + gdofs
            
            # Matrix Assembly
            # Ke size: (dim*NPE + Np) x (dim*NPE + Np)
            for i, I in enumerate(all_dofs):
                for j, J in enumerate(all_dofs):
                    K[I, J] += Ke[i, j]
                    
        return K

    def _assemble_K_compressible(self, u, b0, tr):
        ndof = self.dim * self.NoN; K = np.zeros((ndof, ndof))
        for e, conn in enumerate(self.conn):
            Ke = self._elem_tangent_compressible_analytic(e, u.reshape(-1, self.dim)[conn])
            dofs = [self.dim*n+k for n in conn for k in range(self.dim)]
            for i, I in enumerate(dofs):
                for j, J in enumerate(dofs): K[I,J] += Ke[i,j]
        return K

    def _assemble_K_mixed(self, u, p):
        # [FIX] 차원을 self.dim으로 일반화 (기존 2로 하드코딩 되어 있었음)
        ndofu = self.dim * self.NoN
        nt = ndofu + self.Np
        K = np.zeros((nt, nt))
        
        for e, conn in enumerate(self.conn):
            # [Pressure DOF 처리]
            if self.is_taylor_hood:
                pdofs = [self.p_dof_map[n] for n in conn[:(3 if self.is_th_tri else 4)]]
                pe = p[pdofs]
                gdofs = [ndofu + d for d in pdofs]
            else:
                pe = np.array([p[e]])
                gdofs = [ndofu + e]
            
            # [FIX] u 벡터를 (Nodes, dim) 형태로 변환
            # 3D의 경우 self.dim=3이므로 (-1, 3)으로 정상 변환됨
            ue = u.reshape(-1, self.dim)[conn]
            
            # 요소 단위 Tangent Stiffness 계산
            Ke = self._elem_tangent_mixed_analytic(e, ue, pe)
            
            # [FIX] Global Matrix Assembly 인덱싱 수정 (2 -> self.dim)
            udofs = [self.dim * n + k for n in conn for k in range(self.dim)]
            
            all_dofs = udofs + gdofs
            
            # Matrix Assembly
            for i, I in enumerate(all_dofs):
                for j, J in enumerate(all_dofs):
                    K[I, J] += Ke[i, j]
                    
        return K

    def solve(self, dirichlet, nsteps, atol, rtol, stol, max_iter, b0, traction_edges, conc_forces=[], verbose=True, **kwargs):
        ndofu = self.dim * self.NoN
        
        # [수정 1] 고정할 자유도(DOF)와 목표 값(Target Value)을 매핑하여 저장
        # fixed_dofs: { global_dof_index : max_displacement_value }
        fixed_dofs = {}
        
        for nid, tup in dirichlet.items():
            idx = int(nid) - 1
            vals = list(tup) if isinstance(tup, (list, tuple)) else [tup]
            while len(vals) < self.dim: vals.append(None)
            
            for k, val in enumerate(vals[:self.dim]):
                if val is not None:
                    dof = self.dim * idx + k
                    fixed_dofs[dof] = float(val)

        # 행렬 연산을 위해 인덱스 리스트 생성 (기존 fixed와 동일 역할)
        fixed_indices = np.array(list(fixed_dofs.keys()), dtype=int) if fixed_dofs else np.zeros(0, dtype=int)

        w = np.zeros(ndofu + (self.Np if self.is_incompressible else 0))
        lams = np.linspace(1/nsteps, 1.0, nsteps)

        for istep, lam in enumerate(lams, 1):
            if verbose: print(f"[STEP {istep}] Load Factor {lam:.3f}")
            
            # [수정 2] 현재 Step의 하중 계수(lam)에 맞춰 강제 변위 값을 w 벡터에 미리 주입
            # Newton-Raphson을 시작하기 전에 변위를 목표치로 이동시킴
            if fixed_dofs:
                for dof, val in fixed_dofs.items():
                    w[dof] = val * lam

            # Traction 및 Body Force 스케일링 (기존 코드 유지)
            tr_s = []
            for t in traction_edges or []:
                it = t.copy()
                if "value" in it: 
                    v = it["value"]
                    it["value"] = [lam*x for x in v] if isinstance(v,(list,tuple,np.ndarray)) else lam*v
                elif "T" in it: 
                    it["T"] = [lam*x for x in it["T"]]
                if "pressure" in it and isinstance(it["pressure"], (int, float)):
                    it["pressure"] = lam * float(it["pressure"])
                if "traction" in it:
                    v = it["traction"]
                    it["traction"] = [lam*x for x in v] if isinstance(v,(list,tuple,np.ndarray)) else lam*v
                tr_s.append(it)
            
            conc_s = []
            for c in conc_forces or []:
                ic = c.copy(); F = ic.get("F", ic.get("force"))
                if F is not None: ic["F"] = [lam*float(x) for x in F]
                conc_s.append(ic)
            
            b_s = tuple(np.array(b0)*lam)

            # Newton-Raphson Iteration
            for it in range(1, max_iter+1):
                if self.is_incompressible:
                    u=w[:ndofu]; p=w[ndofu:]
                    R = self._assemble_R_mixed(u, p, b_s, tr_s, conc_s)
                    K = self._assemble_K_mixed(u, p)
                else:
                    u=w
                    R = self._assemble_R_compressible(u, b_s, tr_s, conc_s)
                    K = self._assemble_K_compressible(u, b_s, tr_s)
                
                # [수정 3] 강제 변위 처리를 위한 행렬 수정 (Penalty Method or 1 on diagonal)
                # 이미 위에서 w에 값을 넣었으므로, 보정량 dw는 0이 되어야 함.
                # 따라서 R[fixed] = 0, K[fixed, fixed] = 1 로 설정하면 dw = 0이 됨.
                if len(fixed_indices) > 0:
                    R[fixed_indices] = 0.0
                    K[fixed_indices, :] = 0.0
                    K[:, fixed_indices] = 0.0
                    K[fixed_indices, fixed_indices] = 1.0

                nrm = np.linalg.norm(R)
                if verbose: print(f"  Iter {it}: R={nrm:.3e}")
                
                if nrm < atol: 
                    break
                
                try: 
                    dw = np.linalg.solve(K, -R)
                except np.linalg.LinAlgError: 
                    print("Singular Matrix!"); break
                
                w += dw
        
        if self.is_incompressible:
            if self.is_taylor_hood: self.p_nodal = w[ndofu:]
            else: self.p_elem = w[ndofu:]
            return w[:ndofu]
        return w

    def compute_postprocessing_stresses(self, u):
        NoE, NoN, NPE = self.NoE, self.NoN, self.NPE
        n_comp = 4 if self.dim == 2 else 6
        S_centroid = np.zeros((NoE, n_comp + 1))
        S_node_accum = np.zeros((NoN, n_comp))
        node_count = np.zeros(NoN, int)
        mu, lam = self.mu, self.lam

        for e, conn in enumerate(self.conn):
            ue = u.reshape(-1, self.dim)[conn]
            p_val = 0.0; pe_nodes = None
            if self.is_incompressible:
                if self.p_nodal is not None:
                    pdofs = [self.p_dof_map[n] for n in conn[:self._gpN_p.shape[1]]]
                    pe_nodes = self.p_nodal[pdofs]
                else: p_val = float(self.p_elem[e])

            sig_sum = np.zeros(n_comp); vol_sum = 0.0
            for igp in range(len(self.W)):
                dNdXi = self._gpdN_u[igp]; dNdX = self._invJT_list[e][igp] @ dNdXi
                F = np.eye(self.dim) + ue.T @ dNdX.T
                J = float(np.linalg.det(F)) # 이것은 2D Area Ratio (J_2d)
                FinvT = np.linalg.inv(F).T
                
                # [FIX: Plane Stress 분기 처리 추가]
                if self.plane_stress:
                    # Incompressible Neo-Hookean Plane Stress
                    # J_3d = J_2d * lambda_3 = 1  => lambda_3 = 1 / J_2d
                    # sigma = mu * (b - I) + p*I
                    # sigma_33 = 0 => mu*(lambda_3^2 - 1) + p = 0 => p = -mu*(lambda_3^2 - 1)
                    # sigma_11 = mu*(lambda_1^2 - 1) - mu*(lambda_3^2 - 1) = mu*(lambda_1^2 - lambda_3^2)
                    # Tensor form: sigma_2d = mu * ( b_2d - (lambda_3^2)*I )
                    
                    lambda_3_sq = (1.0 / J)**2
                    b = F @ F.T
                    sigma = mu * (b - lambda_3_sq * np.eye(2))
                    
                    # Plane Stress에서 J는 면적비이지만, 부피적분 가중치로는 t*Area를 사용하므로
                    # Cauchy Stress는 위 식 그대로 사용 (Physics Definition)
                    
                elif self.is_incompressible:
                    # Plane Strain / 3D Incompressible
                    p_gp = float(self._gpN_p[igp]@pe_nodes) if pe_nodes is not None else p_val
                    C = F.T@F; trC = float(np.trace(C)); Jm23 = J**(-2/3)
                    P_iso = mu*(Jm23*F - (Jm23*trC/3)*FinvT); P = P_iso + J*p_gp*FinvT
                    sigma = (1.0/abs(J)) * (P @ F.T)
                    
                else:
                    # Compressible Plane Strain / 3D
                    P = mu*(F - FinvT) + lam*np.log(J)*FinvT
                    sigma = (1.0/abs(J)) * (P @ F.T)
                
                if self.dim == 2:
                    # Plane Stress일 때 s33은 이론적으로 0
                    if self.plane_stress:
                        s33 = 0.0
                    elif self.is_incompressible:
                        s33 = p_gp
                    else:
                        s33 = (lam*np.log(J)/J)
                    
                    sv = np.array([sigma[0,0], sigma[1,1], s33, 0.5*(sigma[0,1]+sigma[1,0])])
                else:
                    sv = np.array([sigma[0,0], sigma[1,1], sigma[2,2], sigma[0,1], sigma[0,2], sigma[1,2]])
                
                w = self.W[igp] * abs(float(np.linalg.det(self._J0_list[e][igp])))
                if self.dim == 2: w *= self.t
                sig_sum += sv * w; vol_sum += w
            
            S_el = sig_sum / vol_sum if vol_sum>0 else np.zeros(n_comp)
            S_centroid[e, :n_comp] = S_el
            S_centroid[e, n_comp] = _calc_mises(S_el)
            for nid in conn:
                S_node_accum[nid] += S_el
                node_count[nid] += 1
        
        S_nodes = np.zeros((NoN, n_comp + 1))
        for i in range(NoN):
            if node_count[i] > 0:
                avg = S_node_accum[i] / node_count[i]
                S_nodes[i, :n_comp] = avg
                S_nodes[i, n_comp] = _calc_mises(avg)
        return S_centroid, S_nodes