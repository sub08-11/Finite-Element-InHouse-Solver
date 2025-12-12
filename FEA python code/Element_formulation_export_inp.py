# -*- coding: utf-8 -*-
# made by changsub
# Element formulation utilities for the in-house FEM solver.
# Provides Gauss quadrature rules, shape functions, and B-matrix
# evaluation for 2D (Quad/Tri) and 3D (Hex/Tet) elements, as well as
# element stiffness assembly and consistent surface traction/pressure
# integration compatible with Abaqus-style element definitions.

import numpy as np
import sympy as sp
import math
from Preprocessing_export_inp import Material_Property  # external dependency

# === Utilities to coerce element settings from etype / mesh ===
def _resolve_from_etype(etype: str):
    if not etype:
        return None, None, None
    e = etype.upper().strip()

    # 3D heat (Abaqus DC3D*)
    if e == "DC3D8":                  return "3D", "Hex", "Linear"
    if e in ("DC3D20", "DC3D20R"):    return "3D", "Hex", "Quadratic"
    if e == "DC3D4":                  return "3D", "Tet", "Linear"
    if e in ("DC3D10", "DC3D10M"):    return "3D", "Tet", "Quadratic"

    # 3D elasticity
    if e == "C3D8":                   return "3D", "Hex", "Linear"
    if e in ("C3D20", "C3D20R"):      return "3D", "Hex", "Quadratic"
    if e == "C3D4":                   return "3D", "Tet", "Linear"
    if e in ("C3D10", "C3D10M"):      return "3D", "Tet", "Quadratic"

    # 2D heat/elasticity
    if e in ("DC2D4", "CPS4", "CPE4"):     return "2D", "Quad", "Linear"
    if e in ("DC2D8", "CPS8", "CPE8"):     return "2D", "Quad", "Quadratic"
    if e in ("DC2D3", "CPS3", "CPE3"):     return "2D", "Tri",  "Linear"
    if e in ("DC2D6", "CPS6", "CPE6"):     return "2D", "Tri",  "Quadratic"
    return None, None, None

def _infer_from_mesh(mesh):
    dim = "3D" if mesh.NL.shape[1] == 3 else "2D"
    npe = mesh.NPE
    if dim == "3D":
        if npe == 8:   return "3D","Hex","Linear"
        if npe == 20:  return "3D","Hex","Quadratic"
        if npe == 4:   return "3D","Tet","Linear"
        if npe == 10:  return "3D","Tet","Quadratic"
    else:
        if npe == 4:   return "2D","Quad","Linear"
        if npe == 8:   return "2D","Quad","Quadratic"
        if npe == 3:   return "2D","Tri","Linear"
        if npe == 6:   return "2D","Tri","Quadratic"
    return None, None, None

def coerce_element_settings(mesh, Dimension, element_shape, element_order):
    """Use INP etype and mesh metadata to coerce (Dimension, shape, order)."""
    etype = getattr(mesh, "etype", None) or getattr(mesh, "etype_from_inp", None)
    d2, s2, o2 = _resolve_from_etype(etype) if etype else (None, None, None)

    Dimension     = d2   or Dimension
    element_shape = s2   or element_shape
    element_order = o2   or element_order

    if (Dimension is None) or (element_shape is None) or (element_order is None):
        d3, s3, o3 = _infer_from_mesh(mesh)
        Dimension     = Dimension     or d3
        element_shape = element_shape or s3
        element_order = element_order or o3

    assert Dimension in ("2D","3D"), f"Dimension must be 2D/3D, got {Dimension}"
    assert element_shape in ("Quad","Tri","Hex","Tet"), f"element_shape invalid: {element_shape}"
    assert element_order in ("Linear","Quadratic"), f"element_order invalid: {element_order}"
    return Dimension, element_shape, element_order

# ------------------------------
# Gauss points & Shape functions
# ------------------------------
class Gauss_Point:
    def __init__(self, analytical_conditions, element_order, element_shape, integration):
        self.analytical_conditions = analytical_conditions
        self.element_order = element_order
        self.element_shape = element_shape
        self.integration   = integration
        self.key = (self.element_shape, self.element_order, self.integration)

        # 2D elements (Quad/Tri)
        self.element_type_2D = {
            ("Quad", "Linear",     "full")   : self.CPS4_CPE4_DC2D4,
            ("Quad", "Linear",     "reduced"): self.CPS4R_CPE4R,
            ("Quad", "Quadratic",  "full")   : self.CPS8_CPE8_DC2D8,
            ("Quad", "Quadratic",  "reduced"): self.CPS8R_CPE8R,
            ("Tri",  "Linear",     "full")   : self.CPS3_CPE3_DC2D3,
            ("Tri",  "Quadratic",  "full")   : self.CPS6_CPE6_DC2D6,
            ("Tri",  "Quadratic",  "reduced"): self.CPS6R_CPE6R,
        }

        # 3D elements (Hex/Tet)
        self.element_type_3D = {
            ("Hex", "Linear",     "full")   : self.C3D8_DC3D8,
            ("Hex", "Linear",     "reduced"): self.C3D8R_DC3D8R,
            ("Hex", "Quadratic",  "full")   : self.C3D20_DC3D20,
            ("Hex", "Quadratic",  "reduced"): self.C3D20R,
            ("Tet", "Linear",     "full")   : self.C3D4_DC3D4,
            ("Tet", "Quadratic",  "full")   : self.C3D10_DC3D10,
            ("Tet", "Quadratic",  "reduced"): self.C3D10M,
        }

        cond = self.analytical_conditions
        if isinstance(cond, str) and cond.strip().lower() == "none":
            cond = None  # normalize "None" string to None

        if self.element_shape in ("Quad", "Tri"):
            if cond in (None, "plane stress", "plane strain"):
                self.gp, self.W, self.NPE = self.calculate_2D_gauss_points_weight()
            else:
                raise ValueError(
                    f"Invalid analytical_conditions for 2D element: {self.analytical_conditions} "
                    "(use 'plane stress', 'plane strain', or None for heat/diffusion)"
                )
        elif self.element_shape in ("Hex", "Tet"):
            self.gp, self.W, self.NPE = self.calculate_3D_gauss_points_weight()
        else:
            raise ValueError(f"Invalid element_shape: {self.element_shape}")

    def calculate_2D_gauss_points_weight(self):
        return self.element_type_2D[self.key]()

    def calculate_3D_gauss_points_weight(self):
        return self.element_type_3D[self.key]()

    # ---- 3D GP ----
    def C3D8_DC3D8(self):
        a = 1 / math.sqrt(3)
        gp = [
            [-a, -a, -a], [ a, -a, -a],
            [-a,  a, -a], [ a,  a, -a],
            [-a, -a,  a], [ a, -a,  a],
            [-a,  a,  a], [ a,  a,  a],
        ]
        W = [1]*8
        return gp, W, 8

    def C3D8R_DC3D8R(self):
        return [[0,0,0]],[8],8

    def C3D20_DC3D20(self):
        a = math.sqrt(3.0/5.0)
        pts_1d = [-a, 0.0, +a]
        w_1d   = [5/9, 8/9, 5/9]
        gp, W = [], []
        for z in range(3):
            for e in range(3):
                for x in range(3):
                    gp.append([pts_1d[x], pts_1d[e], pts_1d[z]])
                    W.append(w_1d[x]*w_1d[e]*w_1d[z])
        return np.array(gp), np.array(W), 20

    def C3D20R(self):
        gp = [[xi/math.sqrt(3), eta/math.sqrt(3), z/math.sqrt(3)]
              for (xi,eta,z) in [(-1,-1, 1),( 1,-1, 1),(-1,-1,-1),( 1,-1, 1),
                                 (-1, 1, 1),( 1, 1, 1),(-1, 1,-1),( 1, 1,-1)]]
        W = [1]*8
        return gp, W, 20

    def C3D4_DC3D4(self):
        # Volume of ref tet is 1/6. detJ gives 6*Vol. So weight needs to be 1/6.
        return np.array([[0.25,0.25,0.25]]), np.array([1.0/6.0]), 4

    def C3D10_DC3D10(self):
        a = (5 + math.sqrt(15)) / 20
        b = (5 - math.sqrt(15)) / 20
        gp = np.array([[b,b,b],[b,b,a],[a,b,b],[b,a,b]])
        W  = np.array([0.25,0.25,0.25,0.25])
        return gp, W, 10

    def C3D10M(self):
        return np.array([[0.25,0.25,0.25]]), np.array([1.0]), 10

    # ---- 2D GP ----
    def CPS4_CPE4_DC2D4(self):
        a = 1/math.sqrt(3)
        gp = [[-a,-a],[ a,-a],[-a, a],[ a, a]]
        W  = [1,1,1,1]
        return gp, W, 4

    def CPS4R_CPE4R(self):
        return [[0,0]],[4],4

    def CPS8_CPE8_DC2D8(self):
        s = math.sqrt(3)/math.sqrt(5)
        gp = [(-s,-s),(0,-s),(s,-s),
              (-s, 0),(0, 0),(s, 0),
              (-s, s),(0, s),(s, s)]
        W  = [25/81,40/81,25/81,
              40/81,64/81,40/81,
              25/81,40/81,25/81]
        return gp, W, 8

    def CPS8R_CPE8R(self):
        a = 1/math.sqrt(3)
        gp = [[-a,-a],[ a,-a],[-a, a],[ a, a]]
        W  = [1,1,1,1]
        return gp, W, 8

    def CPS3_CPE3_DC2D3(self):
        return [[1/3,1/3]],[1/2],3

    def CPS6_CPE6_DC2D6(self):
        gp = [[1/6,1/6],[2/3,1/6],[1/6,2/3]]
        W  = [1/6, 1/6, 1/6]
        return gp, W, 6

    def CPS6R_CPE6R(self):
        return [[1/3,1/3]],[1],6


class Shape_Function(Gauss_Point):
    _c3d20_grad_fun = None  # C3D20 gradient lambdify cache

    def __init__(self, analytical_conditions, element_order, element_shape, integration):
        super().__init__(analytical_conditions, element_order, element_shape, integration)

    def shape(self, gp, NPE, Dimension):
        # 1D edge shape functions (for boundary traction etc.)
        # gp can be scalar or sequence; we take the first component as s in [-1, 1].
        if Dimension == "1D":
            if isinstance(gp, (list, tuple, np.ndarray)):
                s = float(gp[0])
            else:
                s = float(gp)
            if NPE == 2:      # Linear 2-node segment
                N1 = 0.5 * (1.0 - s)
                N2 = 0.5 * (1.0 + s)
                return np.array([N1, N2])
            elif NPE == 3:    # Quadratic 3-node segment (end, mid, end)
                N1 = 0.5 * s * (s - 1.0)
                N2 = 1.0 - s * s
                N3 = 0.5 * s * (s + 1.0)
                return np.array([N1, N2, N3])
            else:
                raise ValueError(f"Unsupported NPE={NPE} for 1D")

        elif Dimension == "2D":
            xi, eta = gp[:2]
            if NPE == 3:      # Linear Tri
                return np.array([1-xi-eta, xi, eta])
            elif NPE == 6:    # Quadratic Tri
                N1 = (1-xi-eta)*(1-2*xi-2*eta)
                N2 = xi*(2*xi-1)
                N3 = eta*(2*eta-1)
                N4 = 4*xi*(1-xi-eta)
                N5 = 4*xi*eta
                N6 = 4*eta*(1-xi-eta)
                return np.array([N1,N2,N3,N4,N5,N6])
            elif NPE == 4:    # Linear Quad
                N = [(1-xi)*(1-eta),(1+xi)*(1-eta),
                     (1+xi)*(1+eta),(1-xi)*(1+eta)]
                return 0.25*np.array(N)
            elif NPE == 8:    # Quadratic Quad
                N = [
                    -(1 - xi)*(1 - eta)*(1 + xi + eta),
                    -(1 + xi)*(1 - eta)*(1 - xi + eta),
                    -(1 + xi)*(1 + eta)*(1 - xi - eta),
                    -(1 - xi)*(1 + eta)*(1 + xi - eta),
                    2*(1 - xi**2)*(1 - eta),
                    2*(1 + xi)*(1 - eta**2),
                    2*(1 - xi**2)*(1 + eta),
                    2*(1 - xi)*(1 - eta**2),
                ]
                return 0.25*np.array(N)
            else:
                raise ValueError(f"Unsupported NPE={NPE} for 2D")

        elif Dimension == "3D":
            xi, eta, zeta = gp[:3]
            if NPE == 4:      # Linear Tet
                return np.array([1-xi-eta-zeta, xi, eta, zeta])
            elif NPE == 10:   # Quadratic Tet
                L1 = 1 - xi - eta - zeta; L2 = xi; L3 = eta; L4 = zeta
                N = [
                    (2*L1-1)*L1, (2*L2-1)*L2, (2*L3-1)*L3, (2*L4-1)*L4,
                    4*L1*L2, 4*L2*L3, 4*L1*L3, 4*L1*L4, 4*L2*L4, 4*L3*L4
                ]
                return np.array(N)
            elif NPE == 8:    # Linear Hex
                N = [
                    (1-xi)*(1-eta)*(1-zeta), (1+xi)*(1-eta)*(1-zeta),
                    (1+xi)*(1+eta)*(1-zeta), (1-xi)*(1+eta)*(1-zeta),
                    (1-xi)*(1-eta)*(1+zeta), (1+xi)*(1-eta)*(1+zeta),
                    (1+xi)*(1+eta)*(1+zeta), (1-xi)*(1+eta)*(1+zeta),
                ]
                return 1/8*np.array(N)
            elif NPE == 20:   # Quadratic Hex (Abaqus order)
                N = [0]*20
                N[0] =  (1/8)*(1 - xi)*(1 - eta)*(1 - zeta)*(-xi - eta - zeta - 2)
                N[1] =  (1/8)*(1 + xi)*(1 - eta)*(1 - zeta)*( xi - eta - zeta - 2)
                N[2] =  (1/8)*(1 + xi)*(1 + eta)*(1 - zeta)*( xi + eta - zeta - 2)
                N[3] =  (1/8)*(1 - xi)*(1 + eta)*(1 - zeta)*(-xi + eta - zeta - 2)
                N[4] =  (1/8)*(1 - xi)*(1 - eta)*(1 + zeta)*(-xi - eta + zeta - 2)
                N[5] =  (1/8)*(1 + xi)*(1 - eta)*(1 + zeta)*( xi - eta + zeta - 2)
                N[6] =  (1/8)*(1 + xi)*(1 + eta)*(1 + zeta)*( xi + eta + zeta - 2)
                N[7] =  (1/8)*(1 - xi)*(1 + eta)*(1 + zeta)*(-xi + eta + zeta - 2)
                N[8]  = (1/4)*(1 - xi**2)*(1 - eta)*(1 - zeta)
                N[9]  = (1/4)*(1 - eta**2)*(1 + xi)*(1 - zeta)
                N[10] = (1/4)*(1 - xi**2)*(1 + eta)*(1 - zeta)
                N[11] = (1/4)*(1 - eta**2)*(1 - xi)*(1 - zeta)
                N[12] = (1/4)*(1 - xi**2)*(1 - eta)*(1 + zeta)
                N[13] = (1/4)*(1 - eta**2)*(1 + xi)*(1 + zeta)
                N[14] = (1/4)*(1 - xi**2)*(1 + eta)*(1 + zeta)
                N[15] = (1/4)*(1 - eta**2)*(1 - xi)*(1 + zeta)
                N[16] = (1/4)*(1 - zeta**2)*(1 - xi)*(1 - eta)
                N[17] = (1/4)*(1 - zeta**2)*(1 + xi)*(1 - eta)
                N[18] = (1/4)*(1 - zeta**2)*(1 + xi)*(1 + eta)
                N[19] = (1/4)*(1 - zeta**2)*(1 - xi)*(1 + eta)
                return np.array(N)
            else:
                raise ValueError(f"Unsupported NPE={NPE} for 3D")

        else:
            raise ValueError(f"Unsupported Dimension: {Dimension}")

    def gradshape(self, gp, NPE, Dimension):
        if Dimension == "2D":
            # normalize gp to a single point if needed
            if not (isinstance(gp, (list, tuple, np.ndarray)) and len(gp) > 0 and isinstance(gp[0], (list, tuple, np.ndarray))):
                point = gp
            else:
                point = gp[0]
            xi, eta = point[:2]

            if NPE == 4:  # linear quad
                local_dN = [
                    [-(1.0 - eta), (1.0 - eta), (1.0 + eta), -(1.0 + eta)],
                    [-(1.0 - xi),  -(1.0 + xi),  (1.0 + xi),   (1.0 - xi)]
                ]
                return 0.25*np.array(local_dN)

            elif NPE == 8:  # quadratic quad
                local_dN = [
                    [(1 - eta)*(2*xi + eta),   (1 - eta)*(2*xi - eta),
                     (1 + eta)*(2*xi + eta),   (1 + eta)*(2*xi - eta),
                     -4*xi*(1 - eta),          2*(1 - eta)*(1 + eta),
                     -4*xi*(1 + eta),         -2*(1 - eta)*(1 + eta)],
                    [(1 - xi)*(xi + 2*eta),    (1 + xi)*(-xi + 2*eta),
                     (1 + xi)*(xi + 2*eta),    (1 - xi)*(-xi + 2*eta),
                     -2*(1 - xi)*(1 + xi),     -4*(1 + xi)*eta,
                      2*(1 - xi)*(1 + xi),     -4*(1 - xi)*eta]
                ]
                return 0.25*np.array(local_dN)

            elif NPE == 3:  # linear tri
                return np.array([[-1, 1, 0],
                                 [-1, 0, 1]])

            elif NPE == 6:  # quadratic tri
                local_dN = [
                    [(-3 + 4*xi + 4*eta), (-1 + 4*xi)        , 0,
                     -4*(-1 + eta + 2*xi),  4*eta            , -4*eta],
                    [(-3 + 4*xi + 4*eta),  0                 , (-1 + 4*eta),
                     -4*xi               ,  4*xi             , -4*(-1 + 2*eta + xi)]
                ]
                return np.array(local_dN)

            else:
                raise ValueError(f"Unsupported NPE: {NPE}")

        elif Dimension == "3D":
            xi, eta, zeta = gp[:3]
            if NPE == 4:  # linear tet
                return np.array([[-1, 1, 0, 0],
                                 [-1, 0, 1, 0],
                                 [-1, 0, 0, 1]])

            elif NPE == 8:  # linear hex
                local_dN = [
                    [-(1 - eta)*(1 - zeta),  (1 - eta)*(1 - zeta),
                      (1 + eta)*(1 - zeta), -(1 + eta)*(1 - zeta),
                     -(1 - eta)*(1 + zeta),  (1 - eta)*(1 + zeta),
                      (1 + eta)*(1 + zeta), -(1 + eta)*(1 + zeta)],
                    [-(1 - xi)*(1 - zeta),  -(1 + xi)*(1 - zeta),
                      (1 + xi)*(1 - zeta),   (1 - xi)*(1 - zeta),
                     -(1 - xi)*(1 + zeta),  -(1 + xi)*(1 + zeta),
                      (1 + xi)*(1 + zeta),   (1 - xi)*(1 + zeta)],
                    [-(1 - xi)*(1 - eta),   -(1 + xi)*(1 - eta),
                     -(1 + xi)*(1 + eta),   -(1 - xi)*(1 + eta),
                      (1 - xi)*(1 - eta),    (1 + xi)*(1 - eta),
                      (1 + xi)*(1 + eta),    (1 - xi)*(1 + eta)]
                ]
                return 1/8*np.array(local_dN)

            elif NPE == 10:  # quadratic tet
                L1 = 1 - xi - eta - zeta; L2 = xi; L3 = eta; L4 = zeta
                dN_dxi   = [-(4*L1 - 1),  4*L2 - 1, 0, 0,  4*(L1 - L2), 4*L3, -4*L3, -4*L4,  4*L4, 0]
                dN_deta  = [-(4*L1 - 1),  0, 4*L3 - 1, 0, -4*L2, 4*L2,  4*(L1 - L3), -4*L4, 0, 4*L4]
                dN_dzeta = [-(4*L1 - 1),  0, 0, 4*L4 - 1, -4*L2, 0, -4*L3, 4*(L1 - L4), 4*L2, 4*L3]
                return np.array([dN_dxi, dN_deta, dN_dzeta])

            elif NPE == 20:  # quadratic hex (Abaqus)
                if Shape_Function._c3d20_grad_fun is None:
                    _xi, _eta, _zeta = sp.symbols('xi eta zeta')
                    _N = [
                        (1/sp.Integer(8))*(1 - _xi)*(1 - _eta)*(1 - _zeta)*(-_xi - _eta - _zeta - 2),
                        (1/sp.Integer(8))*(1 + _xi)*(1 - _eta)*(1 - _zeta)*( _xi - _eta - _zeta - 2),
                        (1/sp.Integer(8))*(1 + _xi)*(1 + _eta)*(1 - _zeta)*( _xi + _eta - _zeta - 2),
                        (1/sp.Integer(8))*(1 - _xi)*(1 + _eta)*(1 - _zeta)*(-_xi + _eta - _zeta - 2),
                        (1/sp.Integer(8))*(1 - _xi)*(1 - _eta)*(1 + _zeta)*(-_xi - _eta + _zeta - 2),
                        (1/sp.Integer(8))*(1 + _xi)*(1 - _eta)*(1 + _zeta)*( _xi - _eta + _zeta - 2),
                        (1/sp.Integer(8))*(1 + _xi)*(1 + _eta)*(1 + _zeta)*( _xi + _eta + _zeta - 2),
                        (1/sp.Integer(8))*(1 - _xi)*(1 + _eta)*(1 + _zeta)*(-_xi + _eta + _zeta - 2),
                        (1/sp.Integer(4))*(1 - _xi**2)*(1 - _eta)*(1 - _zeta),
                        (1/sp.Integer(4))*(1 - _eta**2)*(1 + _xi)*(1 - _zeta),
                        (1/sp.Integer(4))*(1 - _xi**2)*(1 + _eta)*(1 - _zeta),
                        (1/sp.Integer(4))*(1 - _eta**2)*(1 - _xi)*(1 - _zeta),
                        (1/sp.Integer(4))*(1 - _xi**2)*(1 - _eta)*(1 + _zeta),
                        (1/sp.Integer(4))*(1 - _eta**2)*(1 + _xi)*(1 + _zeta),
                        (1/sp.Integer(4))*(1 - _xi**2)*(1 + _eta)*(1 + _zeta),
                        (1/sp.Integer(4))*(1 - _eta**2)*(1 - _xi)*(1 + _zeta),
                        (1/sp.Integer(4))*(1 - _zeta**2)*(1 - _xi)*(1 - _eta),
                        (1/sp.Integer(4))*(1 - _zeta**2)*(1 + _xi)*(1 - _eta),
                        (1/sp.Integer(4))*(1 - _zeta**2)*(1 + _xi)*(1 + _eta),
                        (1/sp.Integer(4))*(1 - _zeta**2)*(1 - _xi)*(1 + _eta),
                    ]
                    _dN_dxi   = [sp.diff(Ni, _xi)   for Ni in _N]
                    _dN_deta  = [sp.diff(Ni, _eta)  for Ni in _N]
                    _dN_dzeta = [sp.diff(Ni, _zeta) for Ni in _N]
                    Shape_Function._c3d20_grad_fun = sp.lambdify((_xi,_eta,_zeta),
                                                                 [_dN_dxi, _dN_deta, _dN_dzeta],
                                                                 'numpy')
                dxi, deta, dzeta = Shape_Function._c3d20_grad_fun(xi, eta, zeta)
                return np.array([dxi, deta, dzeta], dtype=float)

        else:
            raise ValueError(f"Unsupported Dimension: {Dimension}")

    def calculate_N_dN(self, Dimension):
        for gp in self.gp:
            N  = self.shape(gp, self.NPE, Dimension)
            dN = self.gradshape(gp, self.NPE, Dimension)
            print("N: ", N)
            print("dN:", dN)


# ------------------------------
# Global stiffness matrix (K)
# ------------------------------
class Stiffness_Matrix:
    """
    Usage:
      Stiffness_Matrix(E, nu, k_scalar, D_scalar, analytical_conditions, system,
                       Dimension, element_order, element_shape, integration,
                       mesh)
    - mesh: Mesh.from_inp(...) instance
    """
    def __init__(self, E, nu, k_scalar, D_scalar,
                 analytical_conditions, system,
                 Dimension, element_order, element_shape, integration,
                 mesh):
        # Coerce settings from INP etype / mesh
        etype = getattr(mesh, "etype", None) or getattr(mesh, "etype_from_inp", None)
        dim2, shape2, order2 = _resolve_from_etype(etype) if etype else (None, None, None)

        Dimension     = dim2   or Dimension
        element_shape = shape2 or element_shape
        element_order = order2 or element_order

        if (Dimension is None) or (element_shape is None) or (element_order is None):
            d3, s3, o3 = _infer_from_mesh(mesh)
            Dimension     = Dimension     or d3
            element_shape = element_shape or s3
            element_order = element_order or o3

        # safety
        if isinstance(analytical_conditions, str) and analytical_conditions.strip().lower() == "none":
            analytical_conditions = None
        assert Dimension in ("2D","3D"), f"Dimension must be 2D/3D, got {Dimension}"
        assert element_shape in ("Quad","Tri","Hex","Tet"), f"element_shape invalid: {element_shape}"
        assert element_order in ("Linear","Quadratic"), f"element_order invalid: {element_order}"

        # store
        self.system            = system
        self.Dimension         = Dimension
        self.element_order     = element_order
        self.element_shape     = element_shape
        self.integration       = integration
        self.mesh              = mesh

        # shape functions / GP
        self.sf = Shape_Function(analytical_conditions, element_order, element_shape, integration)

        # convenient refs
        self.NL   = mesh.NL
        self.conn = mesh.conn
        self.NoN  = mesh.NoN
        self.NPE  = self.sf.NPE
        self.gp   = self.sf.gp
        self.W    = self.sf.W

        # material
        self.mat = Material_Property(system, Dimension)
        self.t   = self.mat.t  # 2D thickness (=1)

        if self.system == "linear elasticity":
            if self.Dimension == "2D":
                self.D = self.mat.Hookean_matrix_2D(E, nu, analytical_conditions)
            else:  # 3D
                self.D = self.mat.Hookean_matrix_3D(E, nu, analytical_conditions)
        elif self.system == "heat transfer":
            self.D = self.mat.conductance(k_scalar)   # isotropic k
        elif self.system == "diffusion":
            self.D = self.mat.diffusivity(D_scalar)   # isotropic D
        else:
            raise ValueError("Unsupported system")

    @staticmethod
    def _ensure_positive_det(detJ, tol=1e-14):
        if detJ <= tol:
            raise ValueError(f"Non-positive Jacobian determinant detected: detJ={detJ}")

    def global_K(self):
        if self.Dimension == "2D":
            if self.system == "linear elasticity":
                K = np.zeros((2*self.NoN, 2*self.NoN))
                B = np.zeros((3, 2*self.NPE))
                for c in self.conn:
                    xIe = self.NL[c, :]               # (NPE, 2)
                    Ke  = np.zeros((2*self.NPE, 2*self.NPE))
                    if self.element_shape in ("Quad", "Tri"):
                        for gp, w in zip(self.gp, self.W):
                            B.fill(0.0)
                            local_dN = self.sf.gradshape(gp, self.NPE, self.Dimension)  # (2,NPE)
                            J        = local_dN @ xIe                                    # (2,2)
                            detJ     = np.linalg.det(J)
                            self._ensure_positive_det(detJ)
                            
                            global_dN= np.linalg.solve(J, local_dN)                      # inv(J)@dN
                            B[0, 0::2] = global_dN[0, :]
                            B[1, 1::2] = global_dN[1, :]
                            B[2, 0::2] = global_dN[1, :]
                            B[2, 1::2] = global_dN[0, :]
                            Ke += self.t * w * (B.T @ self.D @ B) * detJ
                    # assemble
                    for i, I in enumerate(c):
                        for j, J_ in enumerate(c):
                            K[2*I,   2*J_  ] += Ke[2*i,   2*j  ]
                            K[2*I+1, 2*J_  ] += Ke[2*i+1, 2*j  ]
                            K[2*I+1, 2*J_+1] += Ke[2*i+1, 2*j+1]
                            K[2*I,   2*J_+1] += Ke[2*i,   2*j+1]
                return K

            elif self.system in ("heat transfer", "diffusion"):
                K = np.zeros((self.NoN, self.NoN))
                B = np.zeros((2, self.NPE))
                for c in self.conn:
                    xIe = self.NL[c, :]
                    Ke  = np.zeros((self.NPE, self.NPE))
                    for gp, w in zip(self.gp, self.W):
                        B.fill(0.0)
                        local_dN = self.sf.gradshape(gp, self.NPE, self.Dimension)
                        J        = local_dN @ xIe
                        detJ     = np.linalg.det(J)
                        self._ensure_positive_det(detJ)
                        global_dN= np.linalg.solve(J, local_dN)
                        B[0,:] = global_dN[0,:]
                        B[1,:] = global_dN[1,:]
                        Ke += self.t * w * (B.T @ self.D @ B) * detJ
                    for i, I in enumerate(c):
                        for j, J_ in enumerate(c):
                            K[I, J_] += Ke[i, j]
                return K

        elif self.Dimension == "3D":
            if self.system == "linear elasticity":
                K = np.zeros((3*self.NoN, 3*self.NoN))
                B = np.zeros((6, 3*self.NPE))
                for c in self.conn:
                    xIe = self.NL[c, :]                     # (NPE, 3)
                    Ke  = np.zeros((3*self.NPE, 3*self.NPE))
                    for gp, w in zip(self.gp, self.W):
                        B.fill(0.0)
                        local_dN = self.sf.gradshape(gp, self.NPE, self.Dimension)  # (3,NPE)
                        J        = local_dN @ xIe                                    # (3,3)
                        detJ     = np.linalg.det(J)
                        self._ensure_positive_det(detJ)
                        global_dN= np.linalg.solve(J, local_dN)                      # (3,NPE)

                        B[0, 0::3] = global_dN[0,:]   # εxx
                        B[1, 1::3] = global_dN[1,:]   # εyy
                        B[2, 2::3] = global_dN[2,:]   # εzz
                        B[3, 0::3] = global_dN[1,:];  B[3, 1::3] = global_dN[0,:]   # γxy
                        B[4, 1::3] = global_dN[2,:];  B[4, 2::3] = global_dN[1,:]   # γyz
                        B[5, 2::3] = global_dN[0,:];  B[5, 0::3] = global_dN[2,:]   # γzx

                        vol_factor = 1.0 if self.element_shape == "Hex" else (1.0/6.0)  # Tet
                        Ke += w * (B.T @ self.D @ B) * detJ * vol_factor

                    # assemble
                    for i, I in enumerate(c):
                        for j, J_ in enumerate(c):
                            for di in range(3):
                                for dj in range(3):
                                    K[3*I+di, 3*J_+dj] += Ke[3*i+di, 3*j+dj]
                return K

            elif self.system in ("heat transfer", "diffusion"):
                # 3D scalar field: B is (3,NPE)
                K = np.zeros((self.NoN, self.NoN))
                B = np.zeros((3, self.NPE))
                for c in self.conn:
                    xIe = self.NL[c, :]
                    Ke  = np.zeros((self.NPE, self.NPE))
                    for gp, w in zip(self.gp, self.W):
                        B.fill(0.0)
                        local_dN = self.sf.gradshape(gp, self.NPE, self.Dimension)  # (3,NPE)
                        J        = local_dN @ xIe                                    # (3,3)
                        detJ     = np.linalg.det(J)
                        self._ensure_positive_det(detJ)
                        global_dN= np.linalg.solve(J, local_dN)
                        B[0,:] = global_dN[0,:]
                        B[1,:] = global_dN[1,:]
                        B[2,:] = global_dN[2,:]
                        vol_factor = 1.0 if self.element_shape == "Hex" else (1.0/6.0)  # Tet
                        Ke += w * (B.T @ self.D @ B) * detJ * vol_factor
                    for i, I in enumerate(c):
                        for j, J_ in enumerate(c):
                            K[I, J_] += Ke[i, j]
                return K

        # bad combination
        raise ValueError(f"Unsupported combination: Dim={self.Dimension}, system={self.system}, "
                         f"shape={self.element_shape}, order={self.element_order}, integ={self.integration}")

    # ------------------------------
    # Consistent surface loads for C3D8
    # ------------------------------
    # Face identifiers for convenience:
    #   'XI_M'  : xi = -1  (−x face)
    #   'XI_P'  : xi = +1  (+x face)
    #   'ETA_M' : eta = -1 (−y face)
    #   'ETA_P' : eta = +1 (+y face)
    #   'ZETA_M': zeta = -1(−z face)
    #   'ZETA_P': zeta = +1(+z face)

    @staticmethod
    def _face_gauss_points_2x2():
        a = 1.0/math.sqrt(3.0)
        return [(-a, -a), ( a, -a), (-a,  a), ( a,  a)], [1.0, 1.0, 1.0, 1.0]

    def _eval_parent_shape_and_grad_hex8(self, xi, eta, zeta):
        N  = self.sf.shape([xi,eta,zeta], 8, "3D")
        dN = self.sf.gradshape([xi,eta,zeta], 8, "3D")  # (3,8)
        return N, dN

    def _surface_tangents(self, dN, xIe, vary_idx_a, vary_idx_b):
        # dN: (3,8) with rows [dN/dxi, dN/deta, dN/dzeta]
        dx_a = dN[vary_idx_a, :] @ xIe  # (3,)
        dx_b = dN[vary_idx_b, :] @ xIe  # (3,)
        return dx_a, dx_b

    def face_traction_C3D8(self, xIe, face, traction_vec):
        """Consistent nodal forces for a constant traction vector (global) on a C3D8 face.
        xIe: (8,3) element node coords (Abaqus node order)
        face: one of {'XI_M','XI_P','ETA_M','ETA_P','ZETA_M','ZETA_P'}
        traction_vec: (3,) global vector, e.g., [tx,ty,tz]
        returns: Fe (24,) element force vector
        """
        assert self.element_shape == "Hex" and self.NPE == 8, "Only C3D8 supported here."
        Fe = np.zeros(3*self.NPE)
        gp2d, w2d = self._face_gauss_points_2x2()

        for (r,s), w in zip(gp2d, w2d):
            if face == 'ZETA_P':
                xi, eta, zeta = r, s, 1.0;  a, b = 0, 1   # vary: xi, eta
            elif face == 'ZETA_M':
                xi, eta, zeta = r, s, -1.0; a, b = 0, 1
            elif face == 'ETA_P':
                xi, eta, zeta = r, 1.0, s;  a, b = 0, 2   # vary: xi, zeta
            elif face == 'ETA_M':
                xi, eta, zeta = r, -1.0, s; a, b = 0, 2
            elif face == 'XI_P':
                xi, eta, zeta = 1.0, r, s;  a, b = 1, 2   # vary: eta, zeta
            elif face == 'XI_M':
                xi, eta, zeta = -1.0, r, s; a, b = 1, 2
            else:
                raise ValueError(f"Unknown face key: {face}")

            N, dN = self._eval_parent_shape_and_grad_hex8(xi, eta, zeta)
            dx1, dx2 = self._surface_tangents(dN, xIe, a, b)
            Js = np.linalg.norm(np.cross(dx1, dx2))  # surface Jacobian magnitude

            # Distribute traction as Fe += N^T ⊗ I3 * t * Js * w
            Fe += (np.kron(N, np.eye(3)) @ traction_vec) * Js * w

        return Fe

    def face_pressure_C3D8(self, xIe, face, pressure):
        """Consistent nodal forces for a constant pressure p on a C3D8 face.
        Pressure is assumed INWARD (negative along outward normal). This convention
        avoids needing to know the outward normal sign; we always push inward.
        """
        assert self.element_shape == "Hex" and self.NPE == 8, "Only C3D8 supported here."
        Fe = np.zeros(3*self.NPE)
        gp2d, w2d = self._face_gauss_points_2x2()

        for (r,s), w in zip(gp2d, w2d):
            if face == 'ZETA_P':
                xi, eta, zeta = r, s, 1.0;  a, b = 0, 1
            elif face == 'ZETA_M':
                xi, eta, zeta = r, s, -1.0; a, b = 0, 1
            elif face == 'ETA_P':
                xi, eta, zeta = r, 1.0, s;  a, b = 0, 2
            elif face == 'ETA_M':
                xi, eta, zeta = r, -1.0, s; a, b = 0, 2
            elif face == 'XI_P':
                xi, eta, zeta = 1.0, r, s;  a, b = 1, 2
            elif face == 'XI_M':
                xi, eta, zeta = -1.0, r, s; a, b = 1, 2
            else:
                raise ValueError(f"Unknown face key: {face}")

            N, dN = self._eval_parent_shape_and_grad_hex8(xi, eta, zeta)
            dx1, dx2 = self._surface_tangents(dN, xIe, a, b)
            n_vec = np.cross(dx1, dx2)
            Js = np.linalg.norm(n_vec)
            if Js <= 0.0:
                raise ValueError("Degenerate face with zero area detected.")
            n_hat = n_vec / Js

            # Pressure acts inward: t = -p * n_hat
            traction_vec = -pressure * n_hat
            Fe += (np.kron(N, np.eye(3)) @ traction_vec) * Js * w

        return Fe
