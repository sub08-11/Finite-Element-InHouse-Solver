# -*- coding: utf-8 -*-
# made by changsub
# Preprocessing utilities for Abaqus INP-driven analyses.
# Provides the Material_Property class (Hookean elasticity, thermal
# conductivity, and diffusivity tensors) and the Mesh/INP parsing
# routines that read *Node and *Element data, build node/element
# arrays, track ID–row mappings, and prepare mesh metadata for the
# main FEM workflow.

import os
import numpy as np
from typing import Dict, List, Tuple, Optional

############################# Material (원본 유지)
class Material_Property:
    def __init__(self, system, Dimension):
        self.system = system
        self.Dimension = Dimension
        self.t = 1

    def Hookean_matrix_2D(self, E, nu, analytical_conditions):
        if analytical_conditions == "plane strain":
            D = E / ((1.0 + nu) * (1.0 - 2.0 * nu)) * np.array([
                [1.0 - nu, nu, 0.0],
                [nu, 1.0 - nu, 0.0],
                [0.0, 0.0, (1.0 - 2 * nu) / 2],
            ])
        elif analytical_conditions == "plane stress":
            D = E / (1.0 - nu**2) * np.array([
                [1.0, nu, 0.0],
                [nu, 1.0, 0.0],
                [0.0, 0.0, (1.0 - 2 * nu) / 2],
            ])
        else:
            raise ValueError("Invalid condition. Error")
        return D

    def Hookean_matrix_3D(self, E, nu, analytical_conditions):
        D_3D = E / ((1 + nu) * (1 - 2 * nu)) * np.array([
            [1 - nu, nu, nu, 0, 0, 0],
            [nu, 1 - nu, nu, 0, 0, 0],
            [nu, nu, 1 - nu, 0, 0, 0],
            [0, 0, 0, (1 - 2 * nu) / 2, 0, 0],
            [0, 0, 0, 0, (1 - 2 * nu) / 2, 0],
            [0, 0, 0, 0, 0, (1 - 2 * nu) / 2],
        ])
        return D_3D

    def conductance(self, k):
        if self.Dimension == "2D":
            I = np.eye(2)
        elif self.Dimension == "3D":
            I = np.eye(3)
        else:
            raise ValueError("Dimension must be 2D or 3D")
        return k * I

    def diffusivity(self, D):
        if self.Dimension == "2D":
            I = np.eye(2)
        elif self.Dimension == "3D":
            I = np.eye(3)
        else:
            raise ValueError("Dimension must be 2D or 3D")
        return D * I


############################# INP helpers (노드/요소만)
def _is_section_start(s: str) -> bool:
    return s.lstrip().startswith('*')

def _clean(s: str) -> str:
    return s.strip()

def _parse_floats_csv(s: str) -> List[float]:
    parts = [t.strip() for t in s.replace('\t', ' ').split(',')]
    return [float(p) for p in parts if p != '']

def _parse_ints_csv(s: str) -> List[int]:
    return [int(round(v)) for v in _parse_floats_csv(s)]

def _expected_npe(etype: str) -> int:
    e = etype.strip().upper()
    # 2D
    if e.startswith(("CPE3", "CPS3")): return 3
    if e.startswith(("CPE4", "CPS4")): return 4
    if e.startswith(("CPE6", "CPS6")): return 6
    if e.startswith(("CPE8", "CPS8")): return 8
    # 3D
    if e.startswith("C3D4"):  return 4
    if e.startswith("C3D8"):  return 8
    if e.startswith("C3D10"): return 10
    if e.startswith("C3D20"): return 20
    if e.startswith("C3D20R"):return 20
    return 0  # unknown → one-element-per-line fallback

def _read_inp_nodes(lines: List[str], i0: int) -> Tuple[Dict[int, np.ndarray], int]:
    nodes: Dict[int, np.ndarray] = {}
    i = i0 + 1
    while i < len(lines) and not _is_section_start(lines[i]):
        line = _clean(lines[i])
        if not line or line.startswith('**'):
            i += 1; continue
        vals = _parse_floats_csv(line)
        if len(vals) < 3:
            i += 1; continue
        nid = int(round(vals[0]))
        if len(vals) >= 4:   # 3D
            nodes[nid] = np.array([vals[1], vals[2], vals[3]], dtype=float)
        else:                # 2D
            nodes[nid] = np.array([vals[1], vals[2]], dtype=float)
        i += 1
    return nodes, i

def _read_inp_elements(lines: List[str], i0: int) -> Tuple[str, str, List[Tuple[int, List[int]]], int]:
    header = lines[i0].strip()
    etype, elset = None, ""
    for part in header.split(','):
        p = part.strip()
        if p.upper().startswith("TYPE="):
            etype = p.split('=')[1].strip()
        elif p.upper().startswith("ELSET="):
            elset = p.split('=')[1].strip()
    if etype is None:
        raise ValueError(f"*Element header missing TYPE=: {header}")

    npe = _expected_npe(etype)
    elems: List[Tuple[int, List[int]]] = []

    i = i0 + 1
    pending_eid: Optional[int] = None
    pending_nodes: List[int] = []

    while i < len(lines) and not _is_section_start(lines[i]):
        line = _clean(lines[i])
        if not line or line.startswith('**'):
            i += 1; continue
        ints = _parse_ints_csv(line)
        if not ints:
            i += 1; continue

        idx = 0
        while idx < len(ints):
            if pending_eid is None:
                pending_eid = ints[idx]
                pending_nodes = []
                idx += 1
                if npe == 0:
                    if idx < len(ints):
                        elems.append((pending_eid, ints[idx:]))
                    pending_eid = None
                    pending_nodes = []
                    idx = len(ints)
                    break
            else:
                need = npe - len(pending_nodes) if npe > 0 else 0
                take = min(need, len(ints) - idx) if npe > 0 else 0
                if take > 0:
                    pending_nodes.extend(ints[idx:idx+take])
                    idx += take
                if npe > 0 and len(pending_nodes) == npe:
                    elems.append((pending_eid, pending_nodes[:]))
                    pending_eid = None
                    pending_nodes = []
        i += 1

    if pending_eid is not None:
        raise ValueError(f"Incomplete element {pending_eid}: got {len(pending_nodes)}/{npe} node IDs")

    return etype, elset, elems, i

def load_nodes_elements(inp_path: str,
                        prefer_type: Optional[str] = None,
                        prefer_elset: Optional[str] = None,
                        use_only_used_nodes: bool = True):
    if not os.path.isfile(inp_path):
        raise FileNotFoundError(inp_path)

    with open(inp_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    node_map: Dict[int, np.ndarray] = {}
    elem_blocks: List[Tuple[str, str, List[Tuple[int, List[int]]]]] = []

    i = 0
    while i < len(lines):
        up = lines[i].strip().upper()

        if up.startswith("*NODE"):
            nodes, i = _read_inp_nodes(lines, i)
            node_map.update(nodes)
            continue

        if up.startswith("*ELEMENT"):
            if "TYPE=" not in up:
                i += 1
                while i < len(lines) and not _is_section_start(lines[i]):
                    i += 1
                continue
            et, es, elems, i2 = _read_inp_elements(lines, i)
            elem_blocks.append((et, es, elems))
            i = i2
            continue

        if up.startswith("*NSET") or up.startswith("*STEP"):
            break

        i += 1

    if not node_map:
        raise ValueError("No *Node block found.")
    if not elem_blocks:
        raise ValueError("No *Element block found.")

    chosen = elem_blocks[0]
    etype, elset, elems = chosen

    # Dimension, element_shape, element_order는 main 코드 입력값 사용 → 여기서 자동 판별만 수행
    Dimension, element_shape, element_order = None, None, None

    conn_by_id = [conn for (_eid, conn) in elems]
    used_ids = sorted({nid for c in conn_by_id for nid in c}) if use_only_used_nodes else sorted(node_map.keys())

    sample = node_map[used_ids[0]]
    if len(sample) == 3:
        NL = np.array([[node_map[nid][0], node_map[nid][1], node_map[nid][2]] for nid in used_ids], dtype=float)
    else:
        NL = np.array([[node_map[nid][0], node_map[nid][1]] for nid in used_ids], dtype=float)

    # 좌표 열 수로 차원 확정
    coord_dim = NL.shape[1]
    Dimension = "3D" if coord_dim == 3 else "2D"

    id2row = {nid: i for i, nid in enumerate(used_ids)}
    row2id = {i: nid for nid, i in id2row.items()}
    conn = [[id2row[nid] for nid in c] for c in conn_by_id]

    return NL, conn, Dimension, element_shape, element_order, etype, id2row, row2id


############################# Mesh (INP 전용; as-is NL/conn)
class Mesh:
    def __init__(self, element_shape=None, element_order=None, Dimension=None):
        self.element_shape = element_shape
        self.element_order = element_order
        self.Dimension = Dimension

    def print_nodes(self):
        print("\n[Nodes]")
        # NL의 열 수로 2D/3D를 판별 (Dimension이 None이어도 안전)
        coord_dim = self.NL.shape[1]
        if coord_dim == 2:
            print(" row  nid   x           y")
            for i in range(self.NoN):
                nid = self.row2id[i]
                x, y = self.NL[i, 0], self.NL[i, 1]
                print(f"{i:4d}  {nid:4d}  {x:11.6f}  {y:11.6f}")
        else:
            print(" row  nid   x           y           z")
            for i in range(self.NoN):
                nid = self.row2id[i]
                x, y, z = self.NL[i, 0], self.NL[i, 1], self.NL[i, 2]
                print(f"{i:4d}  {nid:4d}  {x:11.6f}  {y:11.6f}  {z:11.6f}")

    def print_elements(self):
        print("\n[Elements]")
        print(" eidx  conn(0-base)     node IDs(from INP)")
        for eidx, c in enumerate(self.conn):
            c0 = " ".join(f"{v:3d}" for v in c)
            ids = " ".join(f"{self.row2id[v]:3d}" for v in c)
            print(f"{eidx:4d}  [{c0}]   [{ids}]")

    # --- 추가: C3D4 오리엔테이션 표준화 (detJ>0) ---
    def _normalize_c3d4_orientation(self, warn_zero_det: bool = True) -> int:
        """
        3D 선형 테트라(C3D4) 요소의 로컬 노드 순서를 detJ>0(오른손계)로 표준화한다.
        detJ<0이면 로컬 (1↔2) 스왑(0-based로 [1,2] 교환).
        반환값: 스왑 수행된 요소 개수
        """
        if self.Dimension != "3D" or self.NPE != 4 or self.NoE == 0:
            return 0

        swapped = 0
        EL = self.EL.copy()  # (NoE, 4)
        X = self.NL

        for e in range(self.NoE):
            i1, i2, i3, i4 = EL[e, :]
            x1, x2, x3, x4 = X[i1], X[i2], X[i3], X[i4]
            v1 = x2 - x1
            v2 = x3 - x1
            v3 = x4 - x1
            detJ = np.linalg.det(np.column_stack([v1, v2, v3]))

            if detJ < 0.0:
                # swap local 2 <-> 3  (indices 1 and 2 in 0-based)
                EL[e, [1, 2]] = EL[e, [2, 1]]
                swapped += 1
            elif warn_zero_det and abs(detJ) < 1e-16:
                # 퇴화 요소 경고 (필요시 주석처리 가능)
                print(f"[warn] degenerate C3D4 element (detJ≈0) at eidx={e}, ids={self.row2id[EL[e,0]]},{self.row2id[EL[e,1]]},{self.row2id[EL[e,2]]},{self.row2id[EL[e,3]]}")

        if swapped:
            # conn / EL 모두 업데이트
            self.EL = EL
            self.conn = [list(EL[e, :]) for e in range(self.NoE)]
        return swapped

    @classmethod
    def from_inp(cls,
                 inp_path: str,
                 prefer_type: Optional[str] = None,
                 prefer_elset: Optional[str] = None,
                 use_only_used_nodes: bool = True,
                 debug: bool = False):
        (NL, conn, Dimension, element_shape, element_order,
         etype, id2row, row2id) = load_nodes_elements(
            inp_path, prefer_type=prefer_type, prefer_elset=prefer_elset, use_only_used_nodes=use_only_used_nodes
        )

        m = cls()
        m.NL = np.array(NL, dtype=float)
        m.conn = [list(c) for c in conn]
        # NL 열 수로 최종 차원 확정 (상위에서 None이 오더라도 안전)
        m.Dimension = Dimension or ("3D" if m.NL.shape[1] == 3 else "2D")
        m.element_shape = element_shape
        m.element_order = element_order
        m.inp_element_type = etype
        m.id2row = dict(id2row)
        m.row2id = dict(row2id)

        m.NoN = m.NL.shape[0]
        m.NoE = len(m.conn)
        m.NPE = len(m.conn[0]) if m.NoE else 0
        m.EL  = np.array(m.conn, dtype=int)

        # --- 여기서 C3D4 오리엔테이션 표준화 수행 ---
        swapped = m._normalize_c3d4_orientation()
        if debug and swapped:
            print(f"[Mesh.from_inp] normalized C3D4 orientation on {swapped} elements (detJ>0 enforced).")

        if debug:
            print(f"[Mesh.from_inp] type={etype}, Dim={m.Dimension}, Shape={m.element_shape}, Order={m.element_order}")
            print(f"  NoN={m.NoN}, NoE={m.NoE}, NPE={m.NPE}")
            if m.NoE:
                print("  First element (as-is / or normalized):", m.conn[0])

            m.print_nodes()
            m.print_elements()
        return m
