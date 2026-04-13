import numpy as np
import sympy as sp
from sympy import *
from sympy import solve
import inspect
import random
import math


import numpy as np
import sympy as sp
from sympy import solve, sympify
import math
 
# Canonical free variable
x = sp.Symbol("x")
 
 
def sign(integer):
    return "" if integer < 0 else "+"
 
 
# ---------------------------------------------------------------------------
# Quaternion
# ---------------------------------------------------------------------------
 
class Quaternion:
    def __init__(self, w, x, y, z):
        self.w = sp.sympify(w)
        self.x = sp.sympify(x)
        self.y = sp.sympify(y)
        self.z = sp.sympify(z)
 
    def __repr__(self):
        # Use evalf so symbolic components render as floats when possible,
        # but fall back to the symbolic form cleanly.
        def fmt(v):
            try:
                return f"{float(v):.4g}"
            except (TypeError, ValueError):
                return str(v)
        return f"Quaternion({fmt(self.w)}, {fmt(self.x)}, {fmt(self.y)}, {fmt(self.z)})"
 
    def norm(self):
        return sp.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
 
    def normalize(self):
        n = self.norm()
        if n == 0:
            raise ZeroDivisionError("Cannot normalize a zero quaternion.")
        self.w /= n
        self.x /= n
        self.y /= n
        self.z /= n
        return self
 
    def conjugate(self):
        return Quaternion(self.w, -self.x, -self.y, -self.z)
 
    def inverse(self):
        n2 = self.norm() ** 2
        if n2 == 0:
            raise ZeroDivisionError("Cannot invert a zero quaternion.")
        return self.conjugate() * (sp.Integer(1) / n2)
 
    def __mul__(self, other):
        if isinstance(other, Quaternion):
            w = self.w*other.w - self.x*other.x - self.y*other.y - self.z*other.z
            x_ = self.w*other.x + self.x*other.w + self.y*other.z - self.z*other.y
            y_ = self.w*other.y - self.x*other.z + self.y*other.w + self.z*other.x
            z_ = self.w*other.z + self.x*other.y - self.y*other.x + self.z*other.w
            return Quaternion(w, x_, y_, z_)
        return Quaternion(self.w*other, self.x*other, self.y*other, self.z*other)
 
    def rotate_vector(self, vector):
        qv = Quaternion(0, vector.x, vector.y, vector.z)
        qr = self * qv * self.inverse()
        return Vector3(qr.x, qr.y, qr.z)
 
 
# ---------------------------------------------------------------------------
# Vector3
# ---------------------------------------------------------------------------
 
class Vector3:
    __slots__ = ("x", "y", "z")
 
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = sp.sympify(x)
        self.y = sp.sympify(y)
        self.z = sp.sympify(z)
 
    def _fmt(self, v):
        try:
            return f"{float(v):.6g}"
        except (TypeError, ValueError):
            return str(v)
 
    def __repr__(self):
        return f"Vector3({self._fmt(self.x)}, {self._fmt(self.y)}, {self._fmt(self.z)})"
 
    def __str__(self):
        return f"<{self.x}, {self.y}, {self.z}>"
 
    @property
    def components(self):
        return f"{self.x}i + {self.y}j + {self.z}k"
 
    def copy(self):
        return Vector3(self.x, self.y, self.z)
 
    def as_tuple(self):
        return (self.x, self.y, self.z)
 
    def as_array(self):
        return np.array([float(self.x), float(self.y), float(self.z)])
 
    def norm(self):
        return sp.sqrt(self.x*self.x + self.y*self.y + self.z*self.z)
 
    def norm_sq(self):
        return self.x*self.x + self.y*self.y + self.z*self.z
 
    def __abs__(self):
        return self.norm()
 
    def normalize(self):
        n = self.norm()
        if n == 0:
            raise ZeroDivisionError("Cannot normalize zero vector")
        self.x /= n
        self.y /= n
        self.z /= n
        return self
 
    def normalized(self):
        n = self.norm()
        if n == 0:
            raise ZeroDivisionError("Cannot normalize zero vector")
        return Vector3(self.x/n, self.y/n, self.z/n)
 
    def __add__(self, other):
        return Vector3(self.x+other.x, self.y+other.y, self.z+other.z)
 
    def __sub__(self, other):
        return Vector3(self.x-other.x, self.y-other.y, self.z-other.z)
 
    def __mul__(self, scalar):
        return Vector3(self.x*scalar, self.y*scalar, self.z*scalar)
 
    def __rmul__(self, scalar):
        return self * scalar
 
    def __truediv__(self, scalar):
        return Vector3(self.x/scalar, self.y/scalar, self.z/scalar)
 
    def __neg__(self):
        return Vector3(-self.x, -self.y, -self.z)
 
    def dot(self, other):
        return self.x*other.x + self.y*other.y + self.z*other.z
 
    def cross(self, other):
        return Vector3(
            self.y*other.z - self.z*other.y,
            self.z*other.x - self.x*other.z,
            self.x*other.y - self.y*other.x
        )
 
    def distance_to(self, other):
        return (self - other).norm()
 
    def project(self, other):
        d = other.norm_sq()
        if d == 0:
            raise ZeroDivisionError("Projection onto zero vector")
        return other * (self.dot(other) / d)
 
    def reject(self, other):
        return self - self.project(other)
 
    def angle_to(self, other):
        m = float(self.norm()) * float(other.norm())
        if m == 0:
            raise ZeroDivisionError("Angle undefined for zero vector")
        return math.acos(max(-1.0, min(1.0, float(self.dot(other)) / m)))
 
    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z
 
    def __getitem__(self, i):
        if i == 0: return self.x
        if i == 1: return self.y
        if i == 2: return self.z
        raise IndexError
 
    def __eq__(self, other):
        return isinstance(other, Vector3) and \
               self.x == other.x and self.y == other.y and self.z == other.z
 
    def rotated(self, quaternion):
        return quaternion.rotate_vector(self)
 
 
# ---------------------------------------------------------------------------
# Vector2
# ---------------------------------------------------------------------------
 
class Vector2:
    __slots__ = ("x", "y")
 
    def __init__(self, x=0.0, y=0.0):
        self.x = sp.sympify(x)
        self.y = sp.sympify(y)
 
    def _fmt(self, v):
        try:
            return f"{float(v):.6g}"
        except (TypeError, ValueError):
            return str(v)
 
    def __repr__(self):
        return f"Vector2({self._fmt(self.x)}, {self._fmt(self.y)})"
 
    def __str__(self):
        return f"<{self.x}, {self.y}>"
 
    def copy(self):
        return Vector2(self.x, self.y)
 
    def as_tuple(self):
        return (self.x, self.y)
 
    def as_array(self):
        return np.array([float(self.x), float(self.y)])
 
    def norm(self):
        return sp.sqrt(self.x*self.x + self.y*self.y)
 
    def __abs__(self):
        return self.norm()
 
    def norm_sq(self):
        return self.x*self.x + self.y*self.y
 
    def normalize(self):
        n = self.norm()
        if n == 0:
            raise ZeroDivisionError("Cannot normalize zero vector")
        self.x /= n
        self.y /= n
        return self
 
    def normalized(self):
        n = self.norm()
        if n == 0:
            raise ZeroDivisionError("Cannot normalize zero vector")
        return Vector2(self.x/n, self.y/n)
 
    def __add__(self, other):
        return Vector2(self.x+other.x, self.y+other.y)
 
    def __sub__(self, other):
        return Vector2(self.x-other.x, self.y-other.y)
 
    def __mul__(self, scalar):
        return Vector2(self.x*scalar, self.y*scalar)
 
    def __rmul__(self, scalar):
        return self * scalar
 
    def __truediv__(self, scalar):
        return Vector2(self.x/scalar, self.y/scalar)
 
    def __neg__(self):
        return Vector2(-self.x, -self.y)
 
    def dot(self, other):
        return self.x*other.x + self.y*other.y
 
    def cross(self, other):
        return self.x*other.y - self.y*other.x
 
    def angle(self):
        return sp.atan2(self.y, self.x)
 
    def angle_to(self, other):
        if not isinstance(other, Vector2):
            raise TypeError(f"angle_to requires Vector2, not {type(other)}")
        m = self.norm() * other.norm()
        if m == 0:
            raise ZeroDivisionError("Angle undefined for zero vector")
        return sp.acos(sp.Rational(max(-1, min(1, self.dot(other)/m))))
 
    def distance_to(self, other):
        return (self - other).norm()
 
    def project(self, other):
        d = other.norm_sq()
        if d == 0:
            raise ZeroDivisionError("Projection onto zero vector")
        return other * (self.dot(other) / d)
 
    def reject(self, other):
        return self - self.project(other)
 
    def rotate(self, theta):
        c = sp.cos(theta)
        s = sp.sin(theta)
        return Vector2(self.x*c - self.y*s, self.x*s + self.y*c)
 
    def perpendicular(self):
        return Vector2(-self.y, self.x)
 
    def lerp(self, other, t):
        return Vector2(self.x + (other.x-self.x)*t, self.y + (other.y-self.y)*t)
 
    @staticmethod
    def from_polar(r, theta):
        return Vector2(r*sp.cos(theta), r*sp.sin(theta))
 
    def __iter__(self):
        yield self.x
        yield self.y
 
    def __getitem__(self, i):
        if i == 0: return self.x
        if i == 1: return self.y
        raise IndexError
 
    def __eq__(self, other):
        return isinstance(other, Vector2) and self.x == other.x and self.y == other.y
 
 
# ---------------------------------------------------------------------------
# Matrix
# ---------------------------------------------------------------------------
 
class Matrix:
    def __init__(self, *rows):
        # Accept either Matrix(*rows) or Matrix(list_of_rows)
        if len(rows) == 1 and isinstance(rows[0], list) and rows[0] and isinstance(rows[0][0], list):
            rows = rows[0]
 
        if not rows:
            raise ValueError("Matrix cannot be empty")
        if not all(len(row) == len(rows[0]) for row in rows):
            raise ValueError("Rows must have equal length")
 
        self.data = [list(row) for row in rows]
        self.rows = len(self.data)
        self.cols = len(self.data[0])
 
    def __str__(self):
        return "\n".join(" ".join(str(v) for v in row) for row in self.data)
 
    def __repr__(self):
        return f"Matrix({self.data})"
 
    def shape(self):
        return (self.rows, self.cols)
 
    def copy(self):
        return Matrix([row[:] for row in self.data])
 
    def __getitem__(self, idx):
        return self.data[idx]
 
    def __eq__(self, other):
        return isinstance(other, Matrix) and self.data == other.data
 
    def __add__(self, other):
        if self.shape() != other.shape():
            raise ValueError("Matrix dimensions must match for addition")
        return Matrix([
            [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])
 
    def __sub__(self, other):
        if self.shape() != other.shape():
            raise ValueError("Matrix dimensions must match for subtraction")
        return Matrix([
            [self.data[i][j] - other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])
 
    def __mul__(self, scalar):
        return Matrix([[scalar * v for v in row] for row in self.data])
 
    __rmul__ = __mul__
 
    def __matmul__(self, other):
        if isinstance(other, Matrix):
            if self.cols != other.rows:
                raise ValueError(f"Cannot multiply ({self.rows}x{self.cols}) @ ({other.rows}x{other.cols})")
            return Matrix([
                [sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
                 for j in range(other.cols)]
                for i in range(self.rows)
            ])
 
    def __pow__(self, n):
        if self.rows != self.cols:
            raise ValueError("Matrix must be square for exponentiation")
        result = Matrix.identity(self.rows)
        base = self.copy()
        n = int(n)
        while n > 0:
            if n % 2 == 1:
                result = result @ base
            base = base @ base
            n //= 2
        return result
 
    @property
    def T(self):
        return Matrix([
            [self.data[j][i] for j in range(self.rows)]
            for i in range(self.cols)
        ])
 
    def trace(self):
        if self.rows != self.cols:
            raise ValueError("Trace requires a square matrix")
        return sum(self.data[i][i] for i in range(self.rows))
 
    def det(self):
        if self.rows != self.cols:
            raise ValueError("Determinant requires a square matrix")
        if self.rows == 1:
            return self.data[0][0]
        if self.rows == 2:
            a, b = self.data[0]
            c, d = self.data[1]
            return a*d - b*c
        total = 0
        for col in range(self.cols):
            sub = [row[:col] + row[col+1:] for row in self.data[1:]]
            total += ((-1)**col) * self.data[0][col] * Matrix(sub).det()
        return total
 
    def minor(self, r, c):
        sub = [row[:c] + row[c+1:] for i, row in enumerate(self.data) if i != r]
        return Matrix(sub).det()
 
    def cofactors(self):
        return Matrix([
            [((-1)**(i+j)) * self.minor(i, j) for j in range(self.cols)]
            for i in range(self.rows)
        ])
 
    def adj(self):
        return self.cofactors().T
 
    def inv(self):
        d = self.det()
        if d == 0:
            raise ValueError("Matrix is singular (not invertible)")
        return (1/d) * self.adj()
 
    def rank(self):
        m = [row[:] for row in self.data]
        rank = 0
        for c in range(self.cols):
            pivot = next((r for r in range(rank, self.rows) if m[r][c] != 0), None)
            if pivot is None:
                continue
            m[rank], m[pivot] = m[pivot], m[rank]
            pv = m[rank][c]
            m[rank] = [v / pv for v in m[rank]]
            for r in range(self.rows):
                if r != rank:
                    f = m[r][c]
                    m[r] = [iv - f*rv for rv, iv in zip(m[rank], m[r])]
            rank += 1
        return rank
 
    def rref(self):
        m = [row[:] for row in self.data]
        lead = 0
        for r in range(self.rows):
            if lead >= self.cols:
                break
            i = r
            while m[i][lead] == 0:
                i += 1
                if i == self.rows:
                    i = r
                    lead += 1
                    if lead == self.cols:
                        return Matrix(m)
            m[i], m[r] = m[r], m[i]
            lv = m[r][lead]
            m[r] = [val/lv for val in m[r]]
            for i in range(self.rows):
                if i != r:
                    lv = m[i][lead]
                    m[i] = [iv - lv*rv for rv, iv in zip(m[r], m[i])]
            lead += 1
        return Matrix(m)
 
    def solve(self, b):
        if isinstance(b, Matrix):
            b_data = [row[0] if len(row) == 1 else row for row in b.data]
        else:
            b_data = b
        aug = [self.data[i] + [b_data[i]] for i in range(self.rows)]
        rref = Matrix(aug).rref().data
        return [row[-1] for row in rref]
 
    @staticmethod
    def identity(n):
        return Matrix([[1 if i == j else 0 for j in range(n)] for i in range(n)])
 
    @staticmethod
    def zeros(r, c):
        return Matrix([[0]*c for _ in range(r)])
 
 
# ---------------------------------------------------------------------------
# Lim  — fixed signature: Lim(expr, var, point)
# ---------------------------------------------------------------------------
 
class Lim:
    """
    Compute symbolic limits.
 
    Usage:
        Lim(sin(x)/x, x, 0)          -> 1
        Lim(Func(sin(x)/x), x, 0)    -> 1
    """
    def __init__(self, expr, var=None, point=None):
        # Resolve Func wrappers
        if isinstance(expr, Func):
            expr = expr.expr
 
        self.expr = expr
        self.var = var if var is not None else x
        self.point = point
 
    def __rshift__(self, target):
        """Support Lim(expr, x) >> 0  syntax."""
        return Lim(self.expr, self.var, target)
 
    def evaluate(self):
        if self.point is None:
            raise ValueError("Limit point not set.")
        return sp.limit(self.expr, self.var, self.point)
 
    def __repr__(self):
        if self.point is not None:
            return str(self.evaluate())
        return f"Lim({self.expr}, {self.var}, ?)"
 
    def __str__(self):
        return self.__repr__()
 
 
# ---------------------------------------------------------------------------
# Func
# ---------------------------------------------------------------------------
 
class Func:
    """
    A symbolic function wrapper using SymPy.
 
    f = Func(x**2 - 1)
    f(3)            -> 8
    f[1]            -> first derivative as Func
    f[-1]           -> indefinite integral as Func
    f[[0, 1]]       -> definite integral from 0 to 1
    f @ g           -> composition f(g(x))
    f.plot()        -> plot (handled by backend)
    f.derivative(n) -> nth derivative
    f.integrate(n)  -> nth antiderivative
    f.series(n)     -> Taylor series to order n
    f.inverse()     -> symbolic inverse (first branch)
    f >> sym        -> solve f(x) = 0 for sym
    """
    def __init__(self, expression):
        if callable(expression) and not isinstance(expression, sp.Basic):
            raise ValueError("Use symbolic SymPy expressions only, not callables.")
        self.expr = sp.sympify(expression)
        self._vars = sorted(self.expr.free_symbols, key=lambda s: s.name)
 
    def _lambdify(self):
        """Build a numeric evaluator — prefers numpy for robustness."""
        return sp.lambdify(self._vars if self._vars else [x], self.expr, modules=["numpy", "sympy"])
 
    def __call__(self, val):
        return self._lambdify()(val)
 
    def inverse(self):
        y = sp.Symbol('_y')
        sol = sp.solve(self.expr - y, x)
        if not sol:
            raise ValueError("Inverse not found or not unique.")
        return Func(sol[0].subs(y, x))
 
    def solve(self, sym=None):
        sym = sym or x
        return sp.solve(self.expr, sym)
 
    def __rshift__(self, sym):
        if isinstance(sym, sp.Symbol):
            return sp.solve(self.expr, sym)
        raise TypeError("Right-shift argument must be a sympy.Symbol")
 
    def __add__(self, other):
        return Func(self.expr + (other.expr if isinstance(other, Func) else other))
 
    def __radd__(self, other):
        return self + other
 
    def __sub__(self, other):
        return Func(self.expr - (other.expr if isinstance(other, Func) else other))
 
    def __rsub__(self, other):
        return Func(other - self.expr)
 
    def __mul__(self, other):
        return Func(self.expr * (other.expr if isinstance(other, Func) else other))
 
    def __rmul__(self, other):
        return self * other
 
    def __truediv__(self, other):
        return Func(self.expr / (other.expr if isinstance(other, Func) else other))
 
    def __rtruediv__(self, other):
        return Func(other / self.expr)
 
    def __matmul__(self, other):
        if not isinstance(other, Func):
            raise TypeError(f"Cannot compose Func and {type(other)}")
        return Func(self.expr.subs(x, other.expr))
 
    def __pow__(self, n):
        if not isinstance(n, int):
            raise TypeError("Only integer powers are supported.")
        if n == 0:
            return Func(sp.S.One)
        base = self if n > 0 else self.inverse()
        result = base
        for _ in range(abs(n) - 1):
            result = result @ base
        return result
 
    def __eq__(self, other):
        if isinstance(other, Func):
            return sp.simplify(self.expr - other.expr) == 0
        return NotImplemented
 
    def __str__(self):
        return str(self.expr)
 
    def __repr__(self):
        return str(self.expr)
 
    def simplify(self):
        return Func(sp.simplify(self.expr))
 
    def derivative(self, n=1):
        return Func(sp.diff(self.expr, x, n))
 
    def integrate(self, n=1):
        result = self
        for _ in range(n):
            result = Func(sp.integrate(result.expr, x))
        return result
 
    def partial(self, var, n=1):
        if not isinstance(var, sp.Symbol):
            raise TypeError("Variable must be a sympy.Symbol")
        return Func(sp.diff(self.expr, var, n))
 
    def gradient(self, variables=None):
        variables = variables or self._vars
        return [self.partial(v) for v in variables]
 
    def hessian(self, *variables):
        variables = variables or self._vars
        n = len(variables)
        return [[self.partial(variables[i]).partial(variables[j]) for j in range(n)] for i in range(n)]
 
    def series(self, n=6):
        return sp.series(self.expr, x, n=n)
 
    def __getitem__(self, key):
        if isinstance(key, int):
            if key >= 0:
                return Func(sp.diff(self.expr, x, key))
            # Negative: nth antiderivative
            result = self
            for _ in range(abs(key)):
                result = Func(sp.integrate(result.expr, x))
            return result
        if isinstance(key, (list, tuple)) and len(key) == 2:
            lower, upper = key
            return sp.integrate(self.expr, (x, lower, upper))
        raise TypeError("Key must be an int (derivative order) or [lower, upper] (definite integral).")




"""
import ast
import operator as op
import sympy as sp
from types import SimpleNamespace

# --- Import your backend classes ---
# from matlang_backend import Vector2, Vector3, Matrix, Quaternion, Func, Lim, x

# For demonstration, we'll assume the classes are already defined in this environment

# --- Supported operators mapping ---
OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.MatMult: op.matmul,
    ast.Pow: op.pow,
    ast.Div: op.truediv,
    ast.Mod: op.mod,
}

# --- Safe evaluation namespace ---
SAFE_GLOBALS = {
    "Vector2": Vector2,
    "Vector3": Vector3,
    "Matrix": Matrix,
    "Quaternion": Quaternion,
    "Func": Func,
    "Lim": Lim,
    "x": sp.Symbol("x"),
    "pi": sp.pi,
}

# --- Evaluator ---
class MatLangEvaluator(ast.NodeVisitor):
    def __init__(self, local_ns=None):
        self.vars = {} if local_ns is None else local_ns

    def visit_Module(self, node):
        results = []
        for stmt in node.body:
            results.append(self.visit(stmt))
        return results[-1] if results else None

    def visit_Expr(self, node):
        return self.visit(node.value)

    def visit_Assign(self, node):
        value = self.visit(node.value)
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.vars[target.id] = value
            else:
                raise ValueError(f"Unsupported assignment target: {target}")
        return value

    def visit_Name(self, node):
        if node.id in self.vars:
            return self.vars[node.id]
        elif node.id in SAFE_GLOBALS:
            return SAFE_GLOBALS[node.id]
        else:
            raise NameError(f"Unknown variable: {node.id}")

    def visit_Call(self, node):
        func = self.visit(node.func)
        args = [self.visit(arg) for arg in node.args]
        kwargs = {kw.arg: self.visit(kw.value) for kw in node.keywords}
        return func(*args, **kwargs)

    def visit_Attribute(self, node):
        value = self.visit(node.value)
        return getattr(value, node.attr)

    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        op_type = type(node.op)
        if op_type in OPERATORS:
            return OPERATORS[op_type](left, right)
        else:
            raise TypeError(f"Unsupported operator: {op_type}")

    def visit_MatMult(self, node):
        return op.matmul(self.visit(node.left), self.visit(node.right))

    def visit_Subscript(self, node):
        val = self.visit(node.value)
        idx = self.visit(node.slice)
        return val[idx]

    def visit_Index(self, node):
        return self.visit(node.value)

    def visit_Num(self, node):
        return node.n

    def visit_List(self, node):
        return [self.visit(el) for el in node.elts]

    def visit_Tuple(self, node):
        return tuple(self.visit(el) for el in node.elts)

    def visit_UnaryOp(self, node):
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            return -operand
        else:
            raise TypeError(f"Unsupported unary op: {node.op}")

    def generic_visit(self, node):
        raise TypeError(f"Unsupported AST node: {type(node)}")

# --- Parser function ---
def parse_matlang_code(code, local_ns=None):
    tree = ast.parse(code)
    evaluator = MatLangEvaluator(local_ns)
    return evaluator.visit(tree)

# --- Example usage ---
if __name__ == "__main__":
    # Suppose your file is "example.mlg"

    filename = "matlang.mlg"

    # Read the full content
    with open(filename, "r") as f:
        code = f.read()

        # Prepare a local namespace dictionary
        local_vars = {}


    # Access variables from the file
    for name, val in local_vars.items():
        local_vars = {}
        result = parse_matlang_code(code, local_vars)
        
        for var, value in local_vars:
            print(f'{var} = {value}')
"""