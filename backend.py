from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must be set before importing pyplot
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import sympy as sp

from matlang import Vector2, Vector3, Quaternion, Matrix, Func, Lim, x

# ---------------------------------------------------------------------------
# Per-session variable store (module-level; fine for single-user / dev use)
# ---------------------------------------------------------------------------
variables: dict = {}

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="MatLang CAS API",
    description="A computer algebra system with symbolic math, vector/matrix ops, and plotting.",
    version="1.0.0-beta-2",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

v1beta1 = APIRouter(prefix="/v1beta1", tags=["v1beta1"])


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------
class Command(BaseModel):
    code: str = Field(
        ...,
        example="Func(x**2 - 1).plot()",
        description="Mathematical expression or MatLang command to evaluate.",
    )


# ---------------------------------------------------------------------------
# Allowed globals — everything the user can reference
# ---------------------------------------------------------------------------
def _build_globals() -> dict:
    g: dict = {
        # --- MatLang types ---
        "Vector2": Vector2, "vector2": Vector2, "vec2": Vector2, "v2d": Vector2,
        "Vector3": Vector3, "vector3": Vector3, "vec3": Vector3, "v3d": Vector3,
        "Quaternion": Quaternion,
        "Matrix": Matrix, "matrix": Matrix,
        "Func": Func, "Function": Func, "function": Func, "func": Func,
        "Lim": Lim, "lim": Lim,

        # --- Symbols ---
        "x": x,
        "var": sp.Symbol,
        "symbols": sp.symbols,

        # --- Constants ---
        "pi": sp.pi,
        "e": sp.E,
        "oo": sp.oo,
        "inf": sp.oo,

        # --- Trig ---
        "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
        "sec": sp.sec, "csc": sp.csc, "cot": sp.cot,

        # --- Inverse trig ---
        "asin": sp.asin, "acos": sp.acos, "atan": sp.atan, "atan2": sp.atan2,
        "asec": sp.asec, "acsc": sp.acsc, "acot": sp.acot,

        # --- Hyperbolic ---
        "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
        "sech": sp.sech, "csch": sp.csch, "coth": sp.coth,

        # --- Inverse hyperbolic ---
        "asinh": sp.asinh, "acosh": sp.acosh, "atanh": sp.atanh,

        # --- Exp / log ---
        "exp": sp.exp,
        "log": sp.log, "ln": sp.log,
        "log10": lambda v: sp.log(v, 10),
        "log2":  lambda v: sp.log(v, 2),

        # --- Roots ---
        "sqrt": sp.sqrt,
        "cbrt": lambda v: sp.Rational(1, 3).__rpow__(v),  # v**(1/3) symbolically
        "root": lambda v, n: v ** sp.Rational(1, n),

        # --- Utilities ---
        "abs": sp.Abs,
        "floor": sp.floor,
        "ceiling": sp.ceiling,
        "sign": sp.sign,
        "float": float,
        "simplify": sp.simplify,
        "expand": sp.expand,
        "factor": sp.factor,
        "cancel": sp.cancel,
        "apart": sp.apart,
        "together": sp.together,
        "solve": sp.solve,
        "diff": sp.diff,
        "integrate": sp.integrate,
        "limit": sp.limit,
        "series": sp.series,
        "summation": sp.summation,

        # --- Special ---
        "factorial": lambda v: sp.factorial(v),
        "gamma": sp.gamma,
        "binomial": sp.binomial,
        "gcd": sp.gcd,
        "lcm": sp.lcm,
    }
    return g


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_assignment(code: str) -> bool:
    """Return True if code looks like  name = expr  (not ==, !=, <=, >=)."""
    import re
    return bool(re.match(r"^\s*[A-Za-z_][A-Za-z0-9_]*\s*=(?!=)", code))


def _render_plot(func_obj, x_range=(-10, 10), points=600) -> dict:
    """Evaluate a Func over x_range, build a matplotlib figure, return base64 PNG."""
    x_vals = np.linspace(x_range[0], x_range[1], points)
    y_vals = []
    for v in x_vals:
        try:
            y = func_obj(v)
            y = complex(y).real  # handle complex results gracefully
            y_vals.append(float(y) if np.isfinite(y) else np.nan)
        except Exception:
            y_vals.append(np.nan)
    y_vals = np.array(y_vals)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(x_vals, y_vals, color="#55ccff", linewidth=1.8)

    # Axis lines through origin
    ax.axhline(0, color="#555", linewidth=0.6)
    ax.axvline(0, color="#555", linewidth=0.6)

    # Auto y-range: ignore extreme outliers
    finite = y_vals[np.isfinite(y_vals)]
    if finite.size > 0:
        lo, hi = np.percentile(finite, 1), np.percentile(finite, 99)
        pad = max((hi - lo) * 0.15, 0.5)
        ax.set_ylim(lo - pad, hi + pad)

    ax.set_facecolor("#1a1a1a")
    ax.tick_params(colors="#aaa", labelsize=8)
    ax.grid(True, color="#2a2a2a", linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_color("#333")
    fig.patch.set_facecolor("#111")

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", facecolor="#111", dpi=130)
    plt.close(fig)
    buf.seek(0)
    return {"type": "plot", "image": base64.b64encode(buf.read()).decode("utf-8")}


def _safe_str(value) -> str:
    """Convert an eval result to a clean string."""
    if isinstance(value, sp.Basic):
        return str(value)
    return str(value)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", tags=["health"])
def health():
    return {"status": "ok", "message": "MatLang CAS API is running.", "version": "1.0.0-beta-2"}


@v1beta1.post(
    "/eval",
    summary="Evaluate a MatLang expression",
    description=(
        "Executes a MatLang / SymPy expression and returns the result as text or a base64 PNG plot. "
        "Use `Func(expr).plot()` to generate a plot. "
        "Assignments (`name = expr`) persist for the session."
    ),
)
def evaluate(command: Command):
    global variables
    code = command.code.strip()

    if not code:
        return {"type": "error", "result": "Empty input."}

    allowed = _build_globals()
    allowed.update(variables)  # inject session vars

    try:
        # ---- Plot shorthand: Func(...).plot() --------------------------------
        if code.endswith(".plot()"):
            func_code = code[: -len(".plot()")]
            func_obj = eval(func_code, {"__builtins__": {}}, allowed)
            if not isinstance(func_obj, Func):
                return {"type": "error", "result": "`.plot()` can only be called on a Func object."}
            return _render_plot(func_obj)

        # ---- Assignment: name = expr -----------------------------------------
        if _is_assignment(code):
            var_name, _, expr_str = code.partition("=")
            var_name = var_name.strip()
            expr_str = expr_str.strip()
            value = eval(expr_str, {"__builtins__": {}}, allowed)
            variables[var_name] = value
            return {"type": "text", "result": f"{var_name} = {_safe_str(value)}"}

        # ---- General expression ---------------------------------------------
        # Use eval (not parse_expr) so Func/Lim/Vector etc. all work correctly.
        result = eval(code, {"__builtins__": {}}, allowed)

        # If result is a Lim that hasn't been evaluated yet, evaluate it.
        if isinstance(result, Lim):
            result = result.evaluate()

        return {"type": "text", "result": _safe_str(result)}

    except ZeroDivisionError as e:
        return {"type": "error", "result": f"Math error: {e}"}
    except (SyntaxError, NameError, TypeError, ValueError) as e:
        return {"type": "error", "result": f"Error: {e}"}
    except Exception as e:
        return {"type": "error", "result": f"Unexpected error: {e}"}


@v1beta1.post("/reset", summary="Clear session variables", tags=["v1beta1"])
def reset():
    global variables
    variables = {}
    return {"type": "text", "result": "Session variables cleared."}


@v1beta1.get("/vars", summary="List session variables", tags=["v1beta1"])
def list_vars():
    return {"type": "vars", "result": {k: _safe_str(v) for k, v in variables.items()}}


app.include_router(v1beta1)