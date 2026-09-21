"""
NevoRL: Universal Vectorized Reinforcement Learning Domain-Specific Language.
Compiles scalar agent/environment descriptions into high-throughput, auto-resetting
Gymnasium-compatible VectorEnvs.
"""

from __future__ import annotations
import time
from typing import Any, Dict, List, Tuple
import numpy as np
from lark import Lark, Transformer

# =====================================================================
# 1. GRAMMAR DEFINITION
# =====================================================================
NEVORL_GRAMMAR = r"""
    start: env_def

    env_def: "env" CNAME "{" section* "}"

    ?section: state_sec
            | action_sec
            | obs_sec
            | reset_sec
            | step_sec
            | reward_sec
            | terminal_sec

    state_sec: "state" "{" field_decl* "}"
    field_decl: CNAME ":" TYPE ";"?

    action_sec: "action" action_type ";"?
    action_type: "discrete" "(" INT ")"       -> act_discrete
               | "continuous" "(" INT ")"     -> act_continuous

    obs_sec: "observation" "{" stmt* "return" expr ";"? "}"
    reset_sec: "reset" "{" stmt* "}"
    step_sec: "step" "(" CNAME ")" "{" stmt* "}"
    reward_sec: "reward" "{" stmt* "return" expr ";"? "}"
    terminal_sec: "terminal" "{" stmt* "return" expr ";"? "}"

    ?stmt: let_stmt
         | assign_stmt

    let_stmt: "let" CNAME "=" expr ";"?
    assign_stmt: target "=" expr ";"?

    target: "state" "." CNAME -> state_target
          | CNAME             -> var_target

    ?expr: or_expr
    ?or_expr: and_expr (("or" | "||") and_expr)*
    ?and_expr: comp_expr (("and" | "&&") comp_expr)*
    ?comp_expr: arith_expr ((COMP_OP) arith_expr)*
    ?arith_expr: term (("+" | "-") term)*
    ?term: factor (("*" | "/") factor)*
    ?factor: ("+" | "-")? atom
    ?atom: SIGNED_NUMBER                     -> num
         | "state" "." CNAME                 -> state_access
         | CNAME "[" expr "]"                -> array_index
         | CNAME "(" (expr ("," expr)*)? ")" -> func_call
         | CNAME                             -> var
         | "[" (expr ("," expr)*)? "]"       -> list_lit
         | "(" expr ")"

    COMP_OP: ">=" | "<=" | "==" | "!=" | ">" | "<"
    TYPE: "float" | "int" | "bool"

    %import common.CNAME
    %import common.INT
    %import common.SIGNED_NUMBER
    %import common.WS
    %ignore WS
    %ignore /\/\/[^\n]*/
"""

# =====================================================================
# 2. TRANSPILER / CODE GENERATOR
# =====================================================================
class NevoRLTranspiler(Transformer):
    def __init__(self):
        super().__init__()
        self.env_name = ""
        self.state_schema: Dict[str, str] = {}
        self.action_space: Dict[str, Any] = {}
        self.action_var = "act"
        self.in_reset_context = False

    def env_def(self, items):
        self.env_name = str(items[0])
        return self

    def field_decl(self, items):
        name, type_ = str(items[0]), str(items[1])
        self.state_schema[name] = type_
        return (name, type_)

    def state_sec(self, items):
        return ("state", self.state_schema)

    def act_discrete(self, items):
        self.action_space = {"type": "discrete", "n": int(items[0])}
        return self.action_space

    def act_continuous(self, items):
        self.action_space = {"type": "continuous", "dim": int(items[0])}
        return self.action_space

    def action_sec(self, items):
        return ("action", self.action_space)

    # Statements & Expressions
    def state_target(self, items):
        field = str(items[0])
        if self.in_reset_context:
            return f"state['{field}'][mask]"
        return f"state['{field}']"

    def var_target(self, items):
        return str(items[0])

    def let_stmt(self, items):
        return f"    {items[0]} = {items[1]}"

    def assign_stmt(self, items):
        return f"    {items[0]} = {items[1]}"

    def num(self, items):
        return str(items[0])

    def var(self, items):
        return str(items[0])

    def state_access(self, items):
        field = str(items[0])
        if self.in_reset_context:
            return f"state['{field}'][mask]"
        return f"state['{field}']"

    def array_index(self, items):
        target, idx = str(items[0]), str(items[1])
        if target == self.action_var:
            if self.action_space.get("type") == "continuous":
                return f"{target}[:, {idx}]"
        return f"{target}[{idx}]"

    def func_call(self, items):
        name = str(items[0])
        args = [str(a) for a in items[1:]]
        if name == "uniform":
            low, high = args[0], args[1]
            if self.in_reset_context:
                return f"np.random.uniform({low}, {high}, size=len(mask)).astype(np.float32)"
            return f"np.random.uniform({low}, {high}, size=num_envs).astype(np.float32)"
        elif name == "where":
            return f"np.where({args[0]}, {args[1]}, {args[2]})"
        elif name == "clamp":
            return f"np.clip({args[0]}, {args[1]}, {args[2]})"
        elif name in ("cos", "sin", "abs", "sqrt", "exp", "log"):
            return f"np.{name}({args[0]})"
        return f"np.{name}({', '.join(args)})"

    def list_lit(self, items):
        elements = [str(e) for e in items]
        return f"np.column_stack([{', '.join(elements)}]).astype(np.float32)"

    def or_expr(self, items):
        if len(items) == 1:
            return str(items[0])
        return "(" + " | ".join(str(it) for it in items if str(it) not in ("or", "||")) + ")"

    def and_expr(self, items):
        if len(items) == 1:
            return str(items[0])
        return "(" + " & ".join(str(it) for it in items if str(it) not in ("and", "&&")) + ")"

    def comp_expr(self, items):
        if len(items) == 1:
            return str(items[0])
        return f"({items[0]} {items[1]} {items[2]})"

    def arith_expr(self, items):
        if len(items) == 1:
            return str(items[0])
        return f"({items[0]} {items[1]} {items[2]})"

    def term(self, items):
        if len(items) == 1:
            return str(items[0])
        return f"({items[0]} {items[1]} {items[2]})"

    def factor(self, items):
        if len(items) == 1:
            return str(items[0])
        return f"({items[0]}{items[1]})"

    # Blocks
    def reset_sec(self, items):
        self.in_reset_context = True
        body = [str(s) for s in items]
        self.in_reset_context = False
        lines = ["def _reset_kernel(state, mask, num_envs):"] + body
        return ("reset_code", "\n".join(lines))

    def step_sec(self, items):
        self.action_var = str(items[0])
        body = [str(s) for s in items[1:]]
        lines = [f"def _step_kernel(state, {self.action_var}, num_envs):"] + body
        return ("step_code", "\n".join(lines))

    def reward_sec(self, items):
        body = [str(s) for s in items[:-1]]
        ret_expr = str(items[-1])
        lines = ["def _reward_kernel(state, num_envs):"] + body + [f"    return np.asarray({ret_expr}, dtype=np.float32)"]
        return ("reward_code", "\n".join(lines))

    def terminal_sec(self, items):
        body = [str(s) for s in items[:-1]]
        ret_expr = str(items[-1])
        lines = ["def _terminal_kernel(state, num_envs):"] + body + [f"    return np.asarray({ret_expr}, dtype=bool)"]
        return ("terminal_code", "\n".join(lines))

    def obs_sec(self, items):
        body = [str(s) for s in items[:-1]]
        ret_expr = str(items[-1])
        lines = ["def _obs_kernel(state, num_envs):"] + body + [f"    return {ret_expr}"]
        return ("obs_code", "\n".join(lines))


# =====================================================================
# 3. VECTORIZED ENVIRONMENT BASE CLASS (Gymnasium Compatible)
# =====================================================================
class CompiledVectorEnv:
    """
    Standard Vectorized RL Environment generated from NevoRL.
    Supports auto-reset, info dictionaries, and high-performance batched steps.
    """
    def __init__(self, num_envs: int, schema: Dict[str, str], act_space: Dict[str, Any], kernels: Dict[str, Any]):
        self.num_envs = num_envs
        self.schema = schema
        self.action_space_info = act_space
        self.kernels = kernels

        # Contiguous Struct-of-Arrays (SoA) memory
        self.state: Dict[str, np.ndarray] = {}
        for name, dtype_str in self.schema.items():
            dt = np.float32 if dtype_str == "float" else (np.int32 if dtype_str == "int" else bool)
            self.state[name] = np.zeros(self.num_envs, dtype=dt)

        self._reset_all()

    def _reset_all(self):
        all_mask = np.arange(self.num_envs, dtype=np.int32)
        self.kernels["_reset_kernel"](self.state, all_mask, self.num_envs)

    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        self._reset_all()
        obs = self.kernels["_obs_kernel"](self.state, self.num_envs)
        return obs, {}

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        # 1. Physics / Dynamics Step
        self.kernels["_step_kernel"](self.state, actions, self.num_envs)

        # 2. Compute Rewards and Terminations
        rewards = self.kernels["_reward_kernel"](self.state, self.num_envs)
        if rewards.ndim == 0 or (rewards.ndim == 1 and len(rewards) == 1):
            rewards = np.full(self.num_envs, rewards, dtype=np.float32)

        terminated = self.kernels["_terminal_kernel"](self.state, self.num_envs)
        truncated = np.zeros(self.num_envs, dtype=bool)

        # 3. Compute Current (Next) Observations
        obs = self.kernels["_obs_kernel"](self.state, self.num_envs)

        # 4. Standard Gym Auto-Reset
        dones = np.where(terminated | truncated)[0]
        infos: Dict[str, Any] = {}

        if len(dones) > 0:
            infos["final_observation"] = obs[dones].copy()
            infos["_final_observation"] = dones

            # Reset only terminated environments in-place
            self.kernels["_reset_kernel"](self.state, dones, self.num_envs)

            # Replace observation of reset environments with fresh initial observation
            reset_obs = self.kernels["_obs_kernel"](self.state, self.num_envs)
            obs[dones] = reset_obs[dones]

        return obs, rewards, terminated, truncated, infos

    def sample_actions(self) -> np.ndarray:
        if self.action_space_info["type"] == "discrete":
            return np.random.randint(0, self.action_space_info["n"], size=self.num_envs)
        else:
            dim = self.action_space_info["dim"]
            return np.random.uniform(-1.0, 1.0, size=(self.num_envs, dim)).astype(np.float32)


# =====================================================================
# 4. COMPILER ENGINE
# =====================================================================
class NevoRLCompiler:
    def __init__(self):
        self.parser = Lark(NEVORL_GRAMMAR, parser="lalr")

    def compile_source(self, source_code: str) -> type:
        tree = self.parser.parse(source_code)
        transpiler = NevoRLTranspiler()
        transpiler.transform(tree)

        # Compile extracted kernels into executable Python functions
        exec_env = {"np": np}
        for item in tree.children[0].children:
            if isinstance(item, tuple) and item[0].endswith("_code"):
                exec(item[1], exec_env)

        schema = transpiler.state_schema
        act_space = transpiler.action_space

        class GeneratedEnv(CompiledVectorEnv):
            def __init__(self, num_envs: int = 64):
                super().__init__(num_envs=num_envs, schema=schema, act_space=act_space, kernels=exec_env)

        GeneratedEnv.__name__ = transpiler.env_name
        return GeneratedEnv
