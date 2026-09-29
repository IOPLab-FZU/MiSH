"""Minor-Subspace Householder Reflection Adaptation (MiSH).

Novel lightweight PETL mechanism for text-image person re-identification.
MiSH = HRA (Householder Reflection Adaptation)
     + MSC (Minor-Subspace Confinement)
     + DGM (Directional Gain Modulation).

Reference: "Bridging The Gap between Low-rank and Orthogonal Adaptation via
Householder Reflection Adaptation" (HRA), NeurIPS 2024 (arXiv:2405.17484).

HRA keeps the pretrained Linear weight W frozen and composes r learnable
Householder reflections H_i = I - 2 u_i u_i^T (u_i unit vectors). MSC confines
these axes to the minor input subspace. DGM replaces 2 with a learnable beta_i;
the resulting transform need not be orthogonal, but retains MSC's protection
of the selected principal-subspace mapping at each adapted linear layer.

The forward pass folds the transformations into the frozen weight without
materialising the dense d x d transformation matrix:
    for j in 1..r:  W <- W - beta_j * (W u_j) u_j^T
    out = linear(x, W, frozen_bias)

Orthogonality regulariser (added to the loss):  || I_r - U^T U ||_F^2
with U = [u_1, ..., u_r] (columns L2-normalised).

The module is self-contained and toggleable: if no HRA is injected the model
behaves exactly like the original CLIP path.

MSC (Minor-Subspace Confinement): vanilla HRA reflections u_j are free in the
full input space, so they can still re-orient CLIP's high-gain (principal) input
directions -- the ones carrying pretrained image-text alignment. MSC projects
each reflection orthogonal to the frozen principal singular subspace span(V_p)
of the layer's weight, so H_j is exactly identity there and all adaptation is
confined to the low-energy minor subspace. Zero extra trainable params (V_p is a
frozen buffer). Setting subspace_energy=0 disables MSC; disabling DGM as well
recovers vanilla HRA.
"""
import torch
import torch.nn as nn


def _principal_input_basis(weight: torch.Tensor, energy: float) -> torch.Tensor:
    """Top right-singular vectors V_p of a frozen weight, by spectral energy.

    MSC (Minor-Subspace Confinement): a Linear maps inputs x via W x. Its right-singular
    vectors V are the orthonormal *input* directions; the singular values are
    the per-direction gains. The high-gain directions carry the bulk of CLIP's
    pretrained image-text computation, so we treat span(V_p) -- the smallest set
    of leading directions whose squared singular values cover `energy` of the
    total -- as the principal/alignment subspace to PROTECT (freeze).

    Returns V_p in R^{in x p} (fp32, unit-orthonormal columns). p adapts to each
    layer; for rectangular c_proj (in>out) only the <=out non-trivial directions
    can be selected, so the frozen null space is never protected (it has 0 gain).
    """
    W = weight.detach().float()                       # [out, in]
    # right-singular vectors: eigenvectors of W^T W (in x in); economical via SVD.
    # full_matrices=False -> V has min(out,in) columns (the non-trivial ones).
    _, S, Vh = torch.linalg.svd(W, full_matrices=False)
    s2 = S * S
    total = s2.sum()
    if total <= 0:
        return torch.zeros(weight.shape[1], 0)
    cum = torch.cumsum(s2, dim=0) / total
    # smallest p with cumulative energy >= threshold (at least 1 direction).
    p = int(torch.searchsorted(cum, torch.tensor(energy, dtype=cum.dtype)).item()) + 1
    p = max(1, min(p, Vh.shape[0]))
    return Vh[:p].t().contiguous()                    # [in, p]


class HouseholderLinear(nn.Module):
    """Wrap a frozen nn.Linear with r learnable Householder input reflections.

    Args:
        base_linear: the original (pretrained) nn.Linear to be frozen.
        r: number of Householder reflections (capacity vs. params trade-off).
        lam: orthogonality regularisation strength. When lam == 0, columns of U
             are L2-normalised on the fly (max capacity, no orthogonality
             constraint), matching the HRA paper's lambda=0 setting.
        subspace_energy: MSC principal-protection threshold in (0,1). When > 0,
             reflection vectors are projected orthogonal to the frozen principal
             input subspace span(V_p) (top right-singular directions covering this
             fraction of spectral energy), so every reflection H_j = I - 2 u u^T
             is IDENTITY on span(V_p) -> CLIP's dominant pretrained computation is
             left exactly untouched and adaptation lives in the minor subspace.
             0 (default) recovers vanilla HRA exactly.
        dgm: DGM (Directional Gain Modulation). When True, the fixed
             reflection coefficient 2 becomes a per-axis learnable scalar
             beta_j, so H_j = I - beta_j u_j u_j^T. Since u_j is confined to the
             minor subspace (MSC), every principal direction still satisfies
             u_j^T v = 0 -> H_j v = v regardless of beta_j, so span(V_p) stays
             exactly protected; only the minor-axis gain (eigenvalue 1-beta_j
             along u_j) becomes learnable, unlocking amplification/attenuation
             that pure orthogonal reflection (norm-locked, gain==1) cannot express.
             beta_j is initialised to 2.0, so DGM starts bit-identical to HRA+MSC
             (a strict superset). False (default) keeps the constant-2 reflection.
    """

    def __init__(self, base_linear: nn.Linear, r: int = 8, lam: float = 1e-3,
                 subspace_energy: float = 0.0, dgm: bool = False):
        super().__init__()
        assert isinstance(base_linear, nn.Linear)
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.r = r
        self.lam = lam
        self.subspace_energy = subspace_energy
        self.dgm = dgm

        # Keep the pretrained Linear, frozen. Registered as a submodule so its
        # buffers/params move with .to()/.half() but never receive grads.
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad_(False)

        # MSC: frozen principal input basis V_p [in, p]. Computed once from the
        # pretrained weight; non-persistent buffer -> recomputed identically at
        # load time from the (frozen) base weight, so it never bloats checkpoints
        # and stays consistent train<->test. Kept fp32 for a stable projection.
        if subspace_energy and subspace_energy > 0.0:
            Vp = _principal_input_basis(self.base.weight, subspace_energy)
            self.register_buffer('V_p', Vp, persistent=False)
        else:
            self.V_p = None

        self.hra_U = nn.Parameter(torch.randn(self.in_features, r) * 0.02)

        # DGM: per-axis learnable reflection coefficient beta_j (init 2.0 -> the
        # exact HRA reflection I - 2 u u^T). None keeps the constant-2 behaviour.
        if dgm:
            self.hra_beta = nn.Parameter(torch.full((r,), 2.0))
        else:
            self.hra_beta = None

    def _project_minor(self, U: torch.Tensor) -> torch.Tensor:
        """Remove the principal component of each reflection vector (MSC).

        u <- u - V_p (V_p^T u). Then u _|_ span(V_p), so H = I - 2 u u^T fixes
        every principal direction, confining all adaptation to the minor subspace.
        """
        if self.V_p is None:
            return U
        Vp = self.V_p.to(U.dtype)                      # [in, p]
        return U - Vp @ (Vp.t() @ U)                   # [in, r]

    def _get_U(self) -> torch.Tensor:
        """Reflection vectors U in R^{in x r} (fp32): vanilla HRA reflections,
        optionally confined to the minor singular subspace (MSC)."""
        return self._project_minor(self.hra_U)

    def _reflected_weight(self, dtype) -> torch.Tensor:
        """Fold the r Householder reflections into the frozen weight.

        Mathematically  W @ (H_1 ... H_r)  with  H_j = I - 2 u_j u_j^T.
        We reflect the *weight* (out x in, batch-independent) rather than the
        activation (seq x batch x in). This is identical in result but uses far
        less memory: autograd only retains r weight-sized intermediates instead
        of r large activation tensors, which is what caused OOM at higher r.

        Column-update form:  W @ H_j = W - beta_j (W u_j) u_j^T, where beta_j is
        the fixed constant 2 (pure reflection) or a learnable DGM coefficient.
        """
        U = self._get_U()
        U = U / (U.norm(dim=0, keepdim=True) + 1e-8)   # unit reflection vectors
        W = self.base.weight.float()                    # [out, in], frozen
        for j in range(self.r):
            u_j = U[:, j]                               # [in]
            Wu = W @ u_j                                # [out]
            beta_j = self.hra_beta[j] if self.hra_beta is not None else 2.0
            W = W - beta_j * torch.outer(Wu, u_j)       # [out, in]
        return W.to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self._reflected_weight(x.dtype)
        return torch.nn.functional.linear(x, W, self.base.bias)

    def ortho_reg(self) -> torch.Tensor:
        """|| I_r - U^T U ||_F^2 with L2-normalised columns."""
        U = self._get_U()
        U = U / (U.norm(dim=0, keepdim=True) + 1e-8)
        gram = U.t() @ U                       # [r, r]
        eye = torch.eye(self.r, device=U.device, dtype=U.dtype)
        return ((eye - gram) ** 2).sum()

    def beta_reg(self) -> torch.Tensor:
        """DGM coefficient regulariser: || beta - 2 ||^2.

        Pulls each learnable reflection coefficient back toward 2 (the pure
        HRA reflection), so beta only departs from 2 when the minor-axis gain
        genuinely helps retrieval. Returns a zero scalar when DGM is off.
        """
        if self.hra_beta is None:
            return torch.zeros((), device=self.hra_U.device, dtype=torch.float32)
        return ((self.hra_beta.float() - 2.0) ** 2).sum()


def inject_hra(clip_model: nn.Module, r: int = 8, lam: float = 1e-3,
               where=('mlp',), subspace_energy: float = 0.0,
               dgm: bool = False) -> int:
    """Wrap target Linear layers inside a CLIP model with HouseholderLinear.

    Targets ResidualAttentionBlock.mlp.c_fc / c_proj for 'mlp', and the
    attention out_proj for 'attn'. Returns the number of wrapped layers.

    Operates on both the vision (visual.transformer) and text (transformer)
    branches because both use ResidualAttentionBlock. subspace_energy>0 enables
    MSC (per-layer minor-subspace confinement of the reflections).
    """
    where = set(where)
    count = 0

    def wrap_linear(parent, attr):
        nonlocal count
        lin = getattr(parent, attr)
        if isinstance(lin, nn.Linear):
            setattr(parent, attr, HouseholderLinear(lin, r=r, lam=lam,
                                                    subspace_energy=subspace_energy,
                                                    dgm=dgm))
            count += 1

    for module in clip_model.modules():
        # ResidualAttentionBlock has .mlp (Sequential c_fc/gelu/c_proj) and .attn
        if hasattr(module, 'mlp') and isinstance(module.mlp, nn.Sequential):
            if 'mlp' in where:
                if hasattr(module.mlp, 'c_fc'):
                    wrap_linear(module.mlp, 'c_fc')
                if hasattr(module.mlp, 'c_proj'):
                    wrap_linear(module.mlp, 'c_proj')
            if 'attn' in where and hasattr(module, 'attn'):
                # NOTE: nn.MultiheadAttention consumes out_proj/in_proj as raw
                # weight tensors inside F.multi_head_attention_forward (it reads
                # .weight/.bias directly rather than calling the submodule), so a
                # module-wrapper like HouseholderLinear cannot intercept it.
                # Attention adaptation via wrapping is therefore unsupported;
                # we skip it explicitly instead of silently breaking MHA.
                print('[MiSH] WARNING: where="attn" is unsupported for '
                      'nn.MultiheadAttention (weights used functionally); '
                      'skipping attention layers. Use larger hra_r on mlp instead.')

    return count


def collect_ortho_reg(model: nn.Module) -> torch.Tensor:
    """Sum the orthogonality regulariser over all HouseholderLinear modules."""
    reg = None
    for m in model.modules():
        if isinstance(m, HouseholderLinear):
            r = m.ortho_reg()
            reg = r if reg is None else reg + r
    if reg is None:
        # no HRA injected; return a zero scalar on the model's device
        dev = next(model.parameters()).device
        return torch.zeros((), device=dev)
    return reg


def collect_beta_reg(model: nn.Module) -> torch.Tensor:
    """Sum the DGM coefficient regulariser over all HouseholderLinear modules."""
    reg = None
    for m in model.modules():
        if isinstance(m, HouseholderLinear):
            b = m.beta_reg()
            reg = b if reg is None else reg + b
    if reg is None:
        dev = next(model.parameters()).device
        return torch.zeros((), device=dev)
    return reg
