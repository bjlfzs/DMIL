import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
from torch.nn.parallel import DataParallel


class CustomDataParallel(DataParallel):
    """Custom DataParallel that handles dict-returning models and correctly aggregates scalar values across GPUs."""

    def gather(self, outputs, output_device):
        return self._gather_dict_outputs(outputs, output_device)

    def _gather_dict_outputs(self, outputs, output_device):
        """Recursively process dict and nested structures, ensuring scalars are correctly handled."""
        if isinstance(outputs[0], torch.Tensor):
            if all(o.dim() == 0 for o in outputs):
                return torch.stack([o.to(output_device).unsqueeze(0) for o in outputs])
            return nn.parallel.gather(outputs, output_device, dim=self.dim)
        elif isinstance(outputs[0], collections.abc.Mapping):
            return {k: self._gather_dict_outputs([o[k] for o in outputs], output_device)
                    for k in outputs[0]}
        elif isinstance(outputs[0], (tuple, list)):
            result = [self._gather_dict_outputs([o[i] for o in outputs], output_device)
                      for i in range(len(outputs[0]))]
            return type(outputs[0])(result)
        return outputs[0]


def _b(x):
    """Ensure tensor has at least one dimension (required for DataParallel gather)."""
    return x.unsqueeze(0) if isinstance(x, torch.Tensor) and x.dim() == 0 else x


def _recon_loss(x, x_hat):
    """Normalised squared Frobenius reconstruction loss."""
    return _b(torch.norm(x - x_hat) ** 2 / (x.size(0) * x.size(1)))


def diagonal_gaussian_kl(q_mean, q_log_var, p_mean=0.0, p_log_var=0.0):
    """Mean KL divergence ``KL(q || p)`` for diagonal Gaussians.

    ``q_log_var`` and ``p_log_var`` are logarithms of variances, not standard
    deviations. Scalar prior parameters are broadcast by PyTorch, so the
    standard-normal prior is represented by ``p_mean=0`` and ``p_log_var=0``.
    """
    p_mean = torch.as_tensor(p_mean, dtype=q_mean.dtype, device=q_mean.device)
    p_log_var = torch.as_tensor(
        p_log_var, dtype=q_log_var.dtype, device=q_log_var.device
    )
    log_var_delta = q_log_var - p_log_var
    variance_ratio = torch.exp(log_var_delta)
    squared_mean_delta = (q_mean - p_mean).pow(2) * torch.exp(-p_log_var)
    return 0.5 * torch.mean(
        -log_var_delta + variance_ratio + squared_mean_delta - 1
    )


class VariationalBottleneck(nn.Module):
    """Variational Information Bottleneck layer."""

    def __init__(self, input_size=512, embed_size=512):
        super().__init__()
        self.embed_size = embed_size
        self.fc = nn.Linear(input_size, embed_size * 2)

    def forward(self, inputs):
        z = self.fc(inputs)
        return z[:, :self.embed_size], z[:, self.embed_size:]


def normal_reshape(v, a, B=None):
    if B is None:
        B = a.size(0)
    _, C, H, W = v.size()
    v = v.view(B, -1, C, H, W).permute(0, 2, 1, 3, 4)
    _, C, H, W = a.size()
    a = a.view(B, -1, C, H, W).permute(0, 2, 1, 3, 4)
    v = torch.flatten(F.adaptive_avg_pool3d(v, 1), 1)
    a = torch.flatten(F.adaptive_avg_pool3d(a, 1), 1)
    return v, a


class CrossModality(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.fc_out = nn.Linear(embed_size * 3, embed_size)

    def forward(self, x1, x2):
        return self.fc_out(torch.cat((x1, x2, x1 * x2), dim=1))


class DMILModel(nn.Module):
    def __init__(self, video_model, audio_model, embed_size=512, num_class=101,
                 method=None, stage=1, input_size=512):
        super().__init__()
        self.audio_net = audio_model
        self.visual_net = video_model
        self.embed_size = embed_size
        self.stage = stage

        self.embed_v = nn.Linear(input_size, embed_size)
        self.embed_a = nn.Linear(input_size, embed_size)

        # First-level decomposition: task-relevant (z_tr) and task-irrelevant (z_ir)
        self.IB_m1_tr = VariationalBottleneck(input_size=embed_size, embed_size=embed_size)
        self.IB_m1_ir = VariationalBottleneck(input_size=embed_size, embed_size=embed_size)
        self.IB_m2_tr = VariationalBottleneck(input_size=embed_size, embed_size=embed_size)
        self.IB_m2_ir = VariationalBottleneck(input_size=embed_size, embed_size=embed_size)

        # Second-level decomposition: redundancy, uniqueness, synergy
        self.IB_redundancy = VariationalBottleneck(input_size=embed_size * 2, embed_size=embed_size)
        self.IB_m1_r = VariationalBottleneck(input_size=embed_size, embed_size=embed_size)
        self.IB_m2_r = VariationalBottleneck(input_size=embed_size, embed_size=embed_size)
        self.IB_m1_unique = VariationalBottleneck(input_size=embed_size, embed_size=embed_size)
        self.IB_m2_unique = VariationalBottleneck(input_size=embed_size, embed_size=embed_size)

        self.cross_modality = CrossModality(embed_size)

        # Reconstruction networks
        def _recon_net():
            return nn.Sequential(
                nn.Linear(2 * embed_size, embed_size // 2),
                nn.ReLU(),
                nn.Linear(embed_size // 2, embed_size)
            )
        self.recon_1 = _recon_net()
        self.recon_2 = _recon_net()
        self.recon_tr_1 = _recon_net()
        self.recon_tr_2 = _recon_net()

        # First-level decomposition classifiers
        self.tr1_fc = nn.Linear(embed_size, num_class)
        self.tr2_fc = nn.Linear(embed_size, num_class)

        # Second-level decomposition classifiers
        self.red_fc = nn.Linear(embed_size, num_class)
        self.unique1_fc = nn.Linear(embed_size, num_class)
        self.unique2_fc = nn.Linear(embed_size, num_class)
        self.synergy_fc = nn.Linear(embed_size, num_class)

        # Gating network for dynamic fusion of redundancy, uniqueness, and synergy
        self.gating_network = nn.Sequential(
            nn.Linear(4 * embed_size, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, 4),
            nn.Softmax(dim=1)
        )

        print(f"Using {method.method_name} Fusion with Two-Stage Decomposition and Gating Mechanism!!")

    def configure_stage(self, stage):
        """Select a training stage and apply the freezing required by the paper.

        Stage 2 keeps the encoders and intra-modality decomposition fixed while
        the consistency decomposition and synergy modules are learned. Setting
        those modules to evaluation mode also freezes BatchNorm running stats,
        which ``torch.no_grad`` alone does not do.
        """
        if stage not in (1, 2, 3):
            raise ValueError(f'Stage {stage} not defined. Only 1, 2, 3 are supported.')

        self.stage = stage
        stage1_modules = (
            self.visual_net, self.audio_net, self.embed_v, self.embed_a,
            self.IB_m1_tr, self.IB_m1_ir, self.IB_m2_tr, self.IB_m2_ir,
            self.recon_1, self.recon_2, self.tr1_fc, self.tr2_fc,
        )
        stage1_trainable = stage != 2
        for module in stage1_modules:
            for parameter in module.parameters():
                parameter.requires_grad_(stage1_trainable)
            if not stage1_trainable:
                module.eval()

    def get_loss(self, z_mean, z_log_var, r_mean, r_log_var):
        """Sample from z and return its unweighted KL divergence to r.

        Loss weighting intentionally lives only in ``DecompositionTrainer`` so
        that ``cfg.methods.lamb`` is the single source of truth.
        """
        kl_loss = diagonal_gaussian_kl(
            z_mean, z_log_var, r_mean, r_log_var
        )
        kl_loss = _b(kl_loss)
        if self.training:
            u = torch.randn_like(z_mean)
            return z_mean + torch.exp(z_log_var / 2) * u, kl_loss
        return z_mean, kl_loss

    def _stage1_vib(self, feat, IB_tr, IB_ir, recon_net, clf_fc):
        """Stage 1 VIB for one modality: sample z_tr and z_ir, reconstruct input, classify z_tr."""
        z_tr_mu, z_tr_logvar = IB_tr(feat)
        z_ir_mu, z_ir_logvar = IB_ir(feat)
        z_tr, loss_kl_tr = self.get_loss(z_tr_mu, z_tr_logvar, 0.0, 0.0)
        z_ir, loss_kl_ir = self.get_loss(z_ir_mu, z_ir_logvar, 0.0, 0.0)
        rec = recon_net(torch.cat((z_tr, z_ir), dim=1))
        return clf_fc(z_tr), loss_kl_tr + loss_kl_ir, _recon_loss(feat, rec)

    def _second_level_decompose(self, z_tr_1_mu, z_tr_2_mu, z_ir_1, z_ir_2):
        """
        Second-level decomposition: redundancy, uniqueness, synergy, and dynamic gating.
        Shared by Stage 2 (frozen Stage-1 inputs) and Stage 3 (joint fine-tuning).
        """
        # Redundancy
        z_red_mu, z_red_logvar = self.IB_redundancy(torch.cat((z_tr_1_mu, z_tr_2_mu), dim=1))
        z_red, loss_red_ib = self.get_loss(z_red_mu, z_red_logvar, 0.0, 0.0)

        # Uniqueness
        z_unique_1_mu, z_unique_1_logvar = self.IB_m1_unique(z_tr_1_mu)
        z_unique_2_mu, z_unique_2_logvar = self.IB_m2_unique(z_tr_2_mu)
        # Compactness terms from Appendix A.3: KL(q(U^m|M^m) || N(0, I)).
        # Together with reconstruction from (R, U^m), these terms implement
        # the variational R/U decomposition rather than a standalone U loss.
        z_unique_1, loss_uni_1 = self.get_loss(z_unique_1_mu, z_unique_1_logvar, 0.0, 0.0)
        z_unique_2, loss_uni_2 = self.get_loss(z_unique_2_mu, z_unique_2_logvar, 0.0, 0.0)

        # Appendix A.3 alignment/conditional-MI upper bound. The joint
        # q(R|M1,M2) is aligned with each single-modality v_phi(R|M^m).
        z_red_1_mu, z_red_1_logvar = self.IB_m1_r(z_tr_1_mu)
        z_red_2_mu, z_red_2_logvar = self.IB_m2_r(z_tr_2_mu)
        _, loss_inter_1 = self.get_loss(z_red_mu, z_red_logvar, z_red_1_mu, z_red_1_logvar)
        _, loss_inter_2 = self.get_loss(z_red_mu, z_red_logvar, z_red_2_mu, z_red_2_logvar)

        # z_tr reconstruction
        rec_tr_1 = self.recon_tr_1(torch.cat((z_red, z_unique_1), dim=1))
        rec_tr_2 = self.recon_tr_2(torch.cat((z_red, z_unique_2), dim=1))

        # Synergy via cross-modal interaction
        z_synergy = self.cross_modality(z_ir_1, z_ir_2)

        # Classifier outputs
        out_red = self.red_fc(z_red)
        out_unique_1 = self.unique1_fc(z_unique_1)
        out_unique_2 = self.unique2_fc(z_unique_2)
        out_synergy = self.synergy_fc(z_synergy)

        # Dynamic gating — weighted fusion of all components
        gate_weights = self.gating_network(torch.cat((z_unique_1, z_unique_2, z_red, z_synergy), dim=1))
        out_mm = (gate_weights[:, 0:1] * out_unique_1 +
                  gate_weights[:, 1:2] * out_unique_2 +
                  gate_weights[:, 2:3] * out_red +
                  gate_weights[:, 3:4] * out_synergy)

        return {
            "out": out_mm,
            "out_synergy": out_synergy,
            "out_red": out_red,
            "out_unique": (out_unique_1, out_unique_2),
            "loss_inter": (loss_inter_1, loss_inter_2),
            "loss_rec_tr": (_recon_loss(z_tr_1_mu, rec_tr_1), _recon_loss(z_tr_2_mu, rec_tr_2)),
            "loss_uni": (loss_uni_1, loss_uni_2),
            "loss_red": loss_red_ib,
            "gate_weights": gate_weights,
        }

    def forward(self, visual, audio, train_modality='both'):
        """
        Args:
            visual: Visual input tensor.
            audio: Audio input tensor.
            train_modality: Defaults to both modalities, as required by the
                paper's Stage-1 objective. Single-modality values are retained
                for backwards compatibility; Stage 2/3 ignore this argument.
        """
        batch_size = visual.size(0)
        v = self.visual_net(visual)
        a = self.audio_net(audio)
        v, a = normal_reshape(v, a, B=batch_size)
        v = self.embed_v(v)
        a = self.embed_a(a)

        if self.stage == 1:
            return self.train_stage_1_vib(v, a, train_modality)
        elif self.stage == 2:
            return self.train_stage_2_decomposition_routing(v, a)
        elif self.stage == 3:
            return self.train_stage_3_joint_finetune(v, a)
        raise ValueError(f'Stage {self.stage} not defined. Only 1, 2, 3 are supported.')

    def train_stage_1_vib(self, v, a, train_modality='both'):
        """Stage 1: learn both modalities' intra-modality decompositions."""
        zero = v.new_zeros(1)
        if train_modality == 'both':
            v_out, loss_kl_v, loss_rec_v = self._stage1_vib(
                v, self.IB_m1_tr, self.IB_m1_ir, self.recon_1, self.tr1_fc
            )
            a_out, loss_kl_a, loss_rec_a = self._stage1_vib(
                a, self.IB_m2_tr, self.IB_m2_ir, self.recon_2, self.tr2_fc
            )
            return {
                "v_out": v_out,
                "a_out": a_out,
                "out": (v_out + a_out) / 2,
                "loss_IB": (loss_kl_v, loss_kl_a),
                "loss_rec_input": (loss_rec_v, loss_rec_a),
            }
        if train_modality == 'visual':
            out, loss_kl, loss_rec = self._stage1_vib(v, self.IB_m1_tr, self.IB_m1_ir, self.recon_1, self.tr1_fc)
            return {
                "v_out": out, "a_out": out, "out": out,
                "loss_IB": (loss_kl, zero),
                "loss_rec_input": (loss_rec, zero),
            }
        elif train_modality == 'audio':
            out, loss_kl, loss_rec = self._stage1_vib(a, self.IB_m2_tr, self.IB_m2_ir, self.recon_2, self.tr2_fc)
            return {
                "v_out": out, "a_out": out, "out": out,
                "loss_IB": (zero, loss_kl),
                "loss_rec_input": (zero, loss_rec),
            }
        raise ValueError(
            f"Invalid train_modality: {train_modality}. "
            "Expected 'both', 'visual', or 'audio'."
        )

    def train_stage_2_decomposition_routing(self, v, a):
        """Stage 2: Frozen Stage-1 features → second-level decomposition + dynamic gating."""
        with torch.no_grad():
            z_tr_1_mu, _ = self.IB_m1_tr(v)
            z_tr_2_mu, _ = self.IB_m2_tr(a)
            z_ir_1_mu, z_ir_1_logvar = self.IB_m1_ir(v)
            z_ir_2_mu, z_ir_2_logvar = self.IB_m2_ir(a)
            z_ir_1, _ = self.get_loss(z_ir_1_mu, z_ir_1_logvar, 0.0, 0.0)
            z_ir_2, _ = self.get_loss(z_ir_2_mu, z_ir_2_logvar, 0.0, 0.0)

        outputs = self._second_level_decompose(z_tr_1_mu, z_tr_2_mu, z_ir_1, z_ir_2)
        outputs["v_out"] = self.tr1_fc(z_tr_1_mu)
        outputs["a_out"] = self.tr2_fc(z_tr_2_mu)
        return outputs

    def train_stage_3_joint_finetune(self, v, a):
        """Stage 3: Joint fine-tuning — all parameters trainable, Stage 1 + Stage 2 losses combined."""
        z_tr_1_mu, z_tr_1_logvar = self.IB_m1_tr(v)
        z_tr_2_mu, z_tr_2_logvar = self.IB_m2_tr(a)
        z_ir_1_mu, z_ir_1_logvar = self.IB_m1_ir(v)
        z_ir_2_mu, z_ir_2_logvar = self.IB_m2_ir(a)

        z_tr_1, loss_tr_1 = self.get_loss(z_tr_1_mu, z_tr_1_logvar, 0.0, 0.0)
        z_tr_2, loss_tr_2 = self.get_loss(z_tr_2_mu, z_tr_2_logvar, 0.0, 0.0)
        z_ir_1, loss_ir_1 = self.get_loss(z_ir_1_mu, z_ir_1_logvar, 0.0, 0.0)
        z_ir_2, loss_ir_2 = self.get_loss(z_ir_2_mu, z_ir_2_logvar, 0.0, 0.0)

        rec_1 = self.recon_1(torch.cat((z_tr_1, z_ir_1), dim=1))
        rec_2 = self.recon_2(torch.cat((z_tr_2, z_ir_2), dim=1))

        outputs = self._second_level_decompose(z_tr_1_mu, z_tr_2_mu, z_ir_1, z_ir_2)
        outputs.update({
            "v_out": self.tr1_fc(z_tr_1),
            "a_out": self.tr2_fc(z_tr_2),
            "loss_IB": (loss_tr_1 + loss_ir_1, loss_tr_2 + loss_ir_2),
            "loss_rec_input": (_recon_loss(v, rec_1), _recon_loss(a, rec_2)),
        })
        return outputs
